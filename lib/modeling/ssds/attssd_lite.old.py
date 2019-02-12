import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os

from lib.layers import *

def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.
    
    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    
    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes does not equal tensor dim")
    
    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return [tensor.narrow(int(dim), int(start), int(length)) 
                for start, length in zip(splits, split_sizes)]

class AttSSDLite(nn.Module):
    """Single Shot Multibox Architecture for embeded system
    See: https://arxiv.org/pdf/1512.02325.pdf & 
    https://arxiv.org/pdf/1801.04381.pdf for more details.

    Args:
        phase: (string) Can be "eval" or "train" or "feature"
        base: base layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        feature_layer: the feature layers for head to loc and conf
        num_classes: num of classes 
    """

    def __init__(self, base, extras, attentions, head, feature_layer, num_classes):
        super(AttSSDLite, self).__init__()
        self.num_classes = num_classes
        # SSD network
        self.base = nn.ModuleList(base)
        self.norm = L2Norm(feature_layer[1][0], 20)
        self.extras = nn.ModuleList(extras)

        # ----------------- ATTENTION ----------------- #
        self.att_wblocks = nn.ParameterList(attentions[0])
        self.att_fc1 = attentions[1]
        self.att_fc2 = attentions[2]
        self.att_sconvs = nn.ModuleList(attentions[3])
        # --------------------------------------------- #

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)
        
        self.feature_layer = feature_layer[0]

    def forward(self, x, phase='eval'):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]

            feature:
                the features maps of the feature extractor
        """
        sources, pooled_tensors, attentions, channel_sizes, loc, conf = [list() for _ in range(6)]

        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                if len(sources) == 0:
                    s = self.norm(x)
                    sources.append(s)

                    # ----------------- ATTENTION ----------------- #
                    pooled_tensors.append(F.avg_pool2d(s, s.size(2)))
                    channel_sizes.append(s.size(1))
                    # --------------------------------------------- #
                else:
                    sources.append(x)

                    # ----------------- ATTENTION ----------------- #
                    pooled_tensors.append(F.avg_pool2d(x, x.size(2)))
                    channel_sizes.append(x.size(1))
                    # --------------------------------------------- #

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            sources.append(x)
            # if k % 2 == 1:
            #     sources.append(x)

            # ----------------- ATTENTION ----------------- #
            # Squeeze
            pooled_tensors.append(F.avg_pool2d(x, x.size(2)))
            channel_sizes.append(x.size(1))
            # --------------------------------------------- #

        # ----------------- ATTENTION ----------------- #
        # Channel Attention
        x = F.relu(self.att_fc1(torch.cat(pooled_tensors, dim=1)))
        x = size_splits(self.att_fc2(x), channel_sizes, dim=1)
        
        assert len(x) == len(sources)
        softmx = nn.Softmax(0)
        for k in range(len(sources)):
            mixture = softmx(self.att_wblocks[k])

            # Spatial Attention + Mixture
            x[k] = self.att_sconvs[k](sources[k])*mixture[0] + x[k]*mixture[1]
            x[k] = F.sigmoid(x[k])
            # Excitation
            attentions.append(sources[k] * x[k])
        # --------------------------------------------- #

        if phase == 'feature':
            return attentions

        # apply multibox head to attentions layers
        for (x, l, c) in zip(attentions, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if phase == 'eval':
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

def add_extras(base, feature_layer, mbox, num_classes, reduction_ratio=16):
    extra_layers = []
    loc_layers = []
    conf_layers = []

    # -------------------- ATTENTION ----------------------- #
    att_wblocks = []
    att_fc1 = None
    att_fc2 = None
    att_sconvs = []
    # ------------------------------------------------------ #

    in_channels = None
    concat_in_channels = 0
    for layer, depth, box in zip(feature_layer[0], feature_layer[1], mbox):
        if layer == 'S':
            extra_layers += [ _conv_dw(in_channels, depth, stride=2, padding=1, expand_ratio=1) ]
            in_channels = depth
        elif layer == '':
            extra_layers += [ _conv_dw(in_channels, depth, stride=1, expand_ratio=1) ]
            in_channels = depth
        else:
            in_channels = depth
        loc_layers += [nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1)]
    
        # -------------------- ATTENTION ----------------------- #
        concat_in_channels += in_channels
        att_wblocks += [nn.Parameter(torch.zeros(2))]
        att_sconvs += [nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)]
        # ------------------------------------------------------ #

    # -------------------- ATTENTION ----------------------- #
    att_fc1 = nn.Conv2d(concat_in_channels, int(concat_in_channels//reduction_ratio), kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
    att_fc2 = nn.Conv2d(int(concat_in_channels//reduction_ratio), concat_in_channels, kernel_size=1)
    # ------------------------------------------------------ #

    return base, extra_layers, (att_wblocks, att_fc1, att_fc2, att_sconvs), (loc_layers, conf_layers)

# based on the implementation in https://github.com/tensorflow/models/blob/master/research/object_detection/models/feature_map_generators.py#L213
# when the expand_ratio is 1, the implemetation is nearly same. Since the shape is always change, I do not add the shortcut as what mobilenetv2 did.
def _conv_dw(inp, oup, stride=1, padding=0, expand_ratio=1):
    return nn.Sequential(
        # pw
        nn.Conv2d(inp, oup * expand_ratio, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # dw
        nn.Conv2d(oup * expand_ratio, oup * expand_ratio, 3, stride, padding, groups=oup * expand_ratio, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(oup * expand_ratio, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )

def build_attssd_lite(base, feature_layer, mbox, num_classes):
    base_, extras_, attentions_, head_ = add_extras(base(), feature_layer, mbox, num_classes)
    return AttSSDLite(base_, extras_, attentions_, head_, feature_layer, num_classes)