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

class AttFSSDLite(nn.Module):
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

    def __init__(self, base, extras, attentions, head, features, feature_layer, num_classes):
        super(AttFSSDLite, self).__init__()
        self.num_classes = num_classes
        # SSD network
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.att_wblocks = nn.ParameterList(attentions[0])
        self.att_fc1 = attentions[1]
        self.att_fc2 = attentions[2]
        self.att_sconvs = nn.ModuleList(attentions[3])
        self.feature_layer = feature_layer[0][0]
        #self.transforms = nn.ModuleList(features[0])
        self.pyramids = nn.ModuleList(features[1])
        # print(self.base)
        #self.norm = nn.BatchNorm2d(int(feature_layer[0][1][-1]/2)*len(self.transforms),affine=True)
        # print(self.extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        # print(self.loc)

        self.softmax = nn.Softmax(dim=-1)
        

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
        sources, pooled_tensors, attentions, transformed, pyramids, loc, conf = [list() for _ in range(7)]

        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            sources.append(x)
            # if k % 2 == 1:
            #     sources.append(x)

        upsize = (sources[0].size()[2], sources[0].size()[3])
        channel_sizes = []
        for k in range(len(sources)):
            if k > 0:
                sources[k] = F.upsample(sources[k], size=upsize, mode='bilinear')
            # Squeeze
            pooled_tensors.append(F.avg_pool2d(sources[k], sources[k].size(2)))
            channel_sizes.append(sources[k].size(1))

        # Channel Attention + Mixture
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
            
        #assert len(self.transforms) == len(attentions)
        
        #for k, v in enumerate(self.transforms):
        #    size = None if k == 0 else upsize
        #    transformed.append(v(attentions[k], size))
        #x = torch.cat(transformed, 1)
        x = torch.cat(attentions, 1)
        #x = self.norm(x)
        for k, v in enumerate(self.pyramids):
            x = v(x)
            pyramids.append(x)

        if phase == 'feature':
            return pyramids

        # apply multibox head to pyramids layers
        for (x, l, c) in zip(pyramids, self.loc, self.conf):
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
    feature_transform_layers = []
    pyramid_feature_layers = []
    loc_layers = []
    conf_layers = []

    # ----------------- attention extra -------------------- #
    att_wblocks = []
    att_fc1 = None
    att_fc2 = None
    att_sconvs = []
    # ------------------------------------------------------ #

    in_channels = None
    #feature_transform_channel = int(feature_layer[0][1][-1]/2)
    concat_in_channels = 0
    for layer, depth in zip(feature_layer[0][0], feature_layer[0][1]):
        if layer == 'S':
            extra_layers += [ _conv_dw(in_channels, depth, stride=2, padding=1, expand_ratio=1) ]
            in_channels = depth
        elif layer == '':
            extra_layers += [ _conv_dw(in_channels, depth, stride=1, expand_ratio=1) ]
            in_channels = depth
        else:
            in_channels = depth

        concat_in_channels += in_channels
        att_wblocks += [nn.Parameter(torch.zeros(2))]
        att_sconvs += [nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)]

        #feature_transform_layers += [BasicConv(in_channels, feature_transform_channel, kernel_size=1, padding=0)]
    
    att_fc1 = nn.Conv2d(concat_in_channels, int(concat_in_channels//reduction_ratio), kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
    att_fc2 = nn.Conv2d(int(concat_in_channels//reduction_ratio), concat_in_channels, kernel_size=1)

    #in_channels = len(feature_transform_layers) * feature_transform_channel
    in_channels = int(sum(feature_layer[0][1]))
    for layer, depth, box in zip(feature_layer[1][0], feature_layer[1][1], mbox):
        if layer == 'S':
            pyramid_feature_layers += [BasicConv(in_channels, depth, kernel_size=3, stride=2, padding=1)]
            in_channels = depth
        elif layer == '':
            pad = (0,1)[len(pyramid_feature_layers)==0]
            pyramid_feature_layers += [BasicConv(in_channels, depth, kernel_size=3, stride=1, padding=pad)]
            in_channels = depth
        else:
            AssertionError('Undefined layer')
        loc_layers += [nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1)]

    return base, extra_layers, (att_wblocks, att_fc1, att_fc2, att_sconvs), (feature_transform_layers, pyramid_feature_layers), (loc_layers, conf_layers)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        # self.up_size = up_size
        # self.up_sample = nn.Upsample(size=(up_size,up_size),mode='bilinear') if up_size != 0 else None

    def forward(self, x, up_size=None):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if up_size is not None:
            x = F.upsample(x, size=up_size, mode='bilinear')
            # x = self.up_sample(x)
        return x

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

def build_attfssd_lite(base, feature_layer, mbox, num_classes):
    base_, extras_, attentions_, features_, head_ = add_extras(base(), feature_layer, mbox, num_classes)
    return AttFSSDLite(base_, extras_, attentions_, head_, features_, feature_layer, num_classes)