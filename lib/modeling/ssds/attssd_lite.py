import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os

from lib.layers import *

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

    def __init__(self, base, extras, head, feature_layer, num_classes):
        super(AttSSDLite, self).__init__()
        self.num_classes = num_classes
        # SSD network
        self.base = nn.ModuleList(base)
        self.norm = L2Norm(feature_layer[1][0], 20)
        self.extras = nn.ModuleList(extras)

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
        sources = list()
        loc = list()
        conf = list()

        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                if len(sources) == 0:
                    s = self.norm(x)
                    sources.append(s)
                else:
                    sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            sources.append(x)
            # if k % 2 == 1:
            #     sources.append(x)

        if phase == 'feature':
            return sources

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
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

def add_extras(base, feature_layer, mbox, num_classes):
    extra_layers = []
    loc_layers = []
    conf_layers = []
    in_channels = None
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
    return base, extra_layers, (loc_layers, conf_layers)

class AttentionBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(AttentionBlock, self).__init__()
        self.out_channels = channels

        #self.wblock = nn.Parameter(torch.zeros(2))
        self.fc1 = nn.Conv2d(channels, int(channels//reduction_ratio), kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(int(channels//reduction_ratio), channels, kernel_size=1)
        #self.sconv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
    def forward(self, x):
        #softmx = nn.Softmax(0)
        #mixture = softmx(self.wblock)

        w = F.avg_pool2d(x, x.size(2))
        w = F.relu(self.fc1(w))
        #w = F.sigmoid(self.sconv(x) * mixture[0] + self.fc2(w) * mixture[1])
        w = F.sigmoid(self.fc2(w))
        # Excitation
        return x * w

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
        # attention
        AttentionBlock(oup * expand_ratio),
        # pw-linear
        nn.Conv2d(oup * expand_ratio, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )

def build_attssd_lite(base, feature_layer, mbox, num_classes):
    base_, extras_, head_ = add_extras(base(), feature_layer, mbox, num_classes)
    return AttSSDLite(base_, extras_, head_, feature_layer, num_classes)