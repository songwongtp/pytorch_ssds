import torch
import torch.nn as nn

from collections import namedtuple
import functools
import torch.nn.functional as F

Conv = namedtuple('Conv', ['stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['stride', 'depth'])
InvertedResidual = namedtuple('InvertedResidual', ['stride', 'depth', 'num', 't']) # t is the expension factor
InvertedAttResidual = namedtuple('InvertedAttResidual', ['stride', 'depth', 'num', 't', 'exp_group', 'pro_group', 'reduction_ratio']) # t is the expension factor

V1_CONV_DEFS = [
    Conv(stride=2, depth=32),
    DepthSepConv(stride=1, depth=64),
    DepthSepConv(stride=2, depth=128),
    DepthSepConv(stride=1, depth=128),
    DepthSepConv(stride=2, depth=256),
    DepthSepConv(stride=1, depth=256),
    DepthSepConv(stride=2, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=2, depth=1024),
    DepthSepConv(stride=1, depth=1024)
]

V2_CONV_DEFS = [
    Conv(stride=2, depth=32),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=24, num=2, t=6),
    InvertedResidual(stride=2, depth=32, num=3, t=6),
    InvertedResidual(stride=2, depth=64, num=4, t=6),
    InvertedResidual(stride=1, depth=96, num=3, t=6),
    InvertedResidual(stride=2, depth=160, num=3, t=6),
    InvertedResidual(stride=1, depth=320, num=1, t=6),
]

V3_CONV_DEFS = [
    Conv(stride=2, depth=32),
    InvertedAttResidual(stride=1, depth=16, num=1, t=1, exp_group=1, pro_group=1, reduction_ratio=16),
    InvertedAttResidual(stride=2, depth=24, num=2, t=6, exp_group=1, pro_group=1, reduction_ratio=16),
    InvertedAttResidual(stride=2, depth=32, num=3, t=6, exp_group=1, pro_group=1, reduction_ratio=16),
    InvertedAttResidual(stride=2, depth=64, num=4, t=6, exp_group=1, pro_group=1, reduction_ratio=16),
    InvertedAttResidual(stride=1, depth=96, num=3, t=6, exp_group=1, pro_group=1, reduction_ratio=16),
    InvertedAttResidual(stride=2, depth=160, num=3, t=6, exp_group=1, pro_group=1, reduction_ratio=16),
    InvertedAttResidual(stride=1, depth=320, num=1, t=6, exp_group=1, pro_group=1, reduction_ratio=16),
]

class _conv_bn(nn.Module):
    def __init__(self, inp, oup, stride):
        super(_conv_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)


class _conv_dw(nn.Module):
    def __init__(self, inp, oup, stride):
        super(_conv_dw, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)


class _inverted_residual_bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(_inverted_residual_bottleneck, self).__init__()
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.depth = oup
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class _inverted_attresidual_bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, exp_group, pro_group, reduction_ratio):
        super(_inverted_attresidual_bottleneck, self).__init__()
        self.use_res_connect = stride == 1 and inp == oup
        self.conv1 = nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, groups = exp_group, bias=False)
        self.bn1 = nn.BatchNorm2d(inp * expand_ratio)
        self.relu1 = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False)
        self.bn2 = nn.BatchNorm2d(inp * expand_ratio)
        self.relu2 = nn.ReLU6(inplace=True)
        self.conv3 = nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, groups=pro_group, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)
        self.depth = oup

        #channel-wise and spatial-wise attention
        self.wblock = nn.Parameter(torch.zeros(2))
        self.fc1 = nn.Conv2d(inp * expand_ratio, int(inp * expand_ratio//reduction_ratio), kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(int(inp * expand_ratio//reduction_ratio), inp * expand_ratio, kernel_size=1)
        self.sconv = nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel_size=3, stride=1, padding=1, groups=inp * expand_ratio, bias=False)

    def forward(self, x):
        softmx = nn.Softmax(0)
        mixture = softmx(self.wblock)

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.sconv(out) * mixture[0] + self.fc2(w) * mixture[1])
        # Excitation
        out = out * w

        out = self.bn3(self.conv3(out))
        if self.use_res_connect:
            return x + out
        else:
            return out



def mobilenet(conv_defs, depth_multiplier=1.0, min_depth=8):
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    layers = []
    in_channels = 3
    for conv_def in conv_defs:
        if isinstance(conv_def, Conv):
            layers += [_conv_bn(in_channels, depth(conv_def.depth), conv_def.stride)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, DepthSepConv):
            layers += [_conv_dw(in_channels, depth(conv_def.depth), conv_def.stride)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, InvertedResidual):
          for n in range(conv_def.num):
            stride = conv_def.stride if n == 0 else 1
            layers += [_inverted_residual_bottleneck(in_channels, depth(conv_def.depth), stride, conv_def.t)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, InvertedAttResidual):
          for n in range(conv_def.num):
            stride = conv_def.stride if n == 0 else 1
            layers += [_inverted_attresidual_bottleneck(in_channels, depth(conv_def.depth), stride, conv_def.t, conv_def.exp_group, conv_def.pro_group, conv_def.reduction_ratio)]
            in_channels = depth(conv_def.depth)
    return layers

def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func

mobilenet_v1 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=1.0)
mobilenet_v1_075 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=0.75)
mobilenet_v1_050 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=0.50)
mobilenet_v1_025 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=0.25)

mobilenet_v2 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=1.0)
mobilenet_v2_075 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=0.75)
mobilenet_v2_050 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=0.50)
mobilenet_v2_025 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=0.25)

mobilenet_att = wrapped_partial(mobilenet, conv_defs=V3_CONV_DEFS, depth_multiplier=1.0)
mobilenet_att_075 = wrapped_partial(mobilenet, conv_defs=V3_CONV_DEFS, depth_multiplier=0.75)
mobilenet_att_050 = wrapped_partial(mobilenet, conv_defs=V3_CONV_DEFS, depth_multiplier=0.50)
mobilenet_att_025 = wrapped_partial(mobilenet, conv_defs=V3_CONV_DEFS, depth_multiplier=0.25)