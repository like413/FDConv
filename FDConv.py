import torch
import torch.fft
import math
import copy
import numpy as np
import megengine
import megengine.functional as F
import megengine.module as nn
from basecls.layers import DropPath, init_weights
from basecls.utils import registers

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Split(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 使用索引直接分割张量
        x1 = x[:, :3, :, :]  # (b, 3, w, h)
        x2 = x[:, 3:, :, :]  # (b, 1, w, h)
        return {3: x1, 1: x2}


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DSConv, self).__init__()
        self.PWC1 = nn.Conv2d(in_channels, out_channels, 1)
        self.DWC = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.PWC2 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.PWC1(x)
        x = self.DWC(x)
        x = self.PWC2(x)
        return x


class Channle_Mix(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(Channle_Mix, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, dim, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(dim, dim // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(dim // ratio, dim, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


def _fuse_prebn_conv1x1(bn, conv):
    module_output = copy.deepcopy(conv)
    module_output.bias = megengine.Parameter(np.zeros(module_output._infer_bias_shape(), dtype=np.float32))
    assert conv.groups == 1
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = F.sqrt(running_var + eps)
    t = (gamma / std).reshape(1, -1, 1, 1)
    module_output.weight[:] = kernel * t
    module_output.bias[:] = F.conv2d(beta - running_mean * gamma / std, kernel, conv.bias)
    return module_output


def _fuse_conv_bn(conv, bn):
    module_output = copy.deepcopy(conv)
    module_output.bias = megengine.Parameter(np.zeros(module_output._infer_bias_shape(), dtype=np.float32))
    # flatten then reshape in case of group conv
    kernel = F.flatten(conv.weight, end_axis=conv.weight.ndim - 4)
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = F.sqrt(running_var + eps)
    t = (gamma / std).reshape(-1, 1, 1, 1)
    module_output.weight[:] = (kernel * t).reshape(module_output.weight.shape)
    module_output.bias[:] = beta + ((conv.bias if conv.bias is not None else 0) - running_mean) * gamma / std
    return module_output


class ConvBn2d(nn.ConvBn2d):
    def __init__(self, *args, **kwargs):
        bias = kwargs.pop("bias", False) and False
        super().__init__(*args, bias=bias, **kwargs)

    @classmethod
    def fuse_conv_bn(cls, module: nn.Module):
        module_output = module
        if isinstance(module, ConvBn2d):
            return _fuse_conv_bn(module.conv, module.bn)
        for name, child in module.named_children():
            setattr(module_output, name, cls.fuse_conv_bn(child))
        del module
        return module_output


class Rep_Convs(nn.Module):
    def __init__(self, channels, kernel, small_kernels=()):
        super(Rep_Convs, self).__init__()
        self.dw_large = ConvBn2d(channels, channels, kernel, padding=kernel // 2, groups=channels)

        self.small_kernels = small_kernels
        for k in self.small_kernels:
            setattr(self, f"dw_small_{k}", ConvBn2d(channels, channels, k, padding=k // 2, groups=channels))

    def forward(self, inp):
        outp = self.dw_large(inp)
        for k in self.small_kernels:
            outp += getattr(self, f"dw_small_{k}")(inp)
        return outp

    @classmethod
    def convert_to_deploy(cls, module: nn.Module):
        module_output = module
        if isinstance(module, Rep_Convs):
            module = ConvBn2d.fuse_conv_bn(module)
            module_output = copy.deepcopy(module.dw_large)
            kernel = module_output.kernel_size[0]
            for k in module.small_kernels:
                dw_small = getattr(module, f"dw_small_{k}")
                module_output.weight += F.pad(dw_small.weight, [[0, 0]] * 3 + [[(kernel - k) // 2] * 2] * 2)
                module_output.bias += dw_small.bias
            return module_output
        for name, child in module.named_children():
            setattr(module_output, name, cls.convert_to_deploy(child))
        del module
        return module_output


class LFRModule(nn.Module):

    def __init__(self, channels, kernel=7, small_kernels=(3,), activation=nn.ReLU):
        super().__init__()

        self.pre_bn = nn.BatchNorm2d(channels)
        self.dw = Rep_Convs(int(channels), kernel, small_kernels=small_kernels)
        self.dw_act = activation()
        self.cm = Channle_Mix(channels)

    def forward(self, x):
        y = self.pre_bn(x)
        y = self.dw_act(self.dw(y))
        x = x + y
        x = self.cm(x) + x
        return x

    def convert_to_deploy(cls, module: nn.Module):
        module_output = module
        if isinstance(module, LFRModule):
            Rep_Convs.convert_to_deploy(module)
            ConvBn2d.fuse_conv_bn(module)

            module.pre_bn, module.pw1 = nn.Identity(), _fuse_prebn_conv1x1(module.pre_bn, module.pw1)
            module.premlp_bn, module.mlp.fc1 = nn.Identity(), _fuse_prebn_conv1x1(module.premlp_bn, module.mlp.fc1)
            return module_output
        for name, child in module.named_children():
            setattr(module_output, name, cls.convert_to_deploy(child))
        del module
        return module_output


def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class DCTLayer(nn.Module):
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(DCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        # 通道注意力
        # result = torch.sum(x, dim=[2, 3])
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        result = torch.cat([avg_out, max_out], dim=1)

        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter


class FrequencyRefine(nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=8, freq_sel_method='top8'):
        super().__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = DCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        # 通道注意力
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel, bias=False),
        #     nn.Sigmoid()
        # )
        # 空间注意力
        kernel_size = 7
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        # y = self.fc(y).view(n, c, 1, 1)
        # return x * y.expand_as(x)
        y = self.conv(y)
        return x * y + x


class HFAModule(nn.Module):
    def __init__(self, channels, size, groups=1):
        super().__init__()
        self.c1 = Conv(channels, channels // 2, 1, g=groups)
        self.fu = FrequencyRefine(channels // 2, size, size)
        self.c2 = nn.Conv2d(channels // 2, channels, 1, groups=groups)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.fu(x1)
        x3 = self.c2(x1 + x2)
        return x3


class FDConv(nn.Module):

    def __init__(self, in_channels, size, out_channels=None, kernel_size=3,
                 ratio_gin=0.5, ratio_gout=0.5, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        self.cl = nn.Conv2d(in_channels, in_cl, 1)
        self.cg = nn.Conv2d(in_channels, in_cg, 1)
        self.cout = nn.Conv2d(out_channels, out_channels, 1)
        # l2l branch
        module = nn.Identity if in_cl == 0 or out_cl == 0 else LFRModule
        self.convl2l = module(in_cl)
        # cross branch
        module = nn.Identity if in_cl == 0 or out_cl == 0 else DSConv
        self.convl2g = module(in_cl, out_cg, kernel_size, stride, padding, dilation)
        self.convg2l = module(in_cg, out_cl, kernel_size, stride, padding, dilation)
        # h2h branch
        module = nn.Identity if in_cg == 0 or out_cg == 0 else HFAModule
        self.convg2g = module(in_cg, size)

        self.fc = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):

        x_l, x_g = self.cl(x), self.cg(x)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            a = self.convl2l(x_l)
            b = self.convg2l(x_g)
            out_xl = a + b
        if self.ratio_gout != 0:
            a = self.convl2g(x_l)
            b = self.convg2g(x_g)
            out_xg = a + b

        out = self.cout(torch.cat([out_xl, out_xg], dim=1))
        res = out + x
        out = self.fc(out)
        return out + res


class LightMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.compress_conv = nn.Conv2d(in_features, in_features // 2, kernel_size=1)
        self.in_features = in_features
        self.fc1 = nn.Conv2d(in_features // 2, hidden_features // 2, 1)
        self.dwconv = DWConv(hidden_features // 2)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features // 2, out_features // 2, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.compress_conv(x)
        Identity = x.clone()
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = torch.cat([x, Identity], dim=1)
        x = self.drop(x)
        return x


class FDBlock(nn.Module):
    def __init__(self, dim, size, mlp_ratio=4., drop=0., drop_p=0., act_layer=nn.GELU, ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = FDConv(dim, size)
        self.drop_path = DropPath(drop_p) if drop_p > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LightMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
