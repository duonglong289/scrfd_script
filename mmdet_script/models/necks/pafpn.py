import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn import FPN
from typing import List, Tuple

class ConvModule(nn.Module):

    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias # = True

        # reset padding to 0 for conv module
        conv_padding = padding
        
        # build convolution layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed # False
        self.output_padding = self.conv.output_padding # (0, 0)
        self.groups = self.conv.groups # 1


    def forward(self, x):
        x = self.conv(x)
        return x


class PAFPN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(PAFPN,
              self).__init__(in_channels, out_channels, num_outs, start_level,
                             end_level, add_extra_convs, extra_convs_on_inputs,
                             relu_before_extra_convs, no_norm_on_lateral,
                             conv_cfg, norm_cfg, act_cfg)
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals: List[torch.Tensor] = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels: int = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels
        inter_outs = []
        
        for ind, layer_fpn in enumerate(self.fpn_convs):
            res = layer_fpn(laterals[ind])
            inter_outs.append(res)
            
        
        # part 2: add bottom-up path
        # for ind in range(0, used_backbone_levels - 1):
        #     inter_outs[ind + 1] += self.downsample_convs[ind](inter_outs[ind])
        for ind, down_module in enumerate(self.downsample_convs):
            inter_outs[ind + 1] += down_module(inter_outs[ind])

        outs = []
        outs.append(inter_outs[0])
        
        
        for ind, layer_pafpn in enumerate(self.pafpn_convs):
            # res = self.pafpn_convs[i - 1](inter_outs[i])
            res = layer_pafpn(inter_outs[ind+1])
            outs.append(res)
        
        # # part 3: add extra levels
        # if self.num_outs > len(outs):
        #     # use max pool to get more levels on top of outputs
        #     # (e.g., Faster R-CNN, Mask R-CNN)
            
        #     if not self.add_extra_convs:
        #         for i in range(self.num_outs - used_backbone_levels):
        #             outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        #     # add conv layers on top of original feature maps (RetinaNet)
        #     else:
        #         if self.add_extra_convs == 'on_input':
        #             orig = inputs[self.backbone_end_level - 1]
        #             outs.append(self.fpn_convs[used_backbone_levels](orig))
        #         elif self.add_extra_convs == 'on_lateral':
        #             outs.append(self.fpn_convs[used_backbone_levels](
        #                 laterals[-1]))
        #         elif self.add_extra_convs == 'on_output':
        #             outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
        #         else:
        #             raise NotImplementedError
        #         for i in range(used_backbone_levels + 1, self.num_outs):
        #             if self.relu_before_extra_convs:
        #                 outs.append(self.fpn_convs[i](F.relu(outs[-1])))
        #             else:
        #                 outs.append(self.fpn_convs[i](outs[-1]))
        # return tuple(outs)
        return outs
