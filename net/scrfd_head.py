import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.

    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.

            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.

    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr or torch.onnx.is_in_onnx_export():
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        # -1 indexing works abnormal in TensorRT
        # This assumes `dets` has 5 dimensions where
        # the last dimension is score.
        # TODO: more elegant way to handle the dimension issue.
        scores = dets[:, 4]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (bboxes, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]
    if score_factors is not None:
        scores = scores * score_factors[:, None]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    # remove low scoring boxes
    valid_mask = scores > score_thr
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    if inds.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels

    # TODO: add size check before feed into batched_nms
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], keep
    else:
        return dets, labels[keep]

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

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

        self.with_norm: bool = norm_cfg is not None
        self.with_activation: bool = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
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
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups


        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            
            self.bn = nn.BatchNorm2d(norm_channels)
        else:
            self.norm_name = None

        # build activation layer
        if self.with_activation:
            self.activate = nn.ReLU(inplace=inplace)



    def forward(self, x, activate:bool=True, norm:bool=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.bn(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x

@torch.jit.script
class AnchorGenerator(object):

    def __init__(self,
                 strides: List[int],
                 ratios: List[float],
                 scales: List[int],
                 base_sizes: List[int],
                 scale_major: bool=True,
                #  centers=None
                 center_offset:float=0.):

        # calculate base sizes of anchors
        # self.strides = [_pair(stride) for stride in strides]
        self.strides = [(stride, stride) for stride in strides]
        self.base_sizes = base_sizes
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'


        self.scales = torch.tensor(scales)
        self.ratios = torch.tensor(ratios)
        self.scale_major = scale_major
        # self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    def gen_base_anchors(self):
        """Generate base anchors.

        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios))
                    # center=center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size: int,
                                      scales,
                                      ratios):
                                    #   center=None):
        """Generate base anchors of a single level.

        Args:
            base_size (int | float): Basic size of an anchor.
            scales (torch.Tensor): Scales of the anchor.
            ratios (torch.Tensor): The ratio between between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        w = base_size
        h = base_size

        # if center is None:
        x_center = self.center_offset * w
        y_center = self.center_offset * h

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16): 
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project', torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x

class SCRFDHead(nn.Module):
    """Generalized Focal Loss: Learning Qualified and Distributed Bounding
    Boxes for Dense Object Detection.

    GFL head structure is similar with ATSS, however GFL uses
    1) joint representation for classification and localization quality, and
    2) flexible General distribution for bounding box locations,
    which are supervised by
    Quality Focal Loss (QFL) and Distribution Focal Loss (DFL), respectively

    https://arxiv.org/abs/2006.04388

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 4.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='GN', num_groups=32, requires_grad=True).
        loss_qfl (dict): Config of Quality Focal Loss (QFL).
        reg_max (int): Max value of integral set :math: `{0, ..., reg_max}`
            in QFL setting. Default: 16.
    Example:
        >>> self = GFLHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_quality_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_quality_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 feat_mults=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 reg_max=8,
                 cls_reg_share=False,
                 strides_share=True,
                 scale_mode = 1,
                 dw_conv = False,
                 use_kps = True,
                 feat_channels=80):
                
        self.stacked_convs = stacked_convs
        self.feat_mults = feat_mults
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        self.cls_reg_share = cls_reg_share
        self.strides_share = strides_share
        self.scale_mode = scale_mode
        self.dw_conv = dw_conv
        self.NK = 5
        self.extra_flops = 0.0
        self.use_scale = False
        self.use_kps = use_kps
        if self.scale_mode>0 and (self.strides_share or self.scale_mode==2):
            self.use_scale = True
            
        #kwargs ={'feat_channels': 80, 'anchor_generator': {'type': 'AnchorGenerator', 'ratios': [1.0], 'scales': [1, 2], 'base_sizes': [16, 64, 256], 'strides': [8, 16, 32]}, 'loss_cls': {'type': 'QualityFocalLoss', 'use_sigmoid': True, 'beta': 2.0, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'DIoULoss', 'loss_weight': 2.0}, 'test_cfg': {'nms_pre': -1, 'min_bbox_size': 0, 'score_thr': 0.02, 'nms': {'type': 'nms', 'iou_threshold': 0.45}, 'max_per_img': -1}}
        # super(SCRFDHead, self).__init__(num_classes, in_channels, **kwargs)
        super(SCRFDHead, self).__init__()
        
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.cls_out_channels = num_classes

        self.sampling = False

        self.integral = Integral(self.reg_max)
        
        # {'type': 'AnchorGenerator', 'ratios': [1.0], 'scales': [1, 2], 'base_sizes': [16, 64, 256], 'strides': [8, 16, 32]}
        self.anchor_generator = AnchorGenerator(
            ratios=[1.0],
            scales=[1, 2],
            base_sizes=[16, 64, 256],
            strides=[8, 16, 32]
        )
        # # usually the numbers of anchors for each level are the same
        # # except SSD detectors
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self._init_layers()
        

    def _get_conv_module(self, in_channel, out_channel):
        conv = ConvModule(
                in_channel,
                out_channel,
                3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
    
        return conv
    
    def call_cls_stride_convs(self, x: torch.Tensor, stride: str) -> torch.Tensor:
        res: torch.Tensor = x
        for key, module in self.cls_stride_convs.items():
            if key == stride:
                res = module(x)
                return res

        # raise Exception("Key not Found: {}".format(stride))
        return res
        
    def call_stride_cls(self, x: torch.Tensor, stride: str) -> torch.Tensor:
        res: torch.Tensor = x
        for key, module in self.stride_cls.items():
            if key == stride:
                res = module(x)
                return res
        # raise Exception("Key not Found: {}".format(stride))
        return res

    def call_stride_reg(self, x: torch.Tensor, stride: str) -> torch.Tensor:
        res: torch.Tensor = x
        for key, module in self.stride_reg.items():
            if key == stride:
                res = module(x)
                return res
        # raise Exception("Key not Found: {}".format(stride))
        return res
    
    def call_stride_kps(self, x, stride: str):
        res: torch.Tensor = x
        for key, module in self.stride_kps.items():
            if key == stride:
                res = module(x)
                return res
        # raise Exception("Key not Found: {}".format(stride))
        return res

    def call_scale(self, x: torch.Tensor, index: int) -> torch.Tensor:
        res: torch.Tensor = x
        for ind, module in enumerate(self.scales):
            if ind == index:
                res = module(x)
                return res
            
        return res
    
    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        conv_strides = [0] if self.strides_share else self.anchor_generator.strides # [(8, 8), (16, 16), (32, 32)]
        self.cls_stride_convs = nn.ModuleDict()
        self.reg_stride_convs = nn.ModuleDict()
        self.stride_cls = nn.ModuleDict()
        self.stride_reg = nn.ModuleDict()
        if self.use_kps:
            self.stride_kps = nn.ModuleDict()
        
        for stride_idx, conv_stride in enumerate(conv_strides):
            key = str(conv_stride)
            # cls_convs = nn.ModuleList()
            list_cls_convs = []
            # reg_convs = nn.ModuleList()
            list_reg_convs = []
            
            stacked_convs = self.stacked_convs
            feat_mult = 1
            feat_ch = int(self.feat_channels*feat_mult)

            for i in range(stacked_convs):
                chn = self.in_channels if i == 0 else last_feat_ch
                # cls_convs.append( self._get_conv_module(chn, feat_ch) )
                list_cls_convs.append( self._get_conv_module(chn, feat_ch) )
                if not self.cls_reg_share:
                    # reg_convs.append( self._get_conv_module(chn, feat_ch) )
                    list_reg_convs.append( self._get_conv_module(chn, feat_ch) )
                last_feat_ch = feat_ch
            
            cls_convs = nn.Sequential(*list_cls_convs)
            reg_convs = nn.Sequential(*list_reg_convs)
            self.cls_stride_convs[key] = cls_convs
            self.reg_stride_convs[key] = reg_convs
            self.stride_cls[key] = nn.Conv2d(
                feat_ch, self.cls_out_channels * self.num_anchors, 3, padding=1)

            self.stride_reg[key] = nn.Conv2d(
                feat_ch, 4 * self.num_anchors, 3, padding=1)
            if self.use_kps:
                self.stride_kps[key] = nn.Conv2d(
                    feat_ch, self.NK*2*self.num_anchors, 3, padding=1)
        
        if self.use_scale:
            self.scales = nn.ModuleList(
                [Scale(1.0) for _ in self.anchor_generator.strides])
        else:
            self.scales = [None for _ in self.anchor_generator.strides]

    def forward(self, feats: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification and quality (IoU)
                    joint scores for all scale levels, each is a 4D-tensor,
                    the channel number is num_classes.
                bbox_preds (list[Tensor]): Box distribution logits for all
                    scale levels, each is a 4D-tensor, the channel number is
                    4*(n+1), n is max value of integral set.
        """
        res_8: List[torch.Tensor] = []
        res_16: List[torch.Tensor] = []
        res_32: List[torch.Tensor] = []
        for index in range(len(feats)):
            res = self.forward_single(feats[index], index)
            res_8.append(res[0])
            res_16.append(res[1])
            res_32.append(res[2])
            
        results = [res_8, res_16, res_32]
        return results

    def forward_single(self, x, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls and quality joint scores for a single
                    scale level the channel number is num_classes.
                bbox_pred (Tensor): Box distribution logits for a single scale
                    level, the channel number is 4*(n+1), n is max value of
                    integral set.
        """
        cls_feat = x
        reg_feat = x
        stride = self.anchor_generator.strides[index]
        
        cls_feat = self.call_cls_stride_convs(cls_feat, str(stride))
        reg_feat = cls_feat
        
        cls_score = self.call_stride_cls(cls_feat, str(stride))
        _bbox_pred = self.call_stride_reg(reg_feat, str(stride))
        
        bbox_pred = self.call_scale(_bbox_pred, index)
        kps_pred = self.call_stride_kps(reg_feat, str(stride))

        return cls_score, bbox_pred, kps_pred

    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)


    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
