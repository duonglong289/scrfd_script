from functools import partial

import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint


def generate_inputs_and_wrap_model(config_path, checkpoint_path, input_config):
    """Prepare sample input and wrap model for ONNX export.

    The ONNX export API only accept args, and all inputs should be
    torch.Tensor or corresponding types (such as tuple of tensor).
    So we should call this function before exporting. This function will:

    1. generate corresponding inputs which are used to execute the model.
    2. Wrap the model's forward function.

    For example, the MMDet models' forward function has a parameter
    ``return_loss:bool``. As we want to set it as False while export API
    supports neither bool type or kwargs. So we have to replace the forward
    like: ``model.forward = partial(model.forward, return_loss=False)``

    Args:
        config_path (str): the OpenMMLab config for the model we want to
            export to ONNX
        checkpoint_path (str): Path to the corresponding checkpoint
        input_config (dict): the exactly data in this dict depends on the
            framework. For MMSeg, we can just declare the input shape,
            and generate the dummy data accordingly. However, for MMDet,
            we may pass the real img path, or the NMS will return None
            as there is no legal bbox.

    Returns:
        tuple: (model, tensor_data) wrapped model which can be called by \
        model(*tensor_data) and a list of inputs which are used to execute \
            the model while exporting.
    """

    model = build_model_from_cfg(config_path, checkpoint_path)
    # import ipdb; ipdb.set_trace()
    one_img, one_meta = preprocess_example_input(input_config)
    tensor_data = [one_img]
    # import ipdb; ipdb.set_trace()
    model.forward = partial(
        model.forward, img_metas=[[one_meta]], return_loss=False)
    # import ipdb; ipdb.set_trace()
    re = model.forward([one_img])
    # pytorch has some bug in pytorch1.3, we have to fix it
    # by replacing these existing op
    opset_version = 11
    # put the import within the function thus it will not cause import error
    # when not using this function
    try:
        from mmcv.onnx.symbolic import register_extra_symbolics
    except ModuleNotFoundError:
        raise NotImplementedError('please update mmcv to version>=v1.0.4')
    register_extra_symbolics(opset_version)

    return model, tensor_data


def build_model_from_cfg(config_path, checkpoint_path):
    """Build a model from config and load the given checkpoint.

    Args:
        config_path (str): the OpenMMLab config for the model we want to
            export to ONNX
        checkpoint_path (str): Path to the corresponding checkpoint

    Returns:
        torch.nn.Module: the built model
    """
    from mmdet.models import build_detector

    cfg = mmcv.Config.fromfile(config_path)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the model
    import ipdb; ipdb.set_trace()
    # cfg.model = {
        # 'type': 'SCRFD', 
        # 'backbone': {
            # 'type': 'ResNetV1e', 
            # 'depth': 0, 
            # 'block_cfg': {
                # 'block': 'BasicBlock', 
                # 'stage_blocks': (3, 4, 2, 3), 
                # 'stage_planes': [56, 88, 88, 224]
                # }, 
            # 'base_channels': 56, 
            # 'num_stages': 4, 
            # 'out_indices': (0, 1, 2, 3), 
            # 'norm_cfg': {
                # 'type': 'BN', 
                # 'requires_grad': True
                # }, 
            # 'norm_eval': False, 
            # 'style': 'pytorch'
            # }, 
        # 'neck': {
            # 'type': 'PAFPN', 
            # 'in_channels': [56, 88, 88, 224], 
            # 'out_channels': 56, 
            # 'start_level': 1, 
            # 'add_extra_convs': 'on_output', 
            # 'num_outs': 3
            # }, 
        # 'bbox_head': {
            # 'type': 'SCRFDHead', 
            # 'num_classes': 1, 
            # 'in_channels': 56, 
            # 'stacked_convs': 3, 
            # 'feat_channels': 80, 
            # 'norm_cfg': {
                # 'type': 'GN', 
                # 'num_groups': 16, 
                # 'requires_grad': True
                # }, 
            # 'cls_reg_share': True, 
            # 'strides_share': True, 
            # 'scale_mode': 2, 
            # 'anchor_generator': {
                # 'type': 'AnchorGenerator', 
                # 'ratios': [1.0], 
                # 'scales': [1, 2], 
                # 'base_sizes': [16, 64, 256], 
                # 'strides': [8, 16, 32]
                # }, 
            # 'loss_cls': {
                # 'type': 'QualityFocalLoss', 
                # 'use_sigmoid': True, 
                # 'beta': 2.0, 
                # 'loss_weight': 1.0
                # }, 
            # 'loss_dfl': False, 
            # 'reg_max': 8, 
            # 'loss_bbox': {
                # 'type': 'DIoULoss', 
                # 'loss_weight': 2.0
                # }, 
            # 'use_kps': False, 
            # 'loss_kps': {
                # 'type': 'SmoothL1Loss', 
                # 'beta': 0.1111111111111111, 
                # 'loss_weight': 0.1
                # }, 
            # 'train_cfg': {
                # 'assigner': {
                    # 'type': 'ATSSAssigner', 
                    # 'topk': 9
                    # }, 
                # 'allowed_border': -1, 
                # 'pos_weight': -1, 
                # 'debug': False
                # }, 
            # 'test_cfg': {
                # 'nms_pre': -1, 
                # 'min_bbox_size': 0, 
                # 'score_thr': 0.02, 
                # 'nms': {
                    # 'type': 'nms', 
                    # 'iou_threshold': 0.45
                    # }, 
                # 'max_per_img': -1
                # }
            # }, 
        # 'pretrained': None
        # }
        
    # cfg.test_cfg = {
        # 'nms_pre': -1, 
        # 'min_bbox_size': 0, 
        # 'score_thr': 0.02, 
        # 'nms': {
            # 'type': 'nms', 
            # 'iou_threshold': 0.45
            # }, 
        # 'max_per_img': -1
        # }
        
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    
    state_dict = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(state_dict)
    # load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.cpu().eval()
    return model


def preprocess_example_input(input_config):
    """Prepare an example input image for ``generate_inputs_and_wrap_model``.

    Args:
        input_config (dict): customized config describing the example input.

    Returns:
        tuple: (one_img, one_meta), tensor of the example input image and \
            meta information for the example input image.

    Examples:
        >>> from mmdet.core.export import preprocess_example_input
        >>> input_config = {
        >>>         'input_shape': (1,3,224,224),
        >>>         'input_path': 'demo/demo.jpg',
        >>>         'normalize_cfg': {
        >>>             'mean': (123.675, 116.28, 103.53),
        >>>             'std': (58.395, 57.12, 57.375)
        >>>             }
        >>>         }
        >>> one_img, one_meta = preprocess_example_input(input_config)
        >>> print(one_img.shape)
        torch.Size([1, 3, 224, 224])
        >>> print(one_meta)
        {'img_shape': (224, 224, 3),
        'ori_shape': (224, 224, 3),
        'pad_shape': (224, 224, 3),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False}
    """
    input_path = input_config['input_path']
    input_shape = input_config['input_shape']
    one_img = mmcv.imread(input_path)
    one_img = mmcv.imresize(one_img, input_shape[2:][::-1])
    show_img = one_img.copy()
    if 'normalize_cfg' in input_config.keys():
        normalize_cfg = input_config['normalize_cfg']
        mean = np.array(normalize_cfg['mean'], dtype=np.float32)
        std = np.array(normalize_cfg['std'], dtype=np.float32)
        one_img = mmcv.imnormalize(one_img, mean, std)
    one_img = one_img.transpose(2, 0, 1)
    one_img = torch.from_numpy(one_img).unsqueeze(0).float().requires_grad_(
        True)
    (_, C, H, W) = input_shape
    one_meta = {
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
        'show_img': show_img,
    }

    return one_img, one_meta
