from abc import ABCMeta
import numpy as np

import torch.nn as nn


class BaseDetector(nn.Module, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    # @property
    # def with_shared_head(self):
    #     """bool: whether the detector has a shared head in the RoI Head"""
    #     return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    # @property
    # def with_bbox(self):
    #     """bool: whether the detector has a bbox head"""
    #     return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
    #             or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    # @property
    # def with_mask(self):
    #     """bool: whether the detector has a mask head"""
    #     return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
    #             or (hasattr(self, 'mask_head') and self.mask_head is not None))

    # @abstractmethod
    # def extract_feat(self, imgs):
    #     """Extract features from images."""
    #     pass

    # def extract_feats(self, imgs):
    #     """Extract features from multiple images.

    #     Args:
    #         imgs (list[torch.Tensor]): A list of images. The images are
    #             augmented from the same image but in different ways.

    #     Returns:
    #         list[torch.Tensor]: Features of different images
    #     """
    #     assert isinstance(imgs, list)
    #     return [self.extract_feat(img) for img in imgs]






    # def init_weights(self, pretrained=None):
    #     """Initialize the weights in detector.

    #     Args:
    #         pretrained (str, optional): Path to pre-trained weights.
    #             Defaults to None.
    #     """
    #     if pretrained is not None:
    #         logger = get_root_logger()
    #         print_log(f'load model from: {pretrained}', logger=logger)

    
    # def forward_test(self, imgs, img_metas, **kwargs):
    #     """
    #     Args:
    #         imgs (List[Tensor]): the outer list indicates test-time
    #             augmentations and inner Tensor should have a shape NxCxHxW,
    #             which contains all images in the batch.
    #         img_metas (List[List[dict]]): the outer list indicates test-time
    #             augs (multiscale, flip, etc.) and the inner list indicates
    #             images in a batch.
    #     """
    #     for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
    #         if not isinstance(var, list):
    #             raise TypeError(f'{name} must be a list, but got {type(var)}')

    #     # num_augs = len(imgs)
    #     # if num_augs != len(img_metas):
    #     #     raise ValueError(f'num of augmentations ({len(imgs)}) '
    #     #                      f'!= num of image meta ({len(img_metas)})')

    #     # NOTE the batched image size information may be useful, e.g.
    #     # in DETR, this is needed for the construction of masks, which is
    #     # then used for the transformer_head.
    #     for img, img_meta in zip(imgs, img_metas):
    #         batch_size = len(img_meta)
    #         for img_id in range(batch_size):
    #             img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

    #     # if num_augs == 1:
    #         # proposals (List[List[Tensor]]): the outer list indicates
    #         # test-time augs (multiscale, flip, etc.) and the inner list
    #         # indicates images in a batch.
    #         # The Tensor should have a shape Px4, where P is the number of
    #         # proposals.
    #     # if 'proposals' in kwargs:
    #     #     kwargs['proposals'] = kwargs['proposals'][0]
    #     return self.simple_test(imgs[0], img_metas[0], **kwargs)
    #     # else:
    #     #     assert imgs[0].size(0) == 1, 'aug test does not support ' \
    #     #                                  'inference with batch size ' \
    #     #                                  f'{imgs[0].size(0)}'
    #     #     # TODO: support test augmentation for predefined proposals
    #     #     assert 'proposals' not in kwargs
    #     #     return self.aug_test(imgs, img_metas, **kwargs)

    # # @auto_fp16(apply_to=('img', ))
    # def forward(self, img, img_metas, return_loss=True, **kwargs):
    #     """Calls either :func:`forward_train` or :func:`forward_test` depending
    #     on whether ``return_loss`` is ``True``.

    #     Note this setting will change the expected inputs. When
    #     ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
    #     and List[dict]), and when ``resturn_loss=False``, img and img_meta
    #     should be double nested (i.e.  List[Tensor], List[List[dict]]), with
    #     the outer list indicating test time augmentations.
    #     """
    #     return self.forward_test(img, img_metas, **kwargs)

