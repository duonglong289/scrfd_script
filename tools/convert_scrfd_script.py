# -*- coding: utf-8 -*-
# Based on Jia Guo reference implementation at
# https://github.com/deepinsight/insightface/blob/master/detection/scrfd/tools/scrfd.py


from __future__ import division
import time
from typing import Union, Dict, List, Tuple
from functools import wraps
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision
torch.set_grad_enabled(False) # Instead of with torch.no_grad

from net.scrfd_head import SCRFDHead
from net.detectors import SCRFD
from net.backbones import ResNetV1e
from net.neck import PAFPN

@torch.jit.script
def nms_torch(dets: torch.Tensor, thresh: float = 0.4):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort(descending=True)

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = torch.maximum(torch.tensor(0.0), xx2 - xx1 + 1)
        h = torch.maximum(torch.tensor(0.0), yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = torch.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

@torch.jit.script
def single_distance2bbox(point: torch.Tensor, distance: torch.Tensor, stride: int):
    """
    Fast conversion of single bbox distances to coordinates

    :param point: Anchor point
    :param distance: Bbox distances from anchor point
    :param stride: Current stride scale
    :return: bbox
    """
    # distance = distance.copy()
    distance = distance.clone()
    distance[0] = point[0] - distance[0] * stride
    distance[1] = point[1] - distance[1] * stride
    distance[2] = point[0] + distance[2] * stride
    distance[3] = point[1] + distance[3] * stride
    return distance

@torch.jit.script
def single_distance2kps(point: torch.Tensor, distance, stride: int):
    """
    Fast conversion of single keypoint distances to coordinates

    :param point: Anchor point
    :param distance: Keypoint distances from anchor point
    :param stride: Current stride scale
    :return: keypoint
    """
    # distance = distance.copy()
    distance = distance.clone()
    for ix in range(0, distance.shape[0], 2):
        distance[ix] = distance[ix] * stride + point[0]
        distance[ix + 1] = distance[ix + 1] * stride + point[1]
    return distance

@torch.jit.script
def generate_proposals(score_blob: torch.Tensor, 
                       bbox_blob: torch.Tensor, 
                       kpss_blob: torch.Tensor, 
                       stride: int, 
                       anchors: torch.Tensor, 
                       threshold: float, 
                       score_out: torch.Tensor, 
                       bbox_out: torch.Tensor, 
                       kpss_out: torch.Tensor,
                       offset: int):
    """
    Convert distances from anchors to actual coordinates on source image
    and filter proposals by confidence threshold.
    Uses preallocated tensors for output.

    :param score_blob: Raw scores for stride
    :param bbox_blob: Raw bbox distances for stride
    :param kpss_blob: Raw keypoints distances for stride
    :param stride: Stride scale
    :param anchors: Precomputed anchors for stride
    :param threshold: Confidence threshold
    :param score_out: Output scores tensor
    :param bbox_out: Output bbox tensor
    :param kpss_out: Output key points tensor
    :param offset: Write offset for output arrays
    :return:
    """

    total = offset

    for ix in range(0, anchors.shape[0]):
        if score_blob[ix, 0] > threshold:
            score_out[total] = score_blob[ix]
            bbox_out[total] = single_distance2bbox(anchors[ix], bbox_blob[ix], stride)
            kpss_out[total] = single_distance2kps(anchors[ix], kpss_blob[ix], stride)
            total += 1

    return score_out, bbox_out, kpss_out, total

@torch.jit.script
def distance2bbox(points: torch.Tensor, distance: torch.Tensor, stride: int):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).

    Returns:
        Tensor: Decoded bboxes.
    """
    
    res = torch.zeros_like(distance, device=distance.device)
    # res = torch.zeros(distance.shape, device=distance.device)
    x1 = points[:, 0] - distance[:, 0] * stride
    y1 = points[:, 1] - distance[:, 1] * stride
    x2 = points[:, 0] + distance[:, 2] * stride
    y2 = points[:, 1] + distance[:, 3] * stride

    # return torch.stack([x1, y1, x2, y2], dim=-1)
    torch.stack([x1, y1, x2, y2], dim=-1, out=res)
    return res

@torch.jit.script
def distance2kps(points: torch.Tensor, distance: torch.Tensor, stride: int):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    res = torch.zeros_like(distance, device=distance.device)
    preds: List[torch.Tensor] = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i] * stride
        py = points[:, i%2+1] + distance[:, i+1] * stride
        # if max_shape is not None:
        #     px = px.clamp(min=0, max=max_shape[1])
        #     py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    torch.stack(preds, dim=-1, out=res)
    return res

@torch.jit.script
def generate_proposals_v2(score_blob: torch.Tensor, 
                       bbox_blob: torch.Tensor, 
                       kpss_blob: torch.Tensor, 
                       stride: int, 
                       anchors: torch.Tensor, 
                       threshold: float):
    """
    Convert distances from anchors to actual coordinates on source image
    and filter proposals by confidence threshold.
    Uses preallocated tensors for output.

    :param score_blob: Raw scores for stride
    :param bbox_blob: Raw bbox distances for stride
    :param kpss_blob: Raw keypoints distances for stride
    :param stride: Stride scale
    :param anchors: Precomputed anchors for stride
    :param threshold: Confidence threshold
    :param score_out: Output scores tensor
    :param bbox_out: Output bbox tensor
    :param kpss_out: Output key points tensor
    :param offset: Write offset for output arrays
    :return:
    """
    pos_inds = torch.where(score_blob>=threshold)[0]
    
    bboxes = distance2bbox(anchors, bbox_blob, stride)
    score_out = score_blob[pos_inds]
    pos_bboxes = bboxes[pos_inds]

    kpss = distance2kps(anchors, kpss_blob, stride)
    # kpss = kpss.view((kpss.shape[0], -1, 2))
    kpss_out = kpss[pos_inds]

    return score_out, pos_bboxes, kpss_out


# class SCRFDDetector(object):
class SCRFDDetector(torch.jit.ScriptModule):
    def __init__(self, weight_path: str):
        super(SCRFDDetector, self).__init__()
        self.center_cache: Dict[str, List[torch.Tensor]] = {'0x0': [torch.tensor([1]), torch.tensor([1]), torch.tensor([1])]}
        self.nms_threshold:float = 0.4
        self.fmc: int = 3
        self._feat_stride_fpn: List[int] = [8, 16, 32]
        self._num_anchors: int = 2

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.init_model(weight_path)

    @torch.jit.ignore
    def init_model(self, weight_path: str):
        block_cfg = {
            'block': 'BasicBlock', 
            'stage_blocks': (3, 4, 2, 3), 
            'stage_planes': [56, 88, 88, 224]
        }
        backbone = ResNetV1e(
            depth=0,
            block_cfg=block_cfg,
            base_channels=56,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            norm_cfg={'type': 'BN', 'requires_grad':False},
            norm_eval=False,
            style='pytorch')
        
        # Init neck
        neck = PAFPN(
            in_channels=[56, 88, 88, 224],
            out_channels=56,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=3)
        
        bbox_head = SCRFDHead(
            num_classes=1, 
            in_channels=56, 
            stacked_convs=3, 
            feat_channels=80, 
            norm_cfg={
                'type': 'BN', 
                'requires_grad': False}, 
            cls_reg_share=True, 
            strides_share=False, 
            scale_mode=2, 
            reg_max=8, 
            use_kps=True, 
            )

        model_det = SCRFD(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head)
    
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
        model_det.load_state_dict(state_dict)
        self.model = model_det
        self.model.eval()
        self.model.to(self.device)
        
        script_backbone = torch.jit.script(self.model)
        script_backbone.save("models/backbone_det/1/backbone.ts")


    @torch.jit.script_method
    # def forward(self, imgs: List[torch.Tensor], threshold: float=0.5, target_size: int = 640) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    # def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Run detection pipeline for provided image

        :param img: Raw image as nd.ndarray with HWC shape
        :param threshold: Confidence threshold
        :return: Face bboxes with scores [t,l,b,r,score], and key points
        """
        target_size = 640
        threshold: float = 0.4

        batch_size: int = len(imgs)
        input_height = input_width = target_size
        
        processed_tensor, all_scale = self.preprocess_batch(imgs, target_size=target_size)
        
        
        net_outs = self.model(processed_tensor)
        
        dets_list = []
        kpss_list = []

        # start_time = time.time()
        bboxes_by_img, kpss_by_img, scores_by_img = self._postprocess(net_outs, input_height, input_width, threshold)

        # print("post process time 1: ", time.time() - start_time)
        # start_time = time.time()
        for index in range(batch_size):
            bbox: torch.Tensor = bboxes_by_img[index]
            kpss: torch.Tensor = kpss_by_img[index]
            scores: torch.Tensor = scores_by_img[index]
            # det, kpss = filter(bboxes_by_img[e], kpss_by_img[e], scores_by_img[e], self.nms_threshold)
            scale = all_scale[index]
            det, kpss = self.filter_result(bbox, kpss, scores, scale, self.nms_threshold)

            dets_list.append(det)
            kpss_list.append(kpss)
        
        # print("post process time 2: ", time.time() - start_time)
        # print("===============================================================================")
        # import ipdb; ipdb.set_trace()
        # return_dets = torch.tensor(dets_list)
        # return_kps = torch.tensor(kpss_list)
        # return (dets_list[0], kpss_list[0])
        return imgs

    @torch.jit.script_method
    def build_anchors(self, input_height: int, input_width: int, strides: List[int], num_anchors: int) -> List[torch.Tensor]:
        """
        Precompute anchor points for provided image size

        :param input_height: Input image height
        :param input_width: Input image width
        :param strides: Model strides
        :param num_anchors: Model num anchors
        :return: box centers
        """
        centers = []
        
        for stride in strides:
            height = input_height // stride
            width = input_width // stride
            # anchor_centers: torch.Tensor = torch.zeros((height, width, 2), device=self.device, dtype=torch.float32)
            xv, yv = torch.meshgrid(torch.arange(width, device=self.device), torch.arange(height, device=self.device))
            # anchor_centers = torch.stack([xv.T, yv.T], dim=-1, out=anchor_centers).to(torch.float32)
            anchor_centers = torch.stack([xv.T, yv.T], dim=-1).to(torch.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if num_anchors > 1:
                anchor_centers = torch.stack([anchor_centers] * num_anchors, dim=1).reshape((-1,2))
            centers.append(anchor_centers)
            
        return centers

    # @torch.jit.ignore
    @torch.jit.script_method
    def preprocess_batch(self,
            # images: List[torch.Tensor],
            images: torch.Tensor,
            target_size: int = 640
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Resize on the longer side of the image to the target size (keep aspect ratio)
        batch_size = len(images)
        
        resized_images: List[torch.Tensor] = []
        all_height: List[int] = []
        all_width: List[int] = []
        all_scale: List[float] = []
        all_index: List[int] = []

        image_size_dict: Dict[str, List[int]] = {}
        for idx, image in enumerate(images):
            height, width, _ = image.size()
            _size = f"{height}x{width}"
            if _size not in image_size_dict:
                image_size_dict[_size] = [idx]
            else:
                image_size_dict[_size].append(idx)
        
        # Dynamic batching
        for key in image_size_dict.keys():
            # image_height, image_width = list(map(int, key.split('x')))
            image_height, image_width = int(key.split('x')[0]), int(key.split('x')[1])
            scale: float = 1.
            batch = torch.stack([images[idx] for idx in image_size_dict[key]]).permute(0, 3, 1, 2).float().to(self.device)
            if target_size != -1:
                if max(image_height, image_width) > target_size:
                    if image_height >= image_width:
                        scale = target_size / image_height
                        new_height = target_size
                        new_width = int(image_width * scale)
                    else:
                        scale = target_size / image_width
                        new_width = target_size
                        new_height = int(image_height * scale)

                    resized_batch = F.interpolate(batch, size=(
                        new_height, new_width), mode="bicubic", align_corners=False)
                else:
                    new_height = image_height
                    new_width = image_width
                    resized_batch = batch
            else:
                new_height = image_height
                new_width = image_width
                resized_batch = batch

            resized_images.extend(list(resized_batch))
            all_index.extend(image_size_dict[key])
            all_width.append(new_width)
            all_height.append(new_height)
            all_scale.extend([scale] * len(image_size_dict[key]))
        
        # zip(all_index, resized_images)
        temp: List[Tuple[int, torch.Tensor]] = []
        for idx in range(len(all_index)):
            temp.append((all_index[idx], resized_images[idx]))
 
        # zip(all_index, all_scale)
        temp2: List[Tuple[int, float]] = []
        for idx in range(len(all_index)):
            temp2.append((all_index[idx], all_scale[idx]))

        resized_images = [x for _, x in sorted(temp)]
        all_scale = [x for _, x in sorted(temp2)]

        # Zero padding sequential
        # max_width = max(all_width)
        # max_height = max(all_height)
        # batched_tensor = torch.zeros((batch_size, 3, max_height, max_width), device=self.device).float()
        batched_tensor = torch.zeros((batch_size, 3, target_size, target_size), device=self.device).float()
        for index in range(batch_size):
            resized_image = resized_images[index]
            image_size = resized_image.size()
            image_height, image_width = int(image_size[1]), int(image_size[2])
            batched_tensor[index, :, :image_height,:image_width] = resized_image
        batched_tensor = (batched_tensor - 127.5) * 0.0078125
        
        # scale_tensor = torch.zeros((len()))
        scale_tensor = torch.tensor(all_scale, device=self.device, dtype=torch.float32)
        
        return batched_tensor, scale_tensor
    
    @torch.jit.script_method
    # @torch.jit.ignore
    def _postprocess(self, 
            net_outs: List[torch.Tensor], 
            input_height: int, 
            input_width: int, 
            threshold: float
            ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Precompute anchor points for provided image size and process network outputs

        :param net_outs: Network outputs
        :param input_height: Input image height
        :param input_width: Input image width
        :param threshold: Confidence threshold
        :return: filtered bboxes, keypoints and scores
        """

        # key = (input_height, input_width)
        image_size_str = f'{input_height}x{input_width}'

        if image_size_str not in self.center_cache.keys():
            anchor_centers = self.build_anchors(input_height, input_width, self._feat_stride_fpn,
                                                         self._num_anchors)
            self.center_cache[image_size_str] = anchor_centers
        else:
            anchor_centers = self.center_cache[image_size_str]
        
        bboxes, kpss, scores = self._process_strides(net_outs, threshold, anchor_centers)
        return bboxes, kpss, scores

    @torch.jit.script_method
    # @torch.jit.ignore
    def _process_strides(self, 
            net_outs: List[torch.Tensor], 
            threshold: float, 
            anchor_centers: List[torch.Tensor]
            ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Process network outputs by strides and return results proposals filtered by threshold

        :param net_outs: Network outputs
        :param threshold: Confidence threshold
        :param anchor_centers: Precomputed anchor centers for all strides
        :return: filtered bboxes, keypoints and scores
        """

        batch_size: int = len(net_outs[0])
        bboxes_by_img = []
        kpss_by_img = []
        scores_by_img = []

        for n_img in range(batch_size):
            offset = 0
            scores_list = []
            bboxes_list = []
            kpss_list = []
            
            for idx, stride in enumerate(self._feat_stride_fpn):
                score_blob = net_outs[idx][n_img]
                bbox_blob = net_outs[idx + self.fmc][n_img]
                kpss_blob = net_outs[idx + self.fmc * 2][n_img]
                stride_anchors = anchor_centers[idx]
                # self.score_list, self.bbox_list, self.kpss_list, total = generate_proposals(score_blob, bbox_blob,
                #                                                                             kpss_blob, stride,
                #                                                                             stride_anchors, threshold,
                #                                                                             self.score_list,
                #                                                                             self.bbox_list,
                #                                                                             self.kpss_list, offset)
                score_out, bbox_out, kps_out = generate_proposals_v2(score_blob, bbox_blob,
                                                                    kpss_blob, stride,
                                                                    stride_anchors, threshold,)
                scores_list.append(score_out)
                bboxes_list.append(bbox_out)
                kpss_list.append(kps_out)
            
            scores = torch.vstack(scores_list)
            bboxes = torch.vstack(bboxes_list)
            kpss = torch.vstack(kpss_list)
            
            # bboxes_by_img.append(torch.clone(self.bbox_list[:offset]))
            # kpss_by_img.append(torch.clone(self.kpss_list[:offset]))
            # scores_by_img.append(torch.clone(self.score_list[:offset]))
            scores_by_img.append(scores)
            bboxes_by_img.append(bboxes)
            kpss_by_img.append(kpss)

        return bboxes_by_img, kpss_by_img, scores_by_img

    @torch.jit.script_method
    def filter_result(self, bboxes_list: torch.Tensor, 
            kpss_list: torch.Tensor,
            scores_list: torch.Tensor, 
            scale: torch.Tensor,
            nms_threshold: float = 0.4
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter postprocessed network outputs with NMS

        :param bboxes_list: List of bboxes (Tensor)
        :param kpss_list: List of keypoints (Tensor)
        :param scores_list: List of scores (Tensor)
        :return: Face bboxes with scores [t,l,b,r,score], and key points
        """

        scores_ravel = scores_list.ravel()
        pre_det = torch.hstack((bboxes_list, scores_list))
        
        keep = torchvision.ops.boxes.nms(bboxes_list, scores_ravel, iou_threshold=nms_threshold)
        
        det = pre_det[keep, :]
        det[:, :4] = det[:, :4] / scale
        
        kept_kpss = kpss_list[keep, :] / scale
        
        # if len(kept_kpss):
        #     kept_kpss = kept_kpss.reshape((kept_kpss.shape[0], -1, 2)) / scale
        
        return det, kept_kpss

if __name__ == '__main__':
    from tqdm import tqdm
    model = SCRFDDetector('model.pth')
    # path = '/mnt/ssd/genos/Github/insightface/python-package/insightface/data/images/t1.jpg'
    path = '/home/longduong/projects/face_project/scrfd/t1.jpg'
    img = cv2.imread(path)
    img_t = torch.from_numpy(img)
    script_model = torch.jit.script(model)
    script_model.save("models/face_detection/1/script_scrfd.ts")
    # script_model = torch.jit.optimize_for_inference(script_model)
    # det_list, kp_list = script_model.detect([img])
    
    img_t = img_t.unsqueeze(0)
    for i in tqdm(range(1)):
        # det_list, kp_list = script_model(img_t)
        # print(det_list[0].shape)
        # det_list, kp_list = model([img_t])
        res = model.forward(img_t)
        det_list, kp_list = res
        torch.cuda.synchronize()
   
    bboxes = det_list
    kpss = kp_list.reshape(-1, 5, 2)
    print(bboxes.shape)
    print(kpss.shape)
    # if kpss is not None:
    #     print(kpss.shape)
    for i in range(len(bboxes)):
        bbox = bboxes[i].detach().cpu().numpy()
        x1,y1,x2,y2,score = bbox.astype(np.int32)
        cv2.rectangle(img, (x1,y1)  , (x2,y2) , (255,0,0) , 2)
    
        kps = kpss[i].detach().cpu().numpy()
        for kp in kps:
            kp = kp.astype(np.int32)
            cv2.circle(img, tuple(kp) , 1, (0,0,255) , 2)
    
    print('output: debug_script.png',)
    cv2.imwrite('debug_script.png', img)