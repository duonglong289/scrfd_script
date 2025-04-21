# -*- coding: utf-8 -*-
# Based on Jia Guo reference implementation at
# https://github.com/deepinsight/insightface/blob/master/detection/scrfd/tools/scrfd.py


from __future__ import division
import time
from typing import Union, Dict, List, Tuple


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

def resize_image(image, max_size: List = None):
    if max_size is None:
        max_size = [640, 640]

    cw = max_size[0]
    ch = max_size[1]
    h, w, _ = image.shape

    scale_factor = min(cw / w, ch / h)
    # If image is too small, it may contain only single face, which leads to decreased detection accuracy,
    # so we reduce scale factor by some factor
    if scale_factor > 2:
        scale_factor = scale_factor * 0.7

    if scale_factor <= 1.:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC

    if scale_factor == 1.:
        transformed_image = image
    else:
        transformed_image = cv2.resize(image, (0, 0), fx=scale_factor,
                                     fy=scale_factor,
                                     interpolation=interp)

    h, w, _ = transformed_image.shape

    if w < cw:
        transformed_image = cv2.copyMakeBorder(transformed_image, 0, 0, 0, cw - w,
                                             cv2.BORDER_CONSTANT)
    if h < ch:
        transformed_image = cv2.copyMakeBorder(transformed_image, 0, ch - h, 0, 0,
                                             cv2.BORDER_CONSTANT)

    return transformed_image, scale_factor

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


class SCRFDDetector(object):
# class SCRFDDetector(torch.jit.ScriptModule):
    def __init__(self, weight_path: str):
        super(SCRFDDetector, self).__init__()
        self.center_cache: Dict[str, List[torch.Tensor]] = {'0x0': [torch.tensor([1]), torch.tensor([1]), torch.tensor([1])]}
        # self.nms_threshold:float = 0.4
        self.fmc: int = 3
        self._feat_stride_fpn: List[int] = [8, 16, 32]
        self._num_anchors: int = 2
        self.input_width: int = 640
        self.input_height: int = 640

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
    
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'), weights_only=True)['state_dict']
        model_det.load_state_dict(state_dict)
        self.model = model_det
        self.model.eval()
        self.model.to(self.device)
        self.model = self.model.half()
        

    # @torch.jit.script_method
    def forward(self, imgs: torch.Tensor, threshold: torch.Tensor=torch.tensor([0.5]), nms_threshold: torch.Tensor=torch.tensor([0.4])):
        """
        Run detection pipeline for provided image

        :param img: Raw image as nd.ndarray with HWC shape
        :param threshold: Confidence threshold
        :return: Face bboxes with scores [t,l,b,r,score], and key points
        """
        imgs = imgs.half()
        threshold_value = threshold[0].item()
        nms_threshold_value = nms_threshold[0].item()

        batch_size: int = len(imgs)
        
        processed_tensor, scale_tensor = self.preprocess_batch(imgs)
        net_outs = self.model(processed_tensor)
        
        bboxes_by_img, kpss_by_img, scores_by_img = self._postprocess(net_outs, self.input_height, self.input_width, threshold_value)
        
        # Solution 1: For loop nms
        list_result = []
        max_num_bbox = 0
        for index in range(batch_size):
            bbox: torch.Tensor = bboxes_by_img[index]
            kpss: torch.Tensor = kpss_by_img[index]
            scores: torch.Tensor = scores_by_img[index]
            result = self.filter_result(bbox, kpss, scores, scale_tensor.item(), nms_threshold_value)
            
            if result.shape[0] > max_num_bbox:
                max_num_bbox = result.shape[0]
            list_result.append(result)
        
        # Solution 2: Batch nms
        # import ipdb; ipdb.set_trace()
        # list_result = self.filter_result_batch(bboxes_by_img, kpss_by_img, scores_by_img, scale_tensor.item(), nms_threshold_value)
        
        # Pad bbox
        return_result = torch.zeros((batch_size, max_num_bbox, 15), dtype=torch.float16, device=self.device)
        for index in range(batch_size):
            result_bbox = list_result[index]
            num_bbox = result_bbox.shape[0]
            return_result[index, :num_bbox, :] = result_bbox
        
        return_result = return_result.to(torch.float16)
        return return_result

    # @torch.jit.script_method
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
            xv, yv = torch.meshgrid(torch.arange(width, device=self.device), torch.arange(height, device=self.device), indexing='ij')
            anchor_centers = torch.stack([xv.T, yv.T], dim=-1).to(torch.float16)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if num_anchors > 1:
                anchor_centers = torch.stack([anchor_centers] * num_anchors, dim=1).reshape((-1,2))
            centers.append(anchor_centers)
        
        return centers

    # @torch.jit.script_method
    def preprocess_batch(self, images: torch.Tensor):
        target_size: int = 640
        batch_size, image_height, image_width, _ = images.shape
        
        batch = images.permute(0, 3, 1, 2).to(self.device).half()
        
        scale: float = 1.
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
        
        batched_tensor = torch.zeros((batch_size, 3, target_size, target_size), device=self.device, dtype=torch.float16)
        for index in range(batch_size):
            resized_image = resized_batch[index]
            image_size = resized_image.size()
            image_height, image_width = int(image_size[1]), int(image_size[2])
            batched_tensor[index, :, :image_height,:image_width] = resized_image
        batched_tensor = (batched_tensor - 127.5) * 0.0078125
        scale_tensor = torch.tensor(scale, dtype=torch.float16, device=self.device)
        # scale_tensor = torch.tensor(scale, dtype=torch.float32, device=self.device)
        
        return batched_tensor, scale_tensor
    
    def preprocess_single_image(self, image: torch.Tensor, target_size: int = 640) -> Tuple[torch.Tensor, int, int, float]:
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device).float()
        scale: float = 1.
        image_size = image.size()
        image_height, image_width = int(image_size[-2]), int(image_size[-1])

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

                # Mimic torchvision.transforms.ToTensor to use interpolate
                resized_image = F.interpolate(image, size=(
                    new_height, new_width), mode="bicubic", align_corners=False)
            else:
                new_height = image_height
                new_width = image_width
                resized_image = image
        else:
            new_height = image_height
            new_width = image_width
            resized_image = image

        resized_image = resized_image.squeeze(0).permute(1, 2, 0)
        
        # resized_image -= self.mean_tensor  # mean substraction
        # resized_image = resized_image.permute(2, 0, 1)
        return resized_image, new_width, new_height, scale

    # @torch.jit.script_method
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

    # @torch.jit.script_method
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
            
            scores_by_img.append(scores)
            bboxes_by_img.append(bboxes)
            kpss_by_img.append(kpss)

        return bboxes_by_img, kpss_by_img, scores_by_img

    # @torch.jit.script_method
    def filter_result(self, bboxes_list: torch.Tensor, 
            kpss_list: torch.Tensor,
            scores_list: torch.Tensor, 
            scale: float,
            nms_threshold: float,
            ) -> torch.Tensor:
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
        det[:, :4] = (det[:, :4] / scale).to(torch.float16)
        kept_kpss = (kpss_list[keep, :] / scale).to(torch.float16)

        result = torch.concat((det, kept_kpss), dim=1)
        # if len(kept_kpss):
        #     kept_kpss = kept_kpss.reshape((kept_kpss.shape[0], -1, 2)) / scale
        
        return result

    # @torch.jit.script_method
    def filter_result_batch(self, 
            bboxes_list: List[torch.Tensor], 
            kpss_list: List[torch.Tensor],
            scores_list: List[torch.Tensor], 
            scale: float,
            nms_threshold: float,
            ) -> List[torch.Tensor]:
        """
        Filter postprocessed network outputs with batch NMS for faster processing

        :param bboxes_list: List of bboxes (Tensor) for each image in batch
        :param kpss_list: List of keypoints (Tensor) for each image in batch
        :param scores_list: List of scores (Tensor) for each image in batch
        :param scale: Scale factor to apply to coordinates
        :param nms_threshold: IoU threshold for NMS
        :return: List of face bboxes with scores and key points for each image
        """
        batch_size = len(bboxes_list)
        
        # Prepare batch data for torchvision batch NMS
        all_boxes = torch.cat(bboxes_list)
        all_scores = torch.cat([scores.ravel() for scores in scores_list])
        all_kpss = torch.cat(kpss_list)
        
        # Create batch indices for each box
        batch_indices = torch.cat([torch.full((len(bboxes),), i, 
                                dtype=torch.int64, 
                                device=self.device) 
                                for i, bboxes in enumerate(bboxes_list)])
        
        # Apply batch NMS
        keep = torchvision.ops.boxes.batched_nms(
            all_boxes, all_scores, batch_indices, nms_threshold)
        
        # Map kept indices back to their original batches
        result_list = []
        for i in range(batch_size):
            batch_mask = batch_indices[keep] == i
            batch_keep = keep[batch_mask]
            
            # Get boxes, scores, and keypoints for this batch
            batch_boxes = all_boxes[batch_keep]
            batch_scores = all_scores[batch_keep].unsqueeze(1)
            batch_kpss = all_kpss[batch_keep]
            
            # Apply scaling
            batch_boxes = (batch_boxes / scale).to(torch.float16)
            batch_kpss = (batch_kpss / scale).to(torch.float16)
            
            # Combine results
            batch_result = torch.cat((batch_boxes, batch_scores, batch_kpss), dim=1)
            result_list.append(batch_result)
        
        return result_list
    


if __name__ == '__main__':
    from tqdm import tqdm
    model = SCRFDDetector('model.pth')
    # path = '/mnt/ssd/genos/Github/insightface/python-package/insightface/data/images/t1.jpg'
    path = '/home/longduong/projects/face_project/scrfd/t1.jpg'
    # path = '/mnt/hdd/spaces/genos/scrfd/0886332965_FRONT_231112.jpg'
    img = cv2.imread(path)
    resized_img, scale = resize_image(img)
    # resized_img = np.expand_dims(resized_img, 0).astype(np.float16)
    
    path2 = '/home/longduong/projects/face_project/scrfd/debug_script.png'
    img2 = cv2.imread(path2)
    resized_img2, scale2 = resize_image(img2)
    
    batch = np.array([resized_img, resized_img2])
    print(batch.shape)
    # resized_img = np.random.rand(1, 640, 640, 3)
    img_t = torch.from_numpy(batch).cuda()
    
    # script_model = torch.jit.script(model)
    # script_model.save("models/face_detection_script_scrfd_10g/1/script_scrfd_torch25.ts")

    # script_model = torch.jit.load("models/face_detection_script_scrfd_10g/1/script_scrfd_torch25.ts", map_location='cuda')
    # script_model.eval()
    # script_model.to('cuda')

    # # script_model = torch.jit.optimize_for_inference(script_model)
    # # det_list, kp_list = script_model.detect([img])
    
    for i in tqdm(range(1)):
        # det_list, kp_list = script_model(img_t)
        # print(det_list[0].shape)
        det_list, kp_list = model.forward(img_t)
        # print(img_t.device)
        # print(script_model.device)
        # preds_0 = script_model.forward(img_t)
        # print(preds_0.shape)
        # det_list, kp_list = res
        torch.cuda.synchronize()
   
    # # bboxes = det_list
    # # kpss = kp_list.reshape(-1, 5, 2)

    # # if kpss is not None:
    # #     print(kpss.shape)
    # import ipdb; ipdb.set_trace()
    # print(preds_0[0])
    
    # for res in preds_0[0].detach().cpu().numpy():
    #     bbox = res[:4] / scale
    #     score = res[4]
    #     kps = res[5:].reshape(-1, 2) / scale
        
    #     x1,y1,x2,y2 = bbox.astype(np.int32)
        
    #     cv2.rectangle(img, (x1,y1)  , (x2,y2) , (255,0,0) , 2)
    #     for kp in kps:
    #         kp = kp.astype(np.int32)
    #         cv2.circle(img, tuple(kp) , 1, (0,0,255) , 2)

    # print('output: converted_test.png',)
    # cv2.imwrite('converted_test.png', img)