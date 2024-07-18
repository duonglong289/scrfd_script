import datetime

import torch
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np

from typing import List, Tuple, Dict, Optional, Union

from net.scrfd_head import SCRFDHead
from net.detectors import SCRFD
from net.backbones import ResNetV1e
from net.neck import PAFPN


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
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2bbox_torch(points, distance):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]

    return torch.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def distance2kps_torch(points, distance):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds: List[torch.Tensor] = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        # if max_shape is not None:
        #     px = px.clamp(min=0, max=max_shape[1])
        #     py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return torch.stack(preds, axis=-1)

class FaceDetectionSCRFD:
    def __init__(self, weight_path: str):
        self.batched = False
        self.center_cache: Dict[str, torch.Tensor] = {}
        # if torch.cuda.is_available():
        #     self.device = torch.device('cuda')
        # else:
        self.device = torch.device('cpu')
        self.init_model()
        
        self.fmc: int = 3
        self._feat_stride_fpn: List[int] = [8, 16, 32]
        self._num_anchors: int = 2
        self.nms_thresh: float = 0.4
        
        self.input_height: int = 640
        self.input_width: int = 640

    def init_model(self, model_file='model.pth'):
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
        
        # model_det = torch.jit.script(model_det)
        
        ckpt_path = model_file
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
        model_det.load_state_dict(state_dict)
        self.model = model_det
        self.model.eval()
        self.model.to(self.device)
        
    def preprocess(self, img: torch.Tensor):
        processed_image = img.permute(2, 0, 1)
        processed_image = (processed_image - 127.5) * 0.0078125
        processed_image = processed_image.unsqueeze(0)
        return processed_image
    
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
    def preprocess_batch(self,
                         images: List[torch.Tensor],
                         target_size: int = -1):
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
            batch = torch.stack([images[idx] for idx in image_size_dict[key]]).permute(0, 3, 1, 2).to(self.device).float()
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

            resized_batch = resized_batch.permute(0, 2, 3, 1) - self.mean_tensor
            resized_batch = resized_batch.permute(0, 3, 1, 2)

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
        max_width = max(all_width)
        max_height = max(all_height)
        batched_tensor = torch.zeros((batch_size, 3, max_height, max_width), device=self.device).float()
        for index in range(batch_size):
            resized_image = resized_images[index]
            image_size = resized_image.size()
            image_height, image_width = int(image_size[1]), int(image_size[2])
            batched_tensor[index, :, :image_height,
                           :image_width] = resized_image
        return batched_tensor, all_scale
    
    @torch.no_grad()
    def forward_torch(self, img: torch.Tensor, thresh: float):
        scores_list: List[torch.Tensor] = []
        bboxes_list: List[torch.Tensor] = []
        kpss_list: List[torch.Tensor] = []
        
        process_tensor = self.preprocess(img)
        
        net_outs = self.model(process_tensor)
        
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx][0]
            bbox_preds = net_outs[idx + fmc][0]
            bbox_preds = bbox_preds * stride
            kps_preds = net_outs[idx + fmc * 2][0] * stride

            height = self.input_height // stride
            width = self.input_width // stride

            image_size_str = f"{height}-{width}-{stride}"

            if image_size_str in self.center_cache:
                anchor_centers = self.center_cache[image_size_str]
            else:
                xv, yv = torch.meshgrid(torch.arange(width), torch.arange(height))
                anchor_centers = torch.stack([xv.T, yv.T], axis=-1).to(torch.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                anchor_centers = torch.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                self.center_cache[image_size_str] = anchor_centers

            pos_inds = torch.where(scores>=thresh)[0]
            
            bboxes = distance2bbox_torch(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            
            kpss = distance2kps_torch(anchor_centers, kps_preds)
            
            kpss = kpss.view((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            if len(pos_inds):
                import ipdb; ipdb.set_trace()
            kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list

    def detect_torch(self, img, thresh:float = 0.5, input_size: int = 640):
        img = torch.from_numpy(img)
        resized_img, new_width, new_height, det_scale = self.preprocess_single_image(img, input_size)
        det_img = torch.zeros((self.input_height, self.input_width, 3), device=self.device)
        det_img[:new_height, :new_width, :] = resized_img
        scores_list, bboxes_list, kpss_list = self.forward_torch(det_img, thresh)

        scores = torch.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort(descending=True)
        bboxes = torch.vstack(bboxes_list) / det_scale
        
        kpss = torch.vstack(kpss_list) / det_scale
        pre_det = torch.hstack((bboxes, scores)).to(torch.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms_torch(pre_det)
        det = pre_det[keep, :]
        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]
        
        import ipdb; ipdb.set_trace()
        return det, kpss

    def nms_torch(self, dets: torch.Tensor):
        thresh = self.nms_thresh
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


if __name__ == '__main__':
    import glob
    detector = FaceDetectionSCRFD(weight_path='model.pth')
    img_paths = ['/home/longduong/projects/face_project/scrfd/t1.jpg']
    for img_path in img_paths:
        img = cv2.imread(img_path)

        for _ in range(1):
            ta = datetime.datetime.now()
            # bboxes, kpss = detector.detect(img, 0.5, input_size = (640, 640))
            bboxes, kpss = detector.detect_torch(img, 0.5, input_size = 640)
            #bboxes, kpss = detector.detect(img, 0.5)
            tb = datetime.datetime.now()
            print('all cost:', (tb-ta).total_seconds()*1000)
            
        print(img_path, bboxes.shape)
        if kpss is not None:
            print(kpss.shape)
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i].detach().cpu().numpy()
            x1,y1,x2,y2,score = bbox.astype(np.int32)
            cv2.rectangle(img, (x1,y1)  , (x2,y2) , (255,0,0) , 2)
            if kpss is not None:
                kps = kpss[i].detach().cpu().numpy()
                for kp in kps:
                    kp = kp.astype(np.int32)
                    cv2.circle(img, tuple(kp) , 1, (0,0,255) , 2)
        filename = img_path.split('/')[-1]
        print('output: debug.png',)

        # cv2.imwrite('./outputs/%s'%filename, img)
        cv2.imwrite('debug.png', img)

# if __name__ == '__main__':
#     main()




    
