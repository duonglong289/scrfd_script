import torch
import torch.nn as nn
from typing import List


class SCRFD(nn.Module):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head):
        super(SCRFD, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head

    @torch.no_grad()
    def forward(self, img: torch.Tensor):
        batch_size: int = img.shape[0]
        # x1 = self.extract_feat(img)
        x = self.backbone(img)
        x = self.neck(x)
        outs = self.bbox_head(x)

        # list_strides = [8, 16, 32]
        list_scores = outs[0]
        list_bboxes = outs[1]
        list_kps = outs[2]
        
        scores_out = []
        bboxes_out = []
        kps_out = []
        
        # for batch_ind in range(img.shape[0]):
        for ind in range(len(list_bboxes)):
            scores = list_scores[ind].permute(0, 2, 3, 1).reshape(batch_size, -1, 1).sigmoid()
            bboxes = list_bboxes[ind].permute(0, 2, 3, 1)
            bboxes = bboxes.reshape((batch_size, -1, 4)) # * list_strides[ind]
            kps = list_kps[ind].permute(0, 2, 3, 1)
            kps = kps.reshape((batch_size, -1, 10)) # * list_strides[ind]
            
            scores_out.append(scores)
            bboxes_out.append(bboxes)
            kps_out.append(kps)
            # scores_out.append(scores)
            # bboxes_out.append(bboxes)
            # kps_out.append(kps)
            # scores_out.append(scores.unsqueeze(0).detach().cpu().numpy())
            # bboxes_out.append(bboxes.unsqueeze(0).detach().cpu().numpy())
            # kps_out.append(kps.unsqueeze(0).detach().cpu().numpy())
        
        net_outs = []
        net_outs.extend(scores_out)
        net_outs.extend(bboxes_out)
        net_outs.extend(kps_out)
        
        return net_outs[0], net_outs[1], net_outs[2], net_outs[3], net_outs[4], net_outs[5], net_outs[6], net_outs[7], net_outs[8]