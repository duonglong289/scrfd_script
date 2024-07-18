from .single_stage import SingleStageDetector
import torch


# @DETECTORS
class SCRFD(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head):
        super(SCRFD, self).__init__(backbone, neck, bbox_head)

    # def simple_test(self, img, img_metas, rescale=False):
    #     """Test function without test time augmentation.

    #     Args:
    #         imgs (list[torch.Tensor]): List of multiple images
    #         img_metas (list[dict]): List of image information.
    #         rescale (bool, optional): Whether to rescale the results.
    #             Defaults to False.

    #     Returns:
    #         list[list[np.ndarray]]: BBox results of each image and classes.
    #             The outer list corresponds to each image. The inner list
    #             corresponds to each class.
    #     """
    #     x = self.extract_feat(img)
    #     outs = self.bbox_head(x) 
    #     # len(outs) = 3
    #     # Class score
    #     # outs[0][0].shape = [1, 2, 80, 80]
    #     # outs[0][1].shape = [1, 2, 40, 40]
    #     # outs[0][2].shape = [1, 2, 20, 20]
        
    #     # bbox_pred
    #     # outs[1][0].shape = [1, 8, 80, 80]
    #     # outs[1][1].shape = [1, 8, 40, 40]
    #     # outs[1][2].shape = [1, 8, 20, 20]
        
    #     # kps
    #     # outs[2][0].shape = [1, 10, 80, 80]
    #     # outs[2][1].shape = [1, 10, 40, 40]
    #     # outs[2][2].shape = [1, 10, 20, 20]
        
    #     bbox_list = self.bbox_head.get_bboxes(
    #         *outs, img_metas, rescale=rescale)
        
    #     bbox_results = [
    #         bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
    #         for det_bboxes, det_labels in bbox_list
    #     ]
    #     return bbox_results

    @torch.no_grad()
    def forward(self, img):
        x1 = self.extract_feat(img)
        outs = self.bbox_head(x1)
        # C = 3
        # H = W = 640
        # one_meta = {
        #     'img_shape': (H, W, C),
        #     'ori_shape': (H, W, C),
        #     'pad_shape': (H, W, C),
        #     'scale_factor': 1.0,
        #     'flip': False,
        # }
        
        list_strides = [8, 16, 32]
        list_scores = outs[0]
        list_bboxes = outs[1]
        list_kps = outs[2]
        
        scores_out = []
        bboxes_out = []
        kps_out = []
        
        for ind in range(len(list_bboxes)):
            scores = list_scores[ind][0].permute(1, 2, 0).reshape(-1, 1).sigmoid()
            bboxes = list_bboxes[ind][0].permute(1, 2, 0)
            bboxes = bboxes.reshape((-1,4)) # * list_strides[ind]
            kps = list_kps[ind][0].permute(1, 2, 0)
            kps = kps.reshape((-1, 10)) # * list_strides[ind]
            # scores_out.append(scores.unsqueeze(0).detach().cpu().numpy())
            # bboxes_out.append(bboxes.unsqueeze(0).detach().cpu().numpy())
            # kps_out.append(kps.unsqueeze(0).detach().cpu().numpy())
            scores_out.append(scores)
            bboxes_out.append(bboxes)
            kps_out.append(kps)
        
        net_outs = []
        net_outs.extend(scores_out)
        net_outs.extend(bboxes_out)
        net_outs.extend(kps_out)

        return net_outs