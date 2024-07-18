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


# class SCRFDDetector(object):
class CustomModel(torch.jit.ScriptModule):
    def __init__(self,):
        super(CustomModel, self).__init__()
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.abc: int = 2


    @torch.jit.script_method
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Run detection pipeline for provided image

        :param img: Raw image as nd.ndarray with HWC shape
        :param threshold: Confidence threshold
        :return: Face bboxes with scores [t,l,b,r,score], and key points
        """

        return imgs


if __name__ == '__main__':
    from tqdm import tqdm
    model = CustomModel()
    # path = '/mnt/ssd/genos/Github/insightface/python-package/insightface/data/images/t1.jpg'
    path = '/home/longduong/projects/face_project/scrfd/t1.jpg'
    img = cv2.imread(path)
    img = cv2.resize(img, (640, 640))
    img_t = torch.from_numpy(img)
    script_model = torch.jit.script(model)
    script_model.save("models/custom_model/1/model.pt")
    
    img_t = img_t.unsqueeze(0)
    for i in tqdm(range(1)):
        
        res = model.forward(img_t)
        print(res.shape)
        torch.cuda.synchronize()
   