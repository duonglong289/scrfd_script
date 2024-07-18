
import torch 
import torchvision
import cv2
import numpy as np
from tqdm import tqdm

device = torch.device('cuda')
script_model = torch.jit.load('script_scrfd.ts', map_location=device)

script_model.device = device

path = '/home/longduong/projects/face_project/scrfd/t1.jpg'
img = cv2.imread(path)
img_t = torch.from_numpy(img).to(device)

script_model = torch.jit.optimize_for_inference(script_model)


# for i in tqdm(range(10000)):
#     det_list, kp_list = script_model([img_t])
#     # print(det_list[0].shape)
#     # det_list, kp_list = model([img_t])
#     torch.cuda.synchronize()

with torch.no_grad():
    # First dummy run for warmup
    for i in range(100):
        image = torch.rand((640, 480, 3))
        _ = script_model([image])
    print("Done warmup single")

    runs = 10000
    img = cv2.imread('/home/longduong/projects/face_project/scrfd/t1.jpg')
    for _ in tqdm(range(runs), desc="Scripted model"):
        
    # for _ in range(runs):
        # print("random time: ", time.time() - st)
        test_tensor = torch.from_numpy(img).to(device)
        test_tensor = test_tensor + torch.rand_like(test_tensor, device=device, dtype=torch.float32) * 100
        
        res = script_model([test_tensor])
        # res += 1
        torch.cuda.synchronize()

# bboxes = det_list[0]
# kpss = kp_list[0]
# print(bboxes.shape)
# # if kpss is not None:
# #     print(kpss.shape)
# for i in range(len(bboxes)):
#     bbox = bboxes[i].detach().cpu().numpy()
#     x1,y1,x2,y2,score = bbox.astype(np.int32)
#     cv2.rectangle(img, (x1,y1)  , (x2,y2) , (255,0,0) , 2)

#     kps = kpss[i].detach().cpu().numpy()
#     for kp in kps:
#         kp = kp.astype(np.int32)
#         cv2.circle(img, tuple(kp) , 1, (0,0,255) , 2)

# print('output: debug_script.png',)
# cv2.imwrite('debug_script.png', img)