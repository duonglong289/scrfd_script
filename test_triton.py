import numpy as np
import cv2
import tritonclient.grpc as input_client
from tqdm import tqdm
from typing import List
import time

# model_name = 'backbone_det'
# model_name = 'face_detection_script_scrfd_10g'
# model_name = 'face_detection_script_scrfd_10g_32'
# model_name = 'face_detection_script_scrfd_10g_batch_warmup'
model_name = 'face_detection_script_scrfd_10g_batch'
model_version = '1'
triton_url = 'localhost:12001'

triton_client = input_client.InferenceServerClient(url=triton_url, verbose=False)

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

path = '/home/longduong/projects/face_project/scrfd/t1.jpg'
path = '/home/longduong/projects/face_project/scrfd/0886332965_FRONT_231112.jpg'

# img = cv2.imread(path)

# resized_img, scale = resize_image(img)
# resized_img = np.expand_dims(resized_img, 0).astype(np.float16)

img = cv2.imread(path)
resized_img1, scale = resize_image(img)
# resized_img = np.expand_dims(resized_img, 0).astype(np.float16)

path2 = '/home/longduong/projects/face_project/scrfd/debug_script.png'
img2 = cv2.imread(path2)
resized_img2, scale2 = resize_image(img2)

# resized_img = np.array([resized_img1, resized_img2]).astype(np.float16)
resized_img = np.array([resized_img1]).astype(np.float16)


print(resized_img.shape)
batch_size = resized_img.shape[0]

# img_t = torch.from_numpy(img)
# img_t = torch.unsqueeze(0)
# resized_img = np.random.rand(1, 640, 640, 3).astype(np.float16)
inputs = [input_client.InferInput('INPUT__0', resized_img.shape, 'FP16'), input_client.InferInput('INPUT__1', [batch_size, 1], 'FP16'), input_client.InferInput('INPUT__2', [batch_size, 1], 'FP16')]
# inputs = [input_client.InferInput('INPUT__0', resized_img.shape, 'FP16'), input_client.InferInput('INPUT__1', [1], 'FP16'), input_client.InferInput('INPUT__2', [1], 'FP16')]

threshold = np.array([[0.5]] * batch_size).astype(np.float16)
threshold_nms = np.array([[0.4]] * batch_size).astype(np.float16)
# threshold = np.array([0.5]).astype(np.float16)
# threshold_nms = np.array([0.4]).astype(np.float16)


inputs[0].set_data_from_numpy(resized_img)
inputs[1].set_data_from_numpy(threshold)
inputs[2].set_data_from_numpy(threshold_nms)

# outputs = [input_client.InferRequestedOutput('OUTPUT__0'), input_client.InferRequestedOutput('OUTPUT__1')]
outputs = [input_client.InferRequestedOutput('OUTPUT__0')]

# for i in tqdm(range(10000)):
start = time.time()
responses = triton_client.infer(model_name,
                            inputs,
                            model_version=model_version,
                            outputs=outputs)

preds_0 = responses.as_numpy('OUTPUT__0')
print(preds_0.shape)
print(preds_0[0])
detected_result = preds_0[0]

print("Time: ", time.time() - start)
factor = scale
dets_list = []
kpss_list = []
probs = []

# if len(detected_result):
#     for result in detected_result:
#         score = result[4]
#         if score == 0:
#             continue
        
#         bbox = (result[:4] / factor).astype(np.int32)
#         kps = (result[5:] / factor).astype(np.int32).reshape(5, 2)
        
#         dets_list.append(bbox)
#         probs.append(score)
#         kpss_list.append(kps)

if len(detected_result):
    # for result in detected_result:
    #     score = result[4]
    #     if score == 0:
    #         continue
        
    #     bbox = (result[:4] / factor).astype(np.int32)
    #     kps = (result[5:] / factor).astype(np.int32).reshape(5, 2)
        
    #     dets_list.append(bbox)
    #     probs.append(score)
    #     kpss_list.append(kps)
    detected_result = detected_result[detected_result[:, 4] > 0]
    dets_list = (detected_result[:, :4] / factor).astype(np.int32)
    
    
    # all_bboxes = detec



print(dets_list)

# for res in preds_0[0]:
#     bbox = res[:4] / scale
#     score = res[4]
#     kps = res[5:].reshape(-1, 2) / scale
    
#     x1,y1,x2,y2 = bbox.astype(np.int32)
    
#     cv2.rectangle(img, (x1,y1)  , (x2,y2) , (255,0,0) , 2)
#     for kp in kps:
#         kp = kp.astype(np.int32)
#         cv2.circle(img, tuple(kp) , 1, (0,0,255) , 2)
        
# cv2.imwrite('hehe.jpg', img)
# print("Done")



'''
Model batch:
[[ 59.62  270.8   102.8   331.2     0.839  71.4   296.2    92.2   295.2
   82.44  307.     74.75  318.8    89.7   318.   ]]


Model old:
[[ 59.62  270.8   102.8   331.2     0.839  71.4   296.2    92.2   295.2
   82.44  307.     74.75  318.8    89.7   318.   ]]

'''