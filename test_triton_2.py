import numpy as np
import cv2
import tritonclient.grpc as input_client
from tqdm import tqdm
from typing import List

# model_name = 'backbone_det'
model_name = 'face_detection_scrfd_10g_gnkps'
# model_name = 'face_detection_script_scrfd_10g_32'
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
img = cv2.imread(path)
# img = cv2.resize(img, (640, 640))

resized_img, scale = resize_image(img)
resized_img = np.expand_dims(resized_img, 0).astype(np.float16)
# resized_img = np.expand_dims(resized_img, 0).astype(np.float32)
print(resized_img.shape)

# processed_image = processed_image.permute(2, 0, 1)
# processed_image = (processed_image - 127.5) * 0.0078125

print(resized_img.shape)
# img_t = img.astype(np.float32)
# img_t = torch.from_numpy(img)
# img_t = torch.unsqueeze(0)
resized_img = np.random.rand(1, 3, 640, 640).astype(np.float32)
# inputs = [input_client.InferInput('INPUT__0', resized_img.shape, 'FP16')]
inputs = [input_client.InferInput('input.1', resized_img.shape, 'FP32')]

inputs[0].set_data_from_numpy(resized_img)
# outputs = [input_client.InferRequestedOutput('OUTPUT__0'), input_client.InferRequestedOutput('OUTPUT__1')]
outputs = [
    input_client.InferRequestedOutput('score_8'),
    input_client.InferRequestedOutput('score_16'),
    input_client.InferRequestedOutput('score_32'),
    input_client.InferRequestedOutput('bbox_8'),
    input_client.InferRequestedOutput('bbox_16'),
    input_client.InferRequestedOutput('bbox_32'),
    input_client.InferRequestedOutput('kps_8'),
    input_client.InferRequestedOutput('kps_16'),
    input_client.InferRequestedOutput('kps_32'),
    ]

# for i in tqdm(range(10000)):
responses = triton_client.infer(model_name,
                            inputs,
                            model_version=model_version,
                            outputs=outputs)

preds_0 = responses.as_numpy('score_8')
print(preds_0.shape)



# for res in preds_0:
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

# for i in range(50):
#     print("aaaaaaaaa")

# import time
# time.sleep(2)