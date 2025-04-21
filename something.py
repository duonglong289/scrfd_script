import numpy as np
import cv2
import tritonclient.grpc as input_client
from tqdm import tqdm
from typing import List

# model_name = 'backbone_det'
model_name = 'face_detection_script_scrfd_10g'
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
path = 'png-clipart-santa-claus-christmas-santa-claus-holidays-christmas-decoration.png'



img = cv2.imread(path)
resized_img1, scale = resize_image(img)
# resized_img = np.expand_dims(resized_img, 0).astype(np.float16)

path2 = '/home/longduong/projects/face_project/scrfd/debug_script.png'
img2 = cv2.imread(path2)
resized_img2, scale2 = resize_image(img2)

resized_img = np.array([resized_img1, resized_img2]).astype(np.float16)

print(resized_img.shape)

# img_t = torch.from_numpy(img)
# img_t = torch.unsqueeze(0)
# resized_img = np.random.rand(1, 640, 640, 3).astype(np.float16)

def call(input_image, **kwargs):
    inputs = [input_client.InferInput('INPUT__0', input_image.shape, 'FP16')]

    inputs[0].set_data_from_numpy(input_image)
        
    for ind, key in enumerate(kwargs.keys()):
        new_input = input_client.InferInput(f'INPUT__{ind+1}', kwargs[key].shape, 'FP16')
        inputs.append(new_input)
    
    print("hehe: ", len(inputs))
    
    inputs[1].set_data_from_numpy(threshold)
    inputs[2].set_data_from_numpy(threshold_nms)

    outputs = [input_client.InferRequestedOutput('OUTPUT__0')]

    responses = triton_client.infer(model_name,
                                inputs,
                                model_version=model_version,
                                outputs=outputs)

    preds_0 = responses.as_numpy('OUTPUT__0')
    print(preds_0.shape)
    print(preds_0[0])



threshold = np.array([[0.5], [0.5]]).astype(np.float16)
threshold_nms = np.array([[0.4], [0.4]]).astype(np.float16)

call(resized_img, input_1=threshold, input_2=threshold_nms)

