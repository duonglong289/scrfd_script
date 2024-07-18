import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import tritonclient.grpc as input_client

model_name = 'custom_model'
model_version = '1'
triton_url = 'localhost:12001'

triton_client = input_client.InferenceServerClient(url=triton_url, verbose=False)


path = '/home/longduong/projects/face_project/scrfd/t1.jpg'
img = cv2.imread(path)
img = cv2.resize(img, (640, 640))
processed_image = torch.from_numpy(img).to(torch.float32)
# processed_image = processed_image.permute(2, 0, 1)
# processed_image = (processed_image - 127.5) * 0.0078125
# processed_image = processed_image.unsqueeze(0)

img_t = processed_image.numpy()
# img_t = img.astype(np.float32)
# img_t = torch.from_numpy(img)
# img_t = torch.unsqueeze(0)

inputs = [input_client.InferInput('INPUT__0', img_t.shape, 'FP32')]

inputs[0].set_data_from_numpy(img_t)
# outputs = [input_client.InferRequestedOutput('OUTPUT__0'), input_client.InferRequestedOutput('OUTPUT__1')]
outputs = [input_client.InferRequestedOutput('OUTPUT__0')]
responses = triton_client.infer(model_name,
                                    inputs,
                                    model_version=model_version,
                                    outputs=outputs)
preds_0 = responses.as_numpy('OUTPUT__0')
# preds_1 = responses.as_numpy('OUTPUT__1')

print(preds_0.shape)

# for i in range(50):
#     print("aaaaaaaaa")

# import time
# time.sleep(2)