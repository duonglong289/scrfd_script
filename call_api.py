import requests
import base64
import cv2
from tqdm import tqdm
import time
import numpy as np

# url = 'https://s.mservice.io/internal/wedjat-face-gateway/v1/crop-face-selfie'
url = 'https://s.mservice.io/internal/wedjat-face-gateway-staging/v1/crop-face-selfie'
# with open("small.jpg", "rb") as image_file:
# with open("tripham_cropface.jpg", "rb") as image_file:
with open("t1.jpg", 'rb') as image_file:
    img_base64 = base64.b64encode(image_file.read()).decode("utf-8")

myobj = {
    "request_id": "hehe-id",
    "img_b64": img_base64,
    "request_timestamp": 20,
    "agent_id": 20
}

list_times = []

res = requests.post(url, json = myobj).json()
print(res)

# for i in tqdm(range(100)):
#     start_time = time.time()
#     res = requests.post(url, json = myobj).json()
#     duration = time.time() - start_time
#     list_times.append(duration)


# print(np.percentile(np.array(list_times)))