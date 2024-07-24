# !interpreter [optional-arg]
# -*- coding: utf-8 -*-

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(BASE_DIR)
from locust import HttpUser, task
import base64


def raw_image_to_base64(img_url: str) -> str:
  with open(img_url, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
  return encoded_string

with open("tripham_cropface.jpg", "rb") as image_file:
    img_base64 = base64.b64encode(image_file.read()).decode("utf-8")


class QuickstartUser(HttpUser):
  INTERNAL_API = "/internal/wedjat-face-gateway-staging/v1"
  EXTRACT_INFO = "crop-face-selfie"
  image = raw_image_to_base64(os.path.join(BASE_DIR, "images/cavet_xe.jpg"))

  @task(1)
  def check_id_card(self):
    res = self.client.post(f"{self.INTERNAL_API}/{self.EXTRACT_INFO}", json={
    "request_id": "hehe-id",
    "img_b64": img_base64,
    "request_timestamp": 20,
    "agent_id": 20
})