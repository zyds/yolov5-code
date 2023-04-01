# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Perform test request
"""

import pprint
import cv2
import numpy as np
import matplotlib.pyplot as plt

import requests

DETECTION_URL = 'http://localhost:5000/v1/object-detection/yolov5s'
IMAGE = 'data/images/zidane.jpg'

# Read image
# with open(IMAGE, 'rb') as f:
#     image_data = f.read()

img = cv2.imread(IMAGE)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.imencode(".jpg", img)[1].tobytes()

response = requests.post(DETECTION_URL, data=img)

img = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)

plt.imshow(img)
plt.show()

# pprint.pprint(response)
