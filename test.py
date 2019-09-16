from PIL import Image
import torch
import numpy as np
from torchvision.transforms import *
# from natsort import natsorted
# a = ["awe/1.jpg", "awe/11.jpg", "awe/12.jpg", "awe/2.jpg"]
# print(list(natsorted(a)))
img = Image.open("../gesture_recognition/20bn-jester-v1_optical_flow/1/1.jpg")
transform = Compose([
        CenterCrop(84),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

imgs = []
img = transform(img)
img = torch.unsqueeze(img, 0)
imgs.append(img)
imgs.append(img)
imgs = torch.cat(imgs)
print(img.shape)
print(imgs.shape)

