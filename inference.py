import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import cv2	
import numpy as np
import random

# Inference:
from torch.utils.data import Dataset
from PIL import Image

# Helper Functions:
from libs.model_definitions import initialize_model
from libs.train_model import train_model
from libs.dataset_helpers import get_transforms


def load_image(img_path, device="cpu"):
    '''returns
    img: torch.Tensor'''
    img = Image.open(img_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)
    return img


def predict(model, img):
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        return preds


def convert_image_format(img, normalized=True):
    mean = [0.485, 0.456, 0.406] 
    std =  [0.229, 0.224, 0.225]

    if normalized:
        # unnormalize mean and std for visualization
        inv_mean = [-m / s for m, s in zip(mean, std)]
        inv_std = [1 / s for s in std]

        unnorm_transform = transforms.Normalize(mean=inv_mean, std=inv_std)
        img = unnorm_transform(img)

    if isinstance(img, torch.Tensor):
        # torch.Tensor to opencv
        img = torch.squeeze(img)  # remove batch dimension
        img = (img * 255).byte()    # [0,1] -> [0,255]
        img = img.numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB -> BGR
    
    elif isinstance(img, Image.Image):
        # pillow to opencv
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return img


# data directory with [train, val, test] dirs
data_dir = "./dataset/views_split"

model_name = "mobilenet_v3_large"
checkpoint_path = f"checkpoint/{model_name}.pt"
input_size = 224
class_names = ['background', 'down', 'front', 'left', 'missing', 'up']


visualize = True

dir_path = './dataset/views_unlabeled'
max_images = 10

force_CPU = True
device = torch.device("cuda:0" if torch.cuda.is_available() and not force_CPU else "cpu")
print(device)

model = torch.load(checkpoint_path)
model = model.to(torch.device(device))

# set dropout and batch normalization layers to evaluation mode before running inference
model.eval()


data_transforms = get_transforms(input_size)
test_transform= data_transforms["val"]


model = model.to(device)
model.eval()

# create index of all images to predict
filelist = []
for (root,dirs,files) in os.walk(dir_path, topdown=True):
        for i, file in enumerate(files):
            img_path = os.path.join(root,file)
            filelist.append(img_path)

filelist = random.sample(filelist, max_images)


for img_path in filelist:
    img = load_image(img_path, device=device)
    predictions = predict(model, img)
    predicted_label = class_names[predictions[0]]

    print("predicted label:", predicted_label)

    if visualize:
        img = convert_image_format(img)
        cv2.imshow(predicted_label, img)
        cv2.waitKey(0)

cv2.destroyAllWindows()
