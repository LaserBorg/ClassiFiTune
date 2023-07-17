import torch
from torchvision import transforms
import os
import cv2	
import numpy as np
import random

# Inference:
from PIL import Image

# Helper Functions:
from libs.dataset_utils import get_transforms, convert_image_to_cv


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




# data directory with [train, val, test] dirs
data_dir = "./dataset/views_split"

model_name = "mobilenet_v3_large"
checkpoint_path = f"checkpoint/{model_name}.pt"
input_size = 224
class_names = ['background', 'down', 'front', 'left', 'missing', 'up']


visualize = True

dir_path = './dataset/views_unlabeled'
max_images = 50

force_CPU = True
device = torch.device("cuda:0" if torch.cuda.is_available() and not force_CPU else "cpu")
print("inference running on device:", device)

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
        img = convert_image_to_cv(img)
        cv2.imshow(predicted_label, img)
        cv2.waitKey(1)

cv2.waitKey(0)
cv2.destroyAllWindows()
