import numpy as np
import cv2
import PIL.Image as Image
import torch
from torchvision import transforms
import json

from libs.path_utils import modify_path


def load_imagefile_to_tensor(img_path, transform=None, device="cpu"):
    '''loads image file and returns torch.Tensor'''
    img = Image.open(img_path)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    # create dummy transforms if none are given
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    return img


def get_transforms(input_size, scale_range=[1,1], hflip=0, mean=None, std=None):
    # ImageNet Constants
    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406] 
        std =  [0.229, 0.224, 0.225]

    data_transforms = {
        # for training: Feature augmentation
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=scale_range),
            transforms.RandomHorizontalFlip(p=hflip),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),

        # for validation, testing and inference: Just normalization and conversion, no augmentation
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    return data_transforms


def convert_image_to_cv(img, normalized=True, RGB2BGR=True, device="cpu"):
    '''converts torch.Tensor or PIL.Image to cv2 image'''
    
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
        img = torch.squeeze(img)    # remove batch dimension
        img = (img * 255).byte()    # [0,1] -> [0,255]

        if device.lower() == "cpu":
            img = img.cpu()

        img = img.numpy().transpose(1, 2, 0)        # (C, H, W) -> (H, W, C)

        if RGB2BGR:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB -> BGR
    
    elif isinstance(img, Image.Image):
        # pillow to opencv
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return img


def get_model_data(path):
    '''Reads variables from the JSON file. Returns input_size and class_labels from model checkpoint'''

    path = modify_path(path, attrib="metadata", ext="json")

    with open(path, 'r') as f:
        data = json.load(f)
        input_size = data['input_size']
        class_labels = data['class_labels']
    
    return input_size, class_labels
