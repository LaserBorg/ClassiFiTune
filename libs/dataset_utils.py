import numpy as np
import cv2
import PIL.Image as Image
import torch
from torchvision import transforms


def get_transforms(input_size, scale_range=[1,1], hflip=0):
    # ImageNet Constants
    mean = [0.485, 0.456, 0.406] 
    std =  [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=scale_range),
            transforms.RandomHorizontalFlip(p=hflip),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),

        # Just normalization, no augmentation for validation
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    return data_transforms


def convert_image_to_cv(img, normalized=True):
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
