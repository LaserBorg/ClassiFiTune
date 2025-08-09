import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import PIL.Image as Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_ATransform(img_size):
    eval_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=IMAGENET_DEFAULT_MEAN, 
            std=IMAGENET_DEFAULT_STD),
            ToTensorV2()])
    return AlbumentationsTransform(eval_transform)


class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        return self.transform(image=img)["image"]


def get_transforms(input_size, scale_range=[1,1], hflip=0, mean=None, std=None):
    # ImageNet Constants
    if mean is None or std is None:
        mean = IMAGENET_DEFAULT_MEAN 
        std =  IMAGENET_DEFAULT_STD

    # # Use albumentations for training augmentations
    # train_transform = A.Compose([
    #     A.RandomResizedCrop(input_size, input_size, scale=tuple(scale_range)),
    #     A.HorizontalFlip(p=hflip),
    #     A.Normalize(mean=mean, std=std),
    #     ToTensorV2(),
    # ])
    # val_transform = A.Compose([
    #     A.Resize(input_size, input_size),
    #     A.CenterCrop(input_size, input_size),
    #     A.Normalize(mean=mean, std=std),
    #     ToTensorV2(),
    # ])
    # # Albumentations returns dict with 'image' key, so wrap for PyTorch compatibility
    # def alb_wrapper(transform):
    #     def wrapped(img):
    #         img = np.array(img)
    #         return transform(image=img)['image']
    #     return wrapped
    # data_transforms = {
    #     'train': alb_wrapper(train_transform),
    #     'val': alb_wrapper(val_transform),
    # }

    # OLD VERSION without albumentations
    from torchvision import transforms
    data_transforms = {
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
