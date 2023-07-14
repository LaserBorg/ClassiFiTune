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
