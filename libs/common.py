import os
import random
import json
import yaml
import time

import cv2
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


mean = IMAGENET_DEFAULT_MEAN  # [0.485, 0.456, 0.406]
std  = IMAGENET_DEFAULT_STD   # [0.229, 0.224, 0.225]


# ------------------------------------------------------------------
# read and write yaml or json

def load_dict(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"file not found: {config_path}.")
    elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError('Unsupported config file format. Use .yaml, .yml, or .json')

def dump_dict(data, config_path, update=False, indent=2):
    if update:
        with open(config_path) as f:
            data_old = json.load(f)
        data_old.update(data)
        data = data_old
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=indent)
    elif config_path.endswith('.json'):
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=indent)
    else:
        raise ValueError('Unsupported config file format. Use .yaml, .yml, or .json')

# ------------------------------------------------------------------
# HELPERS

def date2string():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

# ------------------------------------------------------------------
# CONVERSION

def ensure_tuple(val, length=None, dtype=None):
    # Converts list/int/float to tuple, optionally checks length and dtype
    if isinstance(val, (list, np.ndarray)):
        val = tuple(val)
    elif isinstance(val, (int, float)):
        val = (val, val)
    if length and len(val) != length:
        raise ValueError(f"Expected tuple of length {length}, got {val}")
    if dtype and not all(isinstance(x, dtype) for x in val):
        raise ValueError(f"Expected tuple of type {dtype}, got {val}")
    return val

def tensor_to_array(image, idx=0, mean=mean, std=std, device="cpu"):
    # img = img_or_tensor_to_array(image, RGB2BGR=False, normalized=True)
    img = image.cpu().data[idx].numpy().transpose((1, 2, 0))  # [-1 |  1 ]
    img = std * img + mean  # denormalize using config values
    img = np.clip(img*255, 0, 255).astype(np.uint8)  # convert to uint8
    return img

# converts torch.Tensor or PIL.Image to cv2 image
def img_or_tensor_to_array(img, normalized=True, RGB2BGR=False, device="cpu"):
    if isinstance(img, torch.Tensor):
        if normalized:
            # unnormalize mean and std for visualization
            inv_mean = [-m / s for m, s in zip(mean, std)]
            inv_std = [1 / s for s in std]

            unnorm_transform = transforms.Normalize(mean=inv_mean, std=inv_std)
            img = unnorm_transform(img)

        # torch.Tensor to opencv
        img = torch.squeeze(img)    # remove batch dimension
        img = (img * 255).byte()    # [0,1] -> [0,255]
    
        if device.lower() == "cpu":
            img = img.cpu()
    
        img = img.numpy().transpose(1, 2, 0)        # (C, H, W) -> (H, W, C)
    
    elif isinstance(img, Image.Image):
        # pillow to opencv
        img = np.array(img)

    if RGB2BGR and isinstance(img, np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img

# converts PIL.Image to torch.Tensor
def img_to_tensor(img, transform=None, device="cpu"):
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # apply transforms
    if transform is None:  # create dummy transforms
        transform = transforms.Compose([transforms.ToTensor()])
    img_transformed = transform(img)

    # Add a batch dimension and move to CPU/GPU
    return img_transformed.unsqueeze(0).to(device)

# ------------------------------------------------------------------
# VISUALISATION

def imshow(title, image, width=640, waitkey=0):
    img_rgb = img_or_tensor_to_array(image)

    # Maintain aspect ratio for a fixed width
    original_height, original_width, _ = img_rgb.shape
    if original_width > 0:
        aspect_ratio = original_height / original_width
        new_height = int(width * aspect_ratio)
        img_rgb = cv2.resize(img_rgb, (width, new_height))

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, img_bgr)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()


def imshow_notebook(title, image):
    img = img_or_tensor_to_array(image)

    ax = plt.subplot(2, 2, 1)
    ax.axis('off')
    if title:
        ax.set_title(title)
    plt.imshow(img)
    plt.pause(0.001)

# ------------------------------------------------------------------
# PATH UTILS

def get_random_image(images_dir, ext=".jpg", seed=None):
    """
    Selects a random image from a directory.
    An optional seed can be provided for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
    # List all files in the directory
    all_files = os.listdir(images_dir)
    # Filter for given file extension
    filtered_files = [f for f in all_files if f.endswith(ext)]
    if not filtered_files:
        return None
    # Select a random file
    random_image = random.choice(filtered_files)
    # Return the full path to the random image
    return os.path.join(images_dir, random_image)

def get_filelist_from_dir(path, recursive=False, filetypes=["jpg", "jpeg", "png"], max_count=False):
    '''create a list of all images in the directory. if the path is a file, return a list with only the file.'''

    if os.path.isdir(path):
        # create index of all images to predict
        filelist = []
        for (root,dirs,files) in os.walk(path, topdown=True):
                for i, file in enumerate(files):
                    if file.split(".")[-1] in filetypes:
                        img_path = os.path.join(root,file)
                        filelist.append(img_path)

                if not recursive:
                    break

        # sample random images only if there are more than max_count
        if max_count >= 1:
            if len(filelist) > max_count:
                filelist = random.sample(filelist, max_count)
    
    # error handling
    elif os.path.isfile(path):
        print(f"[WARNING] {path} should be a directory. Proceding anyway.")
        filelist = path

    return filelist

def get_filename_from_path(path):
    filename = "image" if path is None else os.path.splitext(os.path.basename(path))[0]
    return filename

def add_name_attribute(imagepath, target_dir="images/previews", attribute="shape"):
    '''changes output directory and adds attribute to filename'''

    # TODO: make output directory change optional
    basename = os.path.basename(imagepath)
    namesplit = os.path.splitext(basename)
    savename = namesplit[0] + "_" + attribute + namesplit[1]
    savepath = os.path.join(target_dir, savename)
    return savepath

def modify_path(path, attrib=None, out_dir= None, ext=None):
    root, file = os.path.split(path)
    name, extension = os.path.splitext(file)

    if out_dir is not None:
        root = out_dir

    if ext is not None:
        extension = ext
        
    if attrib is not None:
        attrib = "_" + attrib

    result_path = os.path.join(root, name + attrib + "." + extension)
    return result_path
