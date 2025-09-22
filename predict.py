# ignore UserWarning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="libs.tiny_vit.tiny_vit")

import os
from PIL import Image
import numpy as np
import torch

from libs.common import load_dict, img_to_tensor, ensure_tuple, get_random_image, imshow
from libs.classifier_inference import ClassifierInference

def print_results(results, title=None):
    if title is not None:
        print(title)
    for label, score in results:
        print(f"  {label}: {score:.2f}")


topk = 3
width = 400
waitkey = 1000

# get a random image from the validation set
image_path = get_random_image('dataset/test/ants', seed=13)
img = Image.open(image_path)


#-------------------------------------------------------
## TINYVIT MODEL

## on pretrained TinyViT (using unified wrapper)
classifier = ClassifierInference(backend='tinyvit', variant="21m_22k_384", device="cuda")
results, _ = classifier.predict(img, topk=topk)

print_results(results, title='TinyViT pretrained topk:')
imshow(f"{results[0][0]} ({results[0][1]:.2f})", img, width=width, waitkey=waitkey)


## on finetuned TinyViT (from metadata)
metadata_path = 'output/tinyvit_21m_384_finetuned_metadata.json'
classifier = ClassifierInference.from_metadata(metadata_path, device='cuda')
print(f"finetuned classes: {classifier.class_labels}")
results, _ = classifier.predict(img, topk=topk)

print_results(results, title='TinyViT finetuned topk:')
imshow(f"{results[0][0]} ({results[0][1]:.2f})", img, width=width, waitkey=waitkey)


# ----------------------------------------------------
# TINYVIT MODEL CONVERTED TO ONNX
onnx_path = os.path.join('output', 'tinyvit_21m_384_finetuned.onnx')
classifier = ClassifierInference(backend='onnx', onnx_path=onnx_path, metadata_path=metadata_path, device='cpu')
results, _ = classifier.predict(img, topk=topk)

print_results(results, title='TinyViT finetuned ONNX topk:')
imshow(f"{results[0][0]} ({results[0][1]:.2f})", img_to_tensor(img, transform=classifier.transform, device=torch.device('cpu')), width=width, waitkey=waitkey)


# ----------------------------------------------------
## TORCHVISION MODELS

settings_path = 'settings_torchvision.yaml'
settings = load_dict(settings_path)

input_size = ensure_tuple(settings['input_size'], length=2, dtype=int)
output_dir = settings['output_dir']
model_type = settings['model_type']
model_name = model_type
output_path = os.path.join(output_dir, model_name + ".pth")
json_path = os.path.join(output_dir, model_name + "_metadata.json")
split_data_dir = settings.get('split_data_dir', None)
mean = np.array(settings['mean'])
std = np.array(settings['std'])

# create unified torchvision inference instance (loads weights and metadata)
classifier = ClassifierInference(backend='torchvision', metadata_path=json_path, model_path=output_path, model_name=model_name, device='cpu')
results, _ = classifier.predict(img, topk=topk)

print_results(results, title='Torchvision topk:')
imshow(f"{results[0][0]} ({results[0][1]:.2f})", img_to_tensor(img, transform=classifier.transform, device=torch.device('cpu')), width=width, waitkey=waitkey)


# ----------------------------------------------------
# TORCHVISION MODEL CONVERTED TO ONNX

onnx_path = os.path.join(output_dir, model_name + ".onnx")
classifier = ClassifierInference(backend='onnx', onnx_path=onnx_path, metadata_path=json_path, input_size=input_size, device='cpu')
results, _ = classifier.predict(img, topk=topk)

print_results(results, title='Torchvision ONNX topk:')
imshow(f"{results[0][0]} ({results[0][1]:.2f})", img_to_tensor(img, transform=classifier.transform, device=torch.device('cpu')), width=width, waitkey=waitkey)
