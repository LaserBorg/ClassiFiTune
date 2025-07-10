# ClassiFiTune

flexible image classifier finetuning in pytorch  
with optional ONNX conversion & prediction.


## Features

- Finetune a wide range of popular torchvision models on your own image dataset
- Simple dataset splitting and preprocessing utilities
- Training, validation, and test accuracy tracking
- ONNX export and inference support
- Jupyter notebook workflow for easy experimentation


## available models for finetuning

- resnet18
- resnet50
- alexnet
- vgg11_bn
- squeezenet
- densenet121
- inception_v3
- mobilenet_v2
- mobilenet_v3_large
- regnet_y_16gf
- efficientnet_v2_s
- efficientnet_v2_m
- convnext_base
- swin_v2_b

A full list of torchvision classifiers with weights, accuracy, parameter count, and more is available in
[Classifier-models.xlsx](__info_/__info__/Classifier-models.xlsx)


## installation

```bash
# create python venv
python3 -m venv ~/venvs/venv312
source ~/venvs/venv312/bin/activate

# Install PyTorch
pip install torch torchvision
# pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# Install other requirements (opencv, matplotlib, onnx, jupyter, etc.)
pip install -r requirements.txt

# Download example dataset
wget -O example_dataset.zip "https://1drv.ms/u/s!AgIimkCJY7fxh0wwll9jLBqr_nMQ?e=t72EgV"
unzip example_dataset.zip
mv example_dataset _raw
mv _raw dataset/
rm example_dataset.zip
```

## Usage
- Open and run the Jupyter notebook finetuning.ipynb for a step-by-step workflow.
- Customize model selection, training parameters, and dataset paths as needed.
- The notebook covers data loading, model initialization, training, validation, ONNX export, and inference.

## ToDos
- use best pth file for onnx conversion and inference
- fix & enable albumentations
- get learning rate scheduler type and parameters from yaml
- use yaml parameters for libs functions
- store current epoch and loss in metadata
- save metadata as yaml
- update readme

- [Object detection](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)