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

NEW: TinyViT:
- TinyViT-5M, ImageNet-1K 224
- TinyViT-11M, ImageNet-1K 224
- TinyViT-21M, ImageNet-1K 224
- TinyViT-21M, ImageNet-22K 224 (22K classes)
- TinyViT-21M, ImageNet-1K 384 (higher resolution)
- TinyViT-21M, ImageNet-1K 512 (highest resolution)

using torchvision models:
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
sudo apt update && sudo apt install python3-venv
python3 -m venv ~/.venv
source ~/.venv/bin/activate

# Install PyTorch (see https://pytorch.org/get-started/locally/)
# CUDA: check compute version https://developer.nvidia.com/cuda-gpus, https://developer.nvidia.com/cuda-legacy-gpus
# e.g. for GTX 960 (Maxwell: compute 5.2) <= CUDA 11.8
# pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch torchvision

# Install other requirements (opencv, matplotlib, onnx, jupyter, etc.)
pip install -r requirements.txt


# Download checkpoint, e.g. 384 px
wget -P checkpoints/ "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_384_distill.pth"


# Download example dataset from OneDrive (wget doesn't work for OneDrive)
# https://1drv.ms/u/s!AgIimkCJY7fxh0wwll9jLBqr_nMQ?e=t72EgV
mv example_dataset.zip dataset/_raw && cd dataset/_raw
unzip example_dataset.zip && rm example_dataset.zip


jupyter notebook  # then open finetune_tinyvit.ipynb finetune_torchvision.ipynb and run it
```


- [Object detection](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)



## from the TinyViT repo (https://github.com/wkcn/tinyvit)

TinyViT is a new family of **tiny and efficient** vision transformers pretrained on **large-scale** datasets with our proposed **fast distillation framework**. The central idea is to **transfer knowledge** from **large pretrained models** to small ones. The logits of large teacher models are sparsified and stored in disk in advance to **save the memory cost and computation overheads**.

:rocket: TinyViT with **only 21M parameters** achieves **84.8%** top-1 accuracy on ImageNet-1k, and **86.5%** accuracy under 512x512 resolutions.

### Model Zoo

download pth file and put it into ./checkpoints/

Model                                      | Pretrain | Input | Acc@1 | Acc@5 | #Params | MACs | FPS  | 22k Model | 1k Model
:-----------------------------------------:|:---------|:-----:|:-----:|:-----:|:-------:|:----:|:----:|:---------:|:--------:
TinyViT-5M ![](./.figure/distill.png)       | IN-22k   |224x224| 80.7  | 95.6  | 5.4M    | 1.3G | 3,060|[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22k_distill.pth)/[config](./configs/22k_distill/tiny_vit_5m_22k_distill.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22k_distill.log)|[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22kto1k_distill.pth)/[config](./configs/22kto1k/tiny_vit_5m_22kto1k.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22kto1k_distill.log)
TinyViT-11M ![](./.figure/distill.png)      | IN-22k   |224x224| 83.2  | 96.5  | 11M     | 2.0G | 2,468|[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22k_distill.pth)/[config](./configs/22k_distill/tiny_vit_11m_22k_distill.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22k_distill.log)|[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22kto1k_distill.pth)/[config](./configs/22kto1k/tiny_vit_11m_22kto1k.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22kto1k_distill.log)
TinyViT-21M ![](./.figure/distill.png)      | IN-22k   |224x224| 84.8  | 97.3  | 21M     | 4.3G | 1,571|[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22k_distill.pth)/[config](./configs/22k_distill/tiny_vit_21m_22k_distill.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22k_distill.log)|[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_distill.pth)/[config](./configs/22kto1k/tiny_vit_21m_22kto1k.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_distill.log)
TinyViT-21M-384 ![](./.figure/distill.png)  | IN-22k   |384x384| 86.2  | 97.8  | 21M     | 13.8G| 394  | - |[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_384_distill.pth)/[config](./configs/higher_resolution/tiny_vit_21m_224to384.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_384_distill.log)
TinyViT-21M-512 ![](./.figure/distill.png)  | IN-22k   |512x512| 86.5  | 97.9  | 21M     | 27.0G| 167  | - |[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_512_distill.pth)/[config](./configs/higher_resolution/tiny_vit_21m_384to512.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_512_distill.log)
TinyViT-5M                                 | IN-1k    |224x224| 79.1  | 94.8  | 5.4M    | 1.3G | 3,060| - |[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_1k.pth)/[config](./configs/1k/tiny_vit_5m.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_1k.log)
TinyViT-11M                                | IN-1k    |224x224| 81.5  | 95.8  | 11M     | 2.0G | 2,468| - |[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_1k.pth)/[config](./configs/1k/tiny_vit_11m.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_1k.log)
TinyViT-21M                                | IN-1k    |224x224| 83.1  | 96.5  | 21M     | 4.3G | 1,571| - |[link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_1k.pth)/[config](./configs/1k/tiny_vit_21m.yaml)/[log](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_1k.log)



## TinyViT Fine-tuning Script

Features:
- Load pretrained weights from checkpoint
- Two-stage training: frozen backbone (just head) + full fine-tuning (deep training)
- Early stopping
- TensorBoard logging and Confusion matrix
- Mixed precision training

improvements based on the official TinyViT implementation:

### 1. Learning Rate Strategy (Major Improvement)
- **Cosine annealing with warmup**
- **Layer-wise learning rate decay** (0.8)
- **warmup periods** (1 epoch for stage 1, 2 epochs for stage 2)
- **Low base learning rates** (1e-3 for head, 2.5e-4 for deep training)

### 2. Weight Decay Optimization
- **Selective weight decay** - excludes bias and normalization layers
- **low weight decay** (1e-8) for fine-tuning vs standard 0.05
- **Proper parameter grouping** following TinyViT's approach

### 3. Training Stability Improvements
- **Gradient clipping** (max_norm=5.0) prevents exploding gradients
- **BatchNorm in eval mode** during training
- **Better mixed precision handling** with unscaling for gradient clipping

### 4. Enhanced Data Augmentation
- A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
- A.Rotate(limit=15, p=0.5)
- A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.25)  # Random erasing
- A.OneOf([GaussNoise, GaussianBlur], p=0.2)  # noise augmentation

### 5. Training Configuration Updates
- **15 epochs frozen + 5 epochs fine-tuning** 
- **Increased patience** (7)

### 6. Key Functions
1. `set_weight_decay()` - Proper weight decay handling
2. `get_cosine_scheduler_with_warmup()` - Advanced LR scheduling  
3. Enhanced `train_epoch()` with gradient clipping and BN eval mode



## Citation

If TinyViT is helpful for you, please consider to cite it. :mega: Thank you! :)

```bibtex
@InProceedings{tiny_vit,
  title={TinyViT: Fast Pretraining Distillation for Small Vision Transformers},
  author={Wu, Kan and Zhang, Jinnian and Peng, Houwen and Liu, Mengchen and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
  booktitle={European conference on computer vision (ECCV)},
  year={2022}
}
```

## Licenses

- [License](./LICENSE)
