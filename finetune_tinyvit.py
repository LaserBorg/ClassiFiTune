# # TinyViT Training 
# with Pretrained Weights and Two-Stage Fine-tuning

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="libs.tiny_vit.tiny_vit")

import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report
#from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from libs.common import load_dict, dump_dict
from libs.albumentations_utils import AlbumentationsTransform
from libs.tiny_vit.tiny_vit_train import get_model, load_pretrained_weights, freeze_backbone, unfreeze_all, \
    set_weight_decay, get_cosine_scheduler_with_warmup, train_epoch, validate_epoch, plot_confusion_matrix, EarlyStopping

# ## Settings

settings = load_dict('settings_tinyvit.yaml')

# Load base model configuration from tiny_vit.json
base_model = settings['base_model']
variants_path = "libs/tiny_vit/tiny_vit.json"
variants = load_dict(variants_path)

if base_model not in variants:
    raise ValueError(f"Base model '{base_model}' not found in available variants: {list(variants.keys())}")

base_model_info = variants[base_model]

# Extract model configuration and paths from base model
model_config = base_model_info['model_config']
pretrained_path = base_model_info['weights']
img_size = model_config['img_size']

print(f"Using base model: {base_model}")
print(f"Image size: {img_size}x{img_size}")
print(f"Pretrained weights: {pretrained_path}")

# Directory settings
data_dir = settings['data_dir']
output_dir = settings['output_dir']
logs_dir = settings['logs_dir']

# Output file names
checkpoint_name = settings['checkpoint_name']
checkpoint_savepath = os.path.join(output_dir, checkpoint_name)
checkpoint_base = os.path.splitext(checkpoint_name)[0]  # Remove .pth extension

# Derived directory paths
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
val_dir = os.path.join(data_dir, "val")

# Create directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Count classes
num_classes = sum(os.path.isdir(os.path.join(train_dir, entry)) for entry in os.listdir(train_dir))
print("num_classes:", num_classes)

# Normalization settings
mean = np.array(settings['mean'])  # np.array(IMAGENET_DEFAULT_MEAN)
std = np.array(settings['std'])    # np.array(IMAGENET_DEFAULT_STD)

# Stage 1 training parameters (Head training with frozen backbone)
stage1_epochs = settings['stage1']['epochs']
stage1_lr = float(settings['stage1']['learning_rate'])
stage1_warmup_epochs = settings['stage1']['warmup_epochs']
stage1_min_lr = float(settings['stage1']['min_lr'])
stage1_batch_size = settings['stage1']['batch_size']

# Stage 2 training parameters (Full fine-tuning)  
stage2_epochs = settings['stage2']['epochs']
stage2_lr = float(settings['stage2']['learning_rate'])
stage2_warmup_epochs = settings['stage2']['warmup_epochs']
stage2_min_lr = float(settings['stage2']['min_lr'])
stage2_batch_size = settings['stage2']['batch_size']

# Additional training settings
layer_lr_decay = float(settings['layer_lr_decay'])
weight_decay = float(settings['weight_decay'])
use_amp = settings['use_amp']
patience = settings['patience']
gradient_clip_norm = float(settings['gradient_clip_norm'])
eval_bn = settings['eval_bn']

# Optimizer settings
optimizer_config = settings['optimizer']
optimizer_config['eps'] = float(optimizer_config['eps'])
optimizer_config['betas'] = [float(b) for b in optimizer_config['betas']]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# TensorBoard writer
writer = SummaryWriter(logs_dir)

# ## transforms and dataloaders
# Get augmentation settings from config
aug_config = settings['augmentation']

train_transform = A.Compose([
    A.Resize(img_size, img_size),
    A.HorizontalFlip(p=aug_config['horizontal_flip']),
    A.Rotate(limit=aug_config['rotation_limit'], p=aug_config['rotation_prob']),
    A.ColorJitter(
        brightness=aug_config['color_jitter']['brightness'], 
        contrast=aug_config['color_jitter']['contrast'], 
        saturation=aug_config['color_jitter']['saturation'], 
        hue=aug_config['color_jitter']['hue'], 
        p=aug_config['color_jitter']['prob']
    ),
    A.OneOf([
        A.GaussNoise(
            p=aug_config['noise_and_blur']['gaussian_noise']['prob']
        ),
        A.GaussianBlur(
            blur_limit=aug_config['noise_and_blur']['gaussian_blur']['blur_limit'], 
            p=aug_config['noise_and_blur']['gaussian_blur']['prob']
        ),
    ], p=aug_config['noise_and_blur']['prob']),
    A.CoarseDropout(
        num_holes=aug_config['coarse_dropout']['max_holes'], 
        max_h_size=aug_config['coarse_dropout']['max_height'], 
        max_w_size=aug_config['coarse_dropout']['max_width'], 
        p=aug_config['coarse_dropout']['prob']
    ),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

eval_transform = A.Compose([
    A.Resize(img_size, img_size),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

# Datasets and Dataloaders
train_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "train"),
    transform=AlbumentationsTransform(train_transform)
)

val_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "val"),
    transform=AlbumentationsTransform(eval_transform)
)

test_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "test"),
    transform=AlbumentationsTransform(eval_transform)
)

train_loader = DataLoader(train_dataset, batch_size=stage1_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=stage1_batch_size)
test_loader = DataLoader(test_dataset, batch_size=stage1_batch_size)

# Get class names and create class labels mapping
class_labels = train_dataset.classes
print(f"Classes found: {class_labels}")

# Build model using integrated model configuration
model = get_model(model_config, num_classes, img_size, device)

# Load pretrained weights
model = load_pretrained_weights(model, pretrained_path, num_classes, device)

# Initialize mixed precision scaler
scaler = GradScaler('cuda') if use_amp else None

# Loss function
criterion = nn.CrossEntropyLoss()

# ## STAGE 1: Training classifier head with frozen backbone

# Freeze backbone and train only the head
model = freeze_backbone(model)

# Optimizer and scheduler for stage 1 with improved weight decay
param_groups = set_weight_decay(model, weight_decay=weight_decay)
optimizer = optim.AdamW(
    param_groups, 
    lr=stage1_lr, 
    eps=optimizer_config['eps'], 
    betas=optimizer_config['betas']
)

# Cosine annealing scheduler with warmup
total_steps = stage1_epochs * len(train_loader)
warmup_steps = stage1_warmup_epochs * len(train_loader)
scheduler = get_cosine_scheduler_with_warmup(optimizer, warmup_steps, total_steps, stage1_min_lr)

# Early stopping for stage 1
early_stopping = EarlyStopping(patience=patience)

# Stage 1 training loop
global_step = 0
for epoch in range(stage1_epochs):
    print(f"\nEpoch {epoch+1}/{stage1_epochs}")
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp, eval_bn=eval_bn, gradient_clip_norm=gradient_clip_norm)
    
    # Validate
    val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
    
    # Log to TensorBoard
    writer.add_scalar('Stage1/Train_Loss', train_loss, epoch)
    writer.add_scalar('Stage1/Train_Acc', train_acc, epoch)
    writer.add_scalar('Stage1/Val_Loss', val_loss, epoch)
    writer.add_scalar('Stage1/Val_Acc', val_acc, epoch)
    writer.add_scalar('Stage1/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Step scheduler
    scheduler.step()
    
    # Early stopping check
    if early_stopping(val_loss, model):
        print(f"Early stopping triggered at epoch {epoch+1}")
        break
    
    global_step += 1

# ## STAGE 2: Fine-tuning entire network

# Unfreeze all parameters and fine-tune
model = unfreeze_all(model)

# Create new data loaders with smaller batch size for stage 2
train_loader_stage2 = DataLoader(train_dataset, batch_size=stage2_batch_size, shuffle=True)
val_loader_stage2 = DataLoader(val_dataset, batch_size=stage2_batch_size)

# New optimizer and scheduler for stage 2 with lower learning rate
param_groups = set_weight_decay(model, weight_decay=weight_decay)
optimizer = optim.AdamW(
    param_groups, 
    lr=stage2_lr, 
    eps=optimizer_config['eps'], 
    betas=optimizer_config['betas']
)

# Cosine annealing scheduler with warmup for stage 2
total_steps = stage2_epochs * len(train_loader_stage2)  
warmup_steps = stage2_warmup_epochs * len(train_loader_stage2)
scheduler = get_cosine_scheduler_with_warmup(optimizer, warmup_steps, total_steps, stage2_min_lr)

# Reset early stopping for stage 2
early_stopping = EarlyStopping(patience=patience)

# Stage 2 training loop
for epoch in range(stage2_epochs):
    print(f"\nEpoch {epoch+1}/{stage2_epochs}")
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader_stage2, criterion, optimizer, device, scaler, use_amp, eval_bn=eval_bn, gradient_clip_norm=gradient_clip_norm)
    
    # Validate
    val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader_stage2, criterion, device)
    
    # Log to TensorBoard
    writer.add_scalar('Stage2/Train_Loss', train_loss, global_step + epoch)
    writer.add_scalar('Stage2/Train_Acc', train_acc, global_step + epoch)
    writer.add_scalar('Stage2/Val_Loss', val_loss, global_step + epoch)
    writer.add_scalar('Stage2/Val_Acc', val_acc, global_step + epoch)
    writer.add_scalar('Stage2/Learning_Rate', optimizer.param_groups[0]['lr'], global_step + epoch)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Step scheduler
    scheduler.step()
    
    # Early stopping check
    if early_stopping(val_loss, model):
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# Save final model with metadata
model_save_dict = {
    'model_state_dict': model.state_dict(),
    'num_classes': num_classes,
    'class_labels': class_labels,
    'img_size': img_size,
    'pretrained_path': pretrained_path
}

torch.save(model_save_dict, checkpoint_savepath)

print(f"\nModel saved to {checkpoint_savepath}")



## EVALUATION ON TEST SET

# Evaluate with OOM-safe fallback
model.eval()
test_correct, test_total = 0, 0
test_preds, test_labels = [], []

from contextlib import nullcontext
from torch.utils.data import DataLoader

# Helper that runs the evaluation loop for a given loader and device
def run_evaluation(loader, device_):
    test_correct_local, test_total_local = 0, 0
    preds_local, labels_local = [], []
    # enable autocast on CUDA to reduce memory usage if available
    use_amp = (device_.type == 'cuda') and getattr(torch, 'amp', None) is not None
    if use_amp:
        # Use the new API: torch.amp.autocast with explicit device_type
        amp_ctx_factory = lambda: torch.amp.autocast(device_type='cuda')
    else:
        amp_ctx_factory = nullcontext

    with torch.no_grad():
        with amp_ctx_factory():
            for inputs, labels in tqdm(loader, desc=f"Testing ({device_.type})"):
                inputs, labels = inputs.to(device_), labels.to(device_)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_correct_local += (preds == labels).sum().item()
                test_total_local += labels.size(0)
                preds_local.extend(preds.cpu().numpy())
                labels_local.extend(labels.cpu().numpy())
    return test_correct_local, test_total_local, preds_local, labels_local

# Try the regular test_loader first. On OOM, progressively reduce memory pressure.
try:
    test_correct, test_total, test_preds, test_labels = run_evaluation(test_loader, device)
except RuntimeError as e:
    msg = str(e).lower()
    if 'out of memory' in msg:
        print('CUDA OOM during evaluation. Attempting to reduce memory usage and retry...')
        try:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # 1) Retry on GPU with batch_size=1
            small_loader = DataLoader(test_dataset, batch_size=1)
            try:
                test_correct, test_total, test_preds, test_labels = run_evaluation(small_loader, device)
            except RuntimeError as e2:
                msg2 = str(e2).lower()
                if 'out of memory' in msg2:
                    print('Still OOM on GPU with batch_size=1 — falling back to CPU evaluation.')
                    # 2) Fallback: move model to CPU and evaluate with batch_size=1
                    model_cpu = model.to('cpu')
                    cpu_device = torch.device('cpu')
                    cpu_loader = DataLoader(test_dataset, batch_size=1)
                    test_correct, test_total, test_preds, test_labels = run_evaluation(cpu_loader, cpu_device)
                    # Move model back to original device if it was CUDA
                    try:
                        model.to(device)
                    except Exception:
                        pass
                else:
                    raise
        except Exception:
            raise
    else:
        # Not an OOM — re-raise so user sees the original error
        raise

# Compute accuracy and log results (same as before)
test_acc = test_correct / test_total if test_total > 0 else 0.0
print(f"Test Accuracy: {test_acc:.4f}")

# Log final test accuracy
writer.add_scalar('Final/Test_Acc', test_acc, 0)

# Generate confusion matrix
cm_path = os.path.join(logs_dir, f'{checkpoint_base}_confusion_matrix.png')
plot_confusion_matrix(test_labels, test_preds, class_labels, cm_path)
print(f"Confusion matrix saved to {cm_path}")

# Generate classification report
report = classification_report(test_labels, test_preds, target_names=class_labels)
print("\nClassification Report:")
print(report)

# Save classification report
report_path = os.path.join(logs_dir, f'{checkpoint_base}_classification_report.txt')
with open(report_path, 'w') as f:
    f.write(report)

# Also save the training information as metadata
summary_info = {
    'final_test_accuracy': float(test_acc),
    'num_classes': num_classes,
    'class_labels': class_labels,
    'model_architecture': 'TinyViT',
    'base_model': base_model,
    'pretrained_checkpoint': pretrained_path,
    'training_stages': {
        'stage1_epochs': stage1_epochs,
        'stage1_lr': stage1_lr,
        'stage2_epochs': stage2_epochs,
        'stage2_lr': stage2_lr
    },
    'image_size': img_size,
    'stage1_batch_size': stage1_batch_size,
    'stage2_batch_size': stage2_batch_size
}

summary_path = os.path.join(output_dir, f'{checkpoint_base}_metadata.json')
dump_dict(summary_info, summary_path)
print(f"Training summary saved to {summary_path}")

writer.close()
print("Training complete!")



# ----------------------------------------------------------
# ## convert TinyViT model to ONNX

# Use a single batch from the test loader as example input
model.eval()

# Ensure model and inputs are on the same device
model.to(device)

# Get one batch from test_loader (already uses AlbumentationsTransform -> ToTensorV2)
inputs, _ = next(iter(test_loader))
inputs = inputs.to(device)

onnx_path = os.path.join(output_dir, checkpoint_base + ".onnx")

# Export the model to ONNX
# - dynamic_axes: batch dimension is dynamic for input and output
# - opset_version can be adjusted as needed (default 17 is commonly available)
# - do_constant_folding can be enabled for optimization

export_kwargs = dict(
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=17,
    do_constant_folding=False,
    verbose=False
)

# Try exporting on the current device first. If CUDA OOM occurs, retry on CPU.
try:
    with torch.no_grad():
        torch.onnx.export(model, inputs, onnx_path, **export_kwargs)
    print(f"ONNX model exported to {onnx_path} on device={device}")
except RuntimeError as e:
    msg = str(e).lower()
    if 'out of memory' in msg or 'cuda' in msg and 'out of memory' in msg:
        print('CUDA out of memory during ONNX export. Attempting CPU fallback...')
        try:
            # free GPU caches
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            # Move model and inputs to CPU and retry export
            model_cpu = model.to('cpu')
            inputs_cpu = inputs.cpu()
            with torch.no_grad():
                torch.onnx.export(model_cpu, inputs_cpu, onnx_path, **export_kwargs)
            print(f"ONNX model exported to {onnx_path} on CPU")
            # Move model back to original device
            model.to(device)
        except Exception as e2:
            print('CPU export also failed:', e2)
    else:
        print('Failed to export ONNX:', e)


# ## Quantize the ONNX model
try:
    import onnxruntime
    from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader

    # Create a calibration data reader
    class QuantizationDataReader(CalibrationDataReader):
        def __init__(self, dataloader):
            self.dataloader = dataloader
            self.iterator = iter(dataloader)
            # Get the input name from the ONNX model
            session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            self.input_name = session.get_inputs()[0].name

        def get_next(self):
            try:
                inputs, _ = next(self.iterator)
                return {self.input_name: inputs.cpu().numpy()}
            except StopIteration:
                return None

    # Set paths for quantized model
    quantized_onnx_path = os.path.join(output_dir, checkpoint_base + "_quant.onnx")

    # Create calibration data reader
    calibration_data_reader = QuantizationDataReader(test_loader)

    # Perform static quantization
    print("\nStarting ONNX model quantization...")
    quantize_static(
        model_input=onnx_path,
        model_output=quantized_onnx_path,
        calibration_data_reader=calibration_data_reader,
        quant_format=onnxruntime.quantization.QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
        nodes_to_exclude=[]
    )
    print(f"Quantized ONNX model saved to {quantized_onnx_path}")

    # Compare file sizes
    original_size = os.path.getsize(onnx_path)
    quantized_size = os.path.getsize(quantized_onnx_path)
    print(f"Original ONNX model size: {original_size / 1024 / 1024:.2f} MB")
    print(f"Quantized ONNX model size: {quantized_size / 1024 / 1024:.2f} MB")
    print(f"File size reduction: {100 * (1 - quantized_size / original_size):.2f}%")

except Exception as e:
    print(f"ONNX quantization failed: {e}")