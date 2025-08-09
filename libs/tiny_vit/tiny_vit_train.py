import os
from tqdm import tqdm
import torch
from torch.amp import autocast
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from libs.tiny_vit.tiny_vit import TinyViT


def get_model(model_config, num_classes, img_size, device):
    model = TinyViT(
        img_size=img_size,
        in_chans=3,
        num_classes=num_classes,
        embed_dims=model_config["embed_dims"],
        depths=model_config["depths"],
        num_heads=model_config["num_heads"],
        window_sizes=model_config["window_sizes"],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=model_config.get("drop_path_rate", 0.1),
        use_checkpoint=False
    ).to(device)

    return model


def load_pretrained_weights(model, pretrained_path, num_classes, device):
    """
    Load pretrained weights and handle classifier head mismatch
    """
    print(f"Loading pretrained weights from: {pretrained_path}")
    
    if not os.path.exists(pretrained_path):
        print(f"Warning: Pretrained checkpoint not found at {pretrained_path}")
        return model
    
    # Load pretrained state dict
    pretrained_state = torch.load(pretrained_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model' in pretrained_state:
        pretrained_state = pretrained_state['model']
    elif 'state_dict' in pretrained_state:
        pretrained_state = pretrained_state['state_dict']
    
    # Get model state dict
    model_state = model.state_dict()
    
    # Filter out classifier head if number of classes doesn't match
    filtered_state = {}
    classifier_keys = ['head.weight', 'head.bias', 'classifier.weight', 'classifier.bias']
    
    for key, value in pretrained_state.items():
        if any(cls_key in key for cls_key in classifier_keys):
            # Check if classifier dimensions match
            if key in model_state and value.shape != model_state[key].shape:
                print(f"Skipping {key} due to shape mismatch: {value.shape} vs {model_state[key].shape}")
                continue
        
        if key in model_state:
            if value.shape == model_state[key].shape:
                filtered_state[key] = value
            else:
                print(f"Skipping {key} due to shape mismatch: {value.shape} vs {model_state[key].shape}")
    
    # Load filtered state dict
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    
    print(f"Loaded pretrained weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    if missing_keys:
        print(f"Missing keys: {missing_keys[:10]}...")  # Show first 10
    
    return model


def freeze_backbone(model):
    """
    Freeze all parameters except the classifier head
    """
    frozen_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        # Keep head/classifier unfrozen
        if 'head' in name or 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            frozen_params += 1
    
    print(f"Frozen {frozen_params}/{total_params} parameters")
    return model


def unfreeze_all(model):
    """
    Unfreeze all parameters
    """
    for param in model.parameters():
        param.requires_grad = True
    print("Unfrozen all parameters")
    return model


def set_weight_decay(model, weight_decay=1e-8):
    """
    Set weight decay like TinyViT official implementation
    Excludes bias and normalization layers from weight decay
    """
    # Skip weight decay for these parameter types
    skip_keywords = ['bias', 'norm', 'bn', 'ln']
    
    has_decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        
        # Check if parameter should skip weight decay
        skip = False
        for keyword in skip_keywords:
            if keyword in name.lower():
                skip = True
                break
        
        # Also skip weight decay for 1D parameters (typically bias, norm layers)
        if len(param.shape) == 1:
            skip = True
            
        if skip:
            no_decay.append(param)
        else:
            has_decay.append(param)
    
    print(f"Weight decay applied to {len(has_decay)} parameters, skipped for {len(no_decay)} parameters")
    return [{'params': has_decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def get_cosine_scheduler_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-5):
    """
    Create a cosine annealing scheduler with warmup like TinyViT
    """
    from torch.optim.lr_scheduler import LambdaLR
    import math
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_lr = 0.5 * (1 + math.cos(math.pi * progress))
        
        # Ensure minimum learning rate
        base_lr = optimizer.param_groups[0]['lr']
        min_lr_ratio = min_lr / base_lr
        return max(cosine_lr, min_lr_ratio)
    
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, use_amp=True, eval_bn=True, gradient_clip_norm=5.0):
    model.train()
    
    # Set batch norm to eval mode during training (TinyViT best practice)
    if eval_bn:
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.eval()
    
    train_loss, train_correct, train_total = 0.0, 0, 0
    
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler:
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            # Gradient clipping like TinyViT
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()
        
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)
    
    return train_loss / len(train_loader), train_correct / train_total


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return val_loss / len(val_loader), val_correct / val_total, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, class_labels, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()
