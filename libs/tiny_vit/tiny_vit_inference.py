import torch
import time
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from libs.tiny_vit.tiny_vit import TinyViT
from libs.common import load_dict
from libs.albumentations_utils import AlbumentationsTransform


class TinyViTInference:
    def __init__(self, variant="21m_22k_224", metadata_path=None, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # load tiny_vit definitions
        variants_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_vit.json")
        self.variants = load_dict(variants_path)
        
        if metadata_path:
            # Load from metadata file (for finetuned models)
            self.load_from_metadata(metadata_path)
        elif variant:
            # Load from predefined variant
            self.set_variant(variant)
    
    def load_from_metadata(self, metadata_path):
        """Load model configuration from training metadata file."""
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        metadata = load_dict(metadata_path)
        
        # Extract information from metadata
        self.variant = f"custom_{os.path.basename(metadata_path).replace('.json', '')}"
        
        # Create variant info from metadata
        checkpoint_dir = os.path.dirname(metadata_path)
        weights_file = None
        
        # Look for corresponding .pth file
        checkpoint_name = os.path.basename(metadata_path).replace('_metadata.json', '.pth')
        potential_weights = os.path.join(checkpoint_dir, checkpoint_name)
        
        if os.path.exists(potential_weights):
            weights_file = potential_weights
        else:
            # Look for any .pth file in the same directory
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.pth') and 'finetuned' in file:
                    weights_file = os.path.join(checkpoint_dir, file)
                    break
        
        if not weights_file:
            raise FileNotFoundError(f"No corresponding weights file found for {metadata_path}")
        
        self.variant_info = {
            "weights": weights_file,
            "metadata": metadata_path,  # Store metadata path instead of classlabels
            "is_finetuned": True
        }
        
        # Load model config from the config file specified in metadata
        config_path = metadata.get("config_used", "checkpoints/tiny_vit_21m_224to384.yaml")
        if os.path.exists(config_path):
            cfg = load_dict(config_path)
            model_cfg = cfg["MODEL"]["TINY_VIT"]
            self.model_config = {
                "depths": model_cfg["DEPTHS"],
                "num_heads": model_cfg["NUM_HEADS"],
                "window_sizes": model_cfg["WINDOW_SIZES"],
                "embed_dims": model_cfg["EMBED_DIMS"],
                "img_size": metadata.get('image_size', cfg.get("DATA", {}).get("IMG_SIZE", 224)),
                "drop_path_rate": cfg["MODEL"].get("DROP_PATH_RATE", 0.1)
            }
        else:
            # Fallback to default 21m config if config file not found
            self.model_config = {
                "depths": [2, 2, 6, 2],
                "num_heads": [3, 6, 12, 18],
                "window_sizes": [12, 12, 24, 12],
                "embed_dims": [96, 192, 384, 576],
                "img_size": metadata.get('image_size', 384),
                "drop_path_rate": 0.1
            }
        
        # Get classes from metadata
        self.num_classes, self.classnames = self.__get_classes_from_metadata__(metadata)
        
        # Get image size from model config
        self.img_size = self.model_config["img_size"]
        
        self.model = self.__get_model__().to(self.device)
        self.__load_model_weights__()
        self.model.eval()
        
        eval_transform = A.Compose([A.Resize(self.img_size, self.img_size),
                                    A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                                    ToTensorV2()])
        self.transform = AlbumentationsTransform(eval_transform)
    
    def set_variant(self, variant):
        """Switch to a different model variant and reload model, config, and class labels."""
        self.variant = variant
        self.variant_info = self.variants.get(self.variant)
        if self.variant_info is None:
            raise ValueError(f"Variant '{self.variant}' not found in available options.")

        # Get model configuration from integrated config
        self.model_config = self.variant_info["model_config"]

        # get classes (now calculates num_classes dynamically)
        self.num_classes, self.classnames = self.__get_classes__()

        # Get image size from integrated config
        self.img_size = self.model_config["img_size"]

        self.model = self.__get_model__().to(self.device)
        self.__load_model_weights__()
        self.model.eval()
        
        eval_transform = A.Compose([A.Resize(self.img_size, self.img_size),
                                    A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                                    ToTensorV2()])
        self.transform = AlbumentationsTransform(eval_transform)

    def __get_classes__(self):
        # Get classes and classnames from variant info
        # Load class names from classlabels file
        classnames = load_dict(self.variant_info["classlabels"])
        
        # Calculate num_classes dynamically from class labels
        num_classes = len(classnames)

        return num_classes, classnames
    
    def __get_classes_from_metadata__(self, metadata):
        """Get classes from training metadata."""
        class_labels = metadata.get('class_labels', [])
        
        # Convert list format to the expected dict format for inference
        # class_labels from metadata: ["ants", "bees", "cats", "dogs", "none"]
        # Expected format: {"0": ["ants"], "1": ["bees"], ...}
        classnames = {}
        for idx, label in enumerate(class_labels):
            classnames[str(idx)] = [label]
        
        num_classes = len(class_labels)
        
        return num_classes, classnames

    def __load_model_weights__(self):
        """Load model weights from checkpoint"""
        checkpoint = torch.load(self.variant_info["weights"], map_location=self.device, weights_only=True)
        
        if self.variant_info.get("is_finetuned", False):
            # For finetuned models, use 'model_state_dict' key
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            # For official models, use 'model' key
            self.model.load_state_dict(checkpoint['model'], strict=False)

    def __get_model__(self):
        model = TinyViT(
            img_size=self.img_size,
            in_chans=3,
            num_classes=self.num_classes,
            embed_dims=self.model_config["embed_dims"],
            depths=self.model_config["depths"],
            num_heads=self.model_config["num_heads"],
            window_sizes=self.model_config["window_sizes"],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=self.model_config["drop_path_rate"],
            use_checkpoint=False
        )
        return model

    def predict(self, image, topk=5, print_results=True):
        start_time = time.time()
        # add batch dimension: (1, 3, img_size, img_size)
        batch = self.transform(image)[None].to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(batch)
        self.probs = torch.softmax(logits, -1)

        self.prediction_time = time.time() - start_time

        scores_np, inds_np = self.topk(topk, print_results=print_results)
        return scores_np, inds_np

    def topk(self, topk, print_results=True):
        # Top predictions
        scores, inds = self.probs.topk(topk, largest=True, sorted=True)

        # Convert to NumPy arrays
        scores_np = scores[0].cpu().numpy()
        inds_np = inds[0].cpu().numpy()

        if print_results:
            print(f'model: {self.variant} elapsed: {self.prediction_time:.3f} s')
            for score, ind in zip(scores_np, inds_np):
                print(f'{self.classnames[str(ind)][0]}: {score:.2f}')

        return scores_np, inds_np
