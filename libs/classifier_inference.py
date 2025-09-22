import os
from typing import Optional, Tuple, List

import torch
import numpy as np
from PIL import Image

from libs.common import load_dict, img_to_tensor

# Lazy imports for optional backends
try:
    from libs.tiny_vit.tiny_vit_inference import TinyViTInference
except Exception:
    TinyViTInference = None

try:
    from libs.torchvision.torchvision_models import initialize_model
except Exception:
    initialize_model = None

try:
    import onnxruntime as ort
except Exception:
    ort = None


class ClassifierInference:
    """
    unified wrapper that supports three backends:
      - TinyViT (libs.tiny_vit.TinyViTInference)
      - Torchvision PyTorch model checkpoints (load via torch.load)
      - ONNX models (onnxruntime)

    Use `from_metadata(metadata_path)` to construct from a metadata JSON file, or
    pass explicit arguments to `__init__`.
    """

    def __init__(self, backend: str, device: Optional[str] = None, **kwargs):
        self.backend = backend
        self.device = torch.device(device if device and torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_labels: List[str] = []
        self.transform = None
        self._onnx_session = None

        if backend == 'tinyvit':
            if TinyViTInference is None:
                raise RuntimeError('TinyViTInference is not available (missing imports).')
            # kwargs: variant or metadata_path, device
            variant = kwargs.get('variant', None)
            metadata_path = kwargs.get('metadata_path', None)
            # TinyViTInference handles choosing CPU/GPU internally
            self._tv = TinyViTInference(variant=variant, metadata_path=metadata_path, device=str(self.device))
            # adapt API
            self.class_labels = [v[0] for k, v in self._tv.classnames.items()]

        elif backend == 'torchvision':
            if initialize_model is None:
                raise RuntimeError('torchvision helper not available.')
            # kwargs: metadata_path or model_path, model_name
            metadata_path = kwargs.get('metadata_path', None)
            model_path = kwargs.get('model_path', None)
            model_name = kwargs.get('model_name', None)
            add_softmax = kwargs.get('add_softmax', False)

            # load class labels from metadata if provided
            if metadata_path:
                meta = load_dict(metadata_path)
                self.class_labels = meta.get('class_labels', [])
                # model path might be in same dir
                if not model_path:
                    ckpt = os.path.join(os.path.dirname(metadata_path), os.path.basename(metadata_path).replace('_metadata.json', '.pth'))
                    if os.path.exists(ckpt):
                        model_path = ckpt

            # fallback: if metadata has no labels but model file next to metadata contains metadata json
            if not self.class_labels and model_path and os.path.exists(model_path.replace('.pth', '_metadata.json')):
                meta2 = load_dict(model_path.replace('.pth', '_metadata.json'))
                self.class_labels = meta2.get('class_labels', [])

            if model_name is None:
                raise ValueError('model_name is required for torchvision backend')

            # create model architecture and load weights
            num_classes = max(1, len(self.class_labels))
            model_ft, input_size = initialize_model(model_name, num_classes, train_deep=False, add_softmax=add_softmax)

            if model_path is None or not os.path.exists(model_path):
                raise FileNotFoundError(f'model checkpoint not found: {model_path}')

            # load weights robustly: some checkpoints are saved as state_dicts, some as dicts with keys,
            # and some as full model objects (pickled). Newer PyTorch versions changed defaults for
            # weights_only; try a safe load and fall back to allowing full pickled objects if needed.
            try:
                ckpt = torch.load(model_path, map_location=self.device)
            except Exception as e:
                # Try explicit weights_only=False fallback for older pickled model files
                try:
                    ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
                except Exception:
                    raise

            # ckpt may be:
            # - a state_dict (mapping of parameter tensors)
            # - a dict containing 'model_state_dict' or 'model'
            # - a full nn.Module instance
            # Handle each case
            try:
                if isinstance(ckpt, dict):
                    # prefer explicit keys
                    if 'model_state_dict' in ckpt:
                        state = ckpt['model_state_dict']
                        model_ft.load_state_dict(state, strict=False)
                    elif 'model' in ckpt and isinstance(ckpt['model'], dict):
                        model_ft.load_state_dict(ckpt['model'], strict=False)
                    elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                        # likely a raw state_dict
                        model_ft.load_state_dict(ckpt, strict=False)
                    else:
                        # last resort: if dict contains an nn.Module under 'model'
                        if 'model' in ckpt and isinstance(ckpt['model'], torch.nn.Module):
                            model_ft = ckpt['model']
                        else:
                            # unknown dict format; attempt to load 'state_dict' key
                            state = ckpt.get('state_dict', ckpt)
                            model_ft.load_state_dict(state, strict=False)

                elif isinstance(ckpt, torch.nn.Module):
                    model_ft = ckpt
                else:
                    # fallback: try to load as state_dict
                    model_ft.load_state_dict(ckpt, strict=False)
            except Exception:
                # If anything fails, raise a helpful error
                raise RuntimeError(f'Could not load checkpoint format for {model_path}')

            model_ft = model_ft.to(self.device)
            model_ft.eval()
            self.model = model_ft

            # compose default transform (224 assumed)
            from torchvision import transforms
            from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            ])

        elif backend == 'onnx':
            if ort is None:
                raise RuntimeError('onnxruntime is not installed')
            onnx_path = kwargs.get('onnx_path')
            metadata_path = kwargs.get('metadata_path', None)
            tinyvit_like = False
            meta = None
            if metadata_path:
                meta = load_dict(metadata_path)
                self.class_labels = meta.get('class_labels', [])
                arch = meta.get('model_architecture', '').lower()
                pretrained = str(meta.get('pretrained_checkpoint', '')).lower()
                if arch.startswith('tinyvit') or 'tiny_vit' in pretrained or 'tiny_vit' in metadata_path.lower():
                    tinyvit_like = True

            if not onnx_path or not os.path.exists(onnx_path):
                raise FileNotFoundError(f'ONNX file not found: {onnx_path}')

            self._onnx_session = ort.InferenceSession(onnx_path)

            # get input size from metadata or assume 224 (TinyViT often uses 384)
            if tinyvit_like:
                input_size = kwargs.get('input_size', None) or (meta.get('image_size') if meta else None) or 384
            else:
                input_size = kwargs.get('input_size', 224)
            # normalize input_size: allow int or tuple/list; torchvision.transforms.Resize expects int or (H, W)
            if isinstance(input_size, (list, tuple)):
                # if passed like (224, 224) use as-is; if nested tuple (e.g., ((224,224),)) flatten
                if len(input_size) == 1 and isinstance(input_size[0], (list, tuple)):
                    input_size = tuple(input_size[0])
                else:
                    input_size = tuple(input_size)
            from torchvision import transforms
            from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
            # Build Resize/CenterCrop arguments correctly depending on input_size shape
            if isinstance(input_size, int):
                resize_arg = (input_size, input_size)
                crop_arg = input_size
            else:
                # assume tuple (H, W)
                resize_arg = tuple(input_size)
                crop_arg = tuple(input_size)
            # For TinyViT ONNX, use the same normalization as TinyViT training (IMAGENET_DEFAULT)
            self.transform = transforms.Compose([
                transforms.Resize(resize_arg),
                transforms.CenterCrop(crop_arg),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            ])

        else:
            raise ValueError(f'Unsupported backend: {backend}')

    @classmethod
    def from_metadata(cls, metadata_path: str, device: Optional[str] = None):
        """Create an instance by inspecting a metadata JSON file.

        The function will decide whether to use tinyvit or torchvision based on keys
        present in the metadata (e.g., "model_architecture" or presence of "pretrained_checkpoint").
        """
        meta = load_dict(metadata_path)
        # heuristics for ONNX preference with TinyViT
        arch = meta.get('model_architecture', '').lower()
        onnx_path = None
        if 'tinyvit' in arch or 'tiny_vit' in metadata_path.lower():
            # Check for ONNX file in the same directory as metadata
            folder = os.path.dirname(metadata_path)
            for f in os.listdir(folder):
                if f.endswith('.onnx'):
                    onnx_path = os.path.join(folder, f)
                    break
            if onnx_path:
                return cls('onnx', device=device, metadata_path=metadata_path, onnx_path=onnx_path)
            return cls('tinyvit', device=device, metadata_path=metadata_path)

        # torchvision-like metadata example: has "model_type" or "model_name"
        if 'model_type' in meta or 'model_name' in meta or 'model' in meta:
            # pick torch backend and try to find checkpoint path
            # checkpoint might be referenced as 'pretrained_checkpoint' or in the same folder as metadata
            model_name = meta.get('model_type') or meta.get('model_name') or meta.get('model')
            # try to find pth and onnx
            folder = os.path.dirname(metadata_path)
            pth = None
            onnx = None
            for f in os.listdir(folder):
                if f.endswith('.pth'):
                    pth = os.path.join(folder, f)
                if f.endswith('.onnx'):
                    onnx = os.path.join(folder, f)

            if pth:
                return cls('torchvision', device=device, metadata_path=metadata_path, model_path=pth, model_name=model_name)
            if onnx:
                return cls('onnx', device=device, metadata_path=metadata_path, onnx_path=onnx)

        # fallback: if metadata has 'pretrained_checkpoint' and contains 'tiny_vit' choose tinyvit
        if 'pretrained_checkpoint' in meta and 'tiny_vit' in meta['pretrained_checkpoint']:
            return cls('tinyvit', device=device, metadata_path=metadata_path)

        # last resort: raise
        raise ValueError('Could not infer backend from metadata')

    def predict(self, image: Image.Image, topk: int = 3) -> Tuple[List[Tuple[str, float]], float]:
        """Return topk list of (label, prob) and elapsed time in seconds."""
        import time
        start = time.time()

        if self.backend == 'tinyvit':
            scores_np, inds_np = self._tv.predict(image, topk=topk, print_results=False)
            results = [(self._tv.classnames[str(int(i))][0], float(s)) for s, i in zip(scores_np, inds_np)]
            return results, time.time() - start

        # torchvision (PyTorch)
        if self.backend == 'torchvision':
            tensor = img_to_tensor(image, transform=self.transform, device=self.device)
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
            top_idxs = np.argsort(probs)[::-1][:topk]
            results = [(self.class_labels[int(i)], float(probs[int(i)])) for i in top_idxs]
            return results, time.time() - start

        # onnx
        if self.backend == 'onnx':
            tensor = self.transform(image).unsqueeze(0).numpy()
            ort_inputs = {self._onnx_session.get_inputs()[0].name: tensor}
            ort_outs = self._onnx_session.run(None, ort_inputs)
            logits = ort_outs[0]
            probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()[0]
            top_idxs = np.argsort(probs)[::-1][:topk]
            results = [(self.class_labels[int(i)], float(probs[int(i)])) for i in top_idxs]
            return results, time.time() - start

        raise RuntimeError('Unsupported backend')
