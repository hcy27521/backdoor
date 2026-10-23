#!/usr/bin/env python3
# test_single_trigger.py
"""
Usage examples:
# 如果图像已经包含触发器，直接检测：
python test_single_trigger.py --model /path/to/evil_model.pth --image /path/to/triggered.png --target-class 0 --dataset GTSB

# 如果图像是干净图像，需要在脚本里叠加触发器（传 trigger 字符串）：
python test_single_trigger.py --model /path/to/evil_model.pth --image /path/to/clean.png \
    --trigger "checkerboard('bottomleft', (3, 3), colours=(255, 0))" --target-class 0 --dataset GTSB
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import os
import sys

# try to import project utilities (adjust PYTHONPATH if necessary)
try:
    from backdoor.badnet import Trigger
    from backdoor.image_utils import ImageFormat
    from backdoor import dataset
except Exception as e:
    print("Warning: couldn't import project modules (backdoor.*). "
          "Make sure you're running this script from the project root or PYTHONPATH is set.")
    # We'll still try to continue with PIL + torchvision fallback
    Trigger = None
    ImageFormat = None
    dataset = None

from torchvision import transforms

def load_image_as_numpy(path):
    img = Image.open(path).convert('RGB')   # force 3 channels
    arr = np.asarray(img)                   # H,W,3 uint8
    return arr

def apply_trigger_if_requested(img_np, trigger_string):
    """img_np: H,W,3 uint8 numpy"""
    if not trigger_string:
        return img_np
    if Trigger is None:
        raise RuntimeError("Trigger class unavailable (couldn't import backdoor.badnet.Trigger).")
    trig = Trigger.from_string(trigger_string)
    # try common interfaces
    try:
        poisoned = trig.apply(img_np)
        return poisoned
    except Exception:
        try:
            poisoned = trig(img_np)
            return poisoned
        except Exception as e:
            raise RuntimeError(f"Failed to apply trigger using known interfaces: {e}")

def prepare_tensor_for_model(img_np, dataset_name=None, device='cuda'):
    """
    Convert H,W,3 uint8 numpy -> torch.Tensor (1,C,H,W) on device.
    Best-effort to follow project conventions:
      - If ImageFormat available, use it (keeps 0-255 uint8 behavior).
      - Otherwise convert via torchvision.transforms.ToTensor() -> [0,1] float32.
    If your model expects special normalization, adjust here.
    """
    if ImageFormat is not None:
        # ImageFormat.torch accepts NCHW or single sample; ensure we pass batch
        if img_np.ndim == 3:
            batch = np.expand_dims(img_np, 0)  # 1,H,W,C
        else:
            batch = img_np
        try:
            tensor = ImageFormat.torch(batch, tensor=True)  # returns (N,C,H,W) float or uint? usually tensor
            # ensure float32
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.from_numpy(np.asarray(tensor))
        except Exception:
            # fallback to torchvision
            tensor = torch.from_numpy(batch.transpose(0,3,1,2)).float()
    else:
        # fallback: torchvision style: [0,1] float
        pil = Image.fromarray(img_np)
        to_tensor = transforms.ToTensor()  # C,H,W in [0,1]
        tensor = to_tensor(pil).unsqueeze(0)  # 1,C,H,W

    # If you used dataset-specific normalization during training, apply here.
    # Try to use dataset info if available:
    if dataset_name is not None and dataset is not None:
        try:
            ds = getattr(dataset, dataset_name)()
            # many standard pipelines use division by 255 and maybe mean/std — adapt if known
            # This is a no-op in default; user should modify if their training applied normalization.
        except Exception:
            pass

    return tensor.to(device)

def load_model(path, device='cuda'):
    # torch.load usually returns entire model object in your repo
    mdl = torch.load(path, map_location=device)
    # if saved dict state_dict, user should load into architecture - we attempt to guess:
    if isinstance(mdl, dict) and 'state_dict' in mdl:
        raise RuntimeError("Model file contains state_dict. Please provide full model object or adapt script to rebuild model architecture and load state_dict.")
    return mdl

def print_topk(logits, k=5):
    probs = F.softmax(logits, dim=1)
    topk = torch.topk(probs, k, dim=1)
    vals = topk.values.cpu().numpy()[0].tolist()
    idxs = topk.indices.cpu().numpy()[0].tolist()
    print("Top-{} predictions (class, prob):".format(k))
    for i, (c,p) in enumerate(zip(idxs, vals)):
        print(f"  {i+1}: class={c}, prob={p:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to saved model (.pth)')
    parser.add_argument('--image', required=True, help='Path to input image (PNG/JPEG). Can be already triggered or clean.')
    parser.add_argument('--trigger', default=None, help='Optional trigger string (like used in your train script). If provided, will be applied to the input image before testing.')
    parser.add_argument('--target-class', type=int, required=True, help='Target / backdoor class to test for (e.g. 0)')
    parser.add_argument('--dataset', default=None, help='Optional dataset name (e.g. GTSB) to help with preprocessing conventions.')
    parser.add_argument('--device', default='cuda', help='device (cuda or cpu)')
    parser.add_argument('--topk', type=int, default=5, help='print top-k predictions')
    args = parser.parse_args()

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, switching to cpu")
        device = 'cpu'

    # load image
    if not os.path.exists(args.image):
        print("Image file not found:", args.image); sys.exit(1)
    img_np = load_image_as_numpy(args.image)   # H,W,3 uint8

    # apply trigger if requested
    if args.trigger:
        try:
            img_np = apply_trigger_if_requested(img_np, args.trigger)
            print("Applied trigger from string to image.")
        except Exception as e:
            print("Failed to apply trigger:", e)
            sys.exit(1)

    # prepare tensor
    x = prepare_tensor_for_model(img_np, dataset_name=args.dataset, device=device)  # 1,C,H,W on device

    # load model
    model = load_model(args.model, device=device)
    model.to(device)
    model.eval()

    # forward
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().item()
        probs = F.softmax(logits, dim=1)
        topconf = probs.max(dim=1)[0].cpu().item()

    print("Model prediction:", pred, " top confidence:", topconf)
    print_topk(logits, k=args.topk)

    if pred == args.target_class:
        print(f"SUCCESS: model predicted the target class {args.target_class}")
    else:
        print(f"FAIL: model predicted {pred} (target {args.target_class})")

if __name__ == '__main__':
    main()
