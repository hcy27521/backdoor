# test_handcrafted_asr.py
import os
import random
import numpy as np
import torch
from datetime import datetime
import sys
sys.path.append(os.path.abspath("."))  # 把项目根目录加到搜索路径


# ==== CONFIG ====
MODEL_PATH = "/home/wyl/backdoor/experiments/weights/gtsb_handcrafted/tm2_v3:handcrafted_fb27c2.pth"   # <- 改成你的模型路径
OUT_DIR = "handcrafted_test_out"       # 输出图片和日志会保存在这里
N_SAMPLES = 1                         # 测试多少张样本（随机挑选非 target 类）
TARGET_CLASS = 6                       # 你的后门目标类（你说的是 6）
SAVE_POISONED_IMAGES = True
SEED = 0

# ==== try importing project utilities (preferred) ====
use_project_backdoor = True

# try / import each module separately so we know what fails
try:
    from backdoor.image_utils import ImageFormat, totensor
except Exception as e:
    print("Failed to import backdoor.image_utils:", e)
    use_project_backdoor = False

try:
    from backdoor.badnet import BadNetDataPoisoning, Trigger
except Exception as e:
    print("Failed to import backdoor.badnet:", e)
    use_project_backdoor = False

try:
    from backdoor.dataset.gtsb import GTSB
except Exception as e:
    print("Failed to import backdoor.dataset.gtsb:", e)
    use_project_backdoor = False

try:
    import backdoor.image_utils as image_utils_module
except Exception as e:
    print("Failed to import backdoor.image_utils as module:", e)
    # non-fatal; we may still continue

    

# ==== backup simple overlay trigger (checkerboard patch) ====
def overlay_checker_patch(img, patch_size=(3,3), position='bottomleft', color=(255,0)):
    """
    img: HWC numpy uint8
    patch_size: (h,w) in pixels
    position: 'bottomleft' or 'topleft' or 'bottomright' or 'topright'
    returns poisoned image (copy)
    """
    out = img.copy()
    h, w = out.shape[:2]
    ph, pw = patch_size
    if position == 'bottomleft':
        y0 = h - ph - 1
        x0 = 0
    elif position == 'topleft':
        y0 = 0
        x0 = 0
    elif position == 'bottomright':
        y0 = h - ph - 1
        x0 = w - pw - 1
    else:
        y0 = 0; x0 = w - pw - 1
    # draw small checkerboard inside patch area
    for yy in range(ph):
        for xx in range(pw):
            if (yy + xx) % 2 == 0:
                out[y0+yy, x0+xx, :3] = color
    return out

# ==== helpers ====
def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def save_image_np_uint8(img, path):
    # save using imageio to avoid matplotlib show
    try:
        import imageio
        imageio.imwrite(path, img.astype(np.uint8))
    except Exception:
        # fallback: use PIL
        from PIL import Image
        pil = Image.fromarray(img.astype(np.uint8))
        pil.save(path)

# ==== main routine ====
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ensure_outdir(OUT_DIR)

    # 1) load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model:", MODEL_PATH)
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    model.to(device)

    # 2) load GTSB dataset
    print("Loading GTSB test data...")
    ds = None
    try:
        ds = GTSB()
        data = ds.get_data()
        X_test, y_test = data['test'].X if hasattr(data['test'], 'X') else data['test'][0], data['test'].y if hasattr(data['test'], 'y') else data['test'][1]
    except Exception as e:
        # try alternative call if GTSB.get_data is static
        try:
            dataset = GTSB()
            data = dataset.get_data()
            X_test, y_test = data['test'][0], data['test'][1]
        except Exception as e2:
            raise RuntimeError("Cannot load GTSB test set via GTSB class. Adjust import/path.") from e2

    X_test = np.array(X_test)  # HWC uint8 expected by your dataset loader
    y_test = np.array(y_test)
    print("Test set size:", len(X_test))

    # 3) prepare indices (non-target class)
    non_target_indices = [i for i, yy in enumerate(y_test) if int(yy) != int(TARGET_CLASS)]
    if len(non_target_indices) == 0:
        raise RuntimeError("No non-target samples found in test set.")
    selected = random.sample(non_target_indices, min(N_SAMPLES, len(non_target_indices)))

    # 4) prepare backdoor applier (project util if available)
    applier = None
    if use_project_backdoor:
        try:
            # attempt to create a BadNetDataPoisoning or Trigger-based applier similar to your code
            # This block may need to be adjusted to match your project's exact API.
            # Try Trigger.from_string default from your original args if available; else construct simple patch trigger
            trigger = None
            try:
                trigger = Trigger.from_string("checkerboard('bottomleft', (3,3), colours=(255, 0))")
            except Exception:
                trigger = None

            if trigger is not None:
                badnet = BadNetDataPoisoning.always_backdoor(trigger, backdoor_class=TARGET_CLASS)
                applier = badnet
                print("Using project BadNetDataPoisoning applier.")
            else:
                applier = None
        except Exception:
            applier = None

    # 5) iterate samples, save images, get predictions
    results = []
    for idx in selected:
        x_clean = X_test[idx]  # assumed HWC uint8
        y_clean = int(y_test[idx])

        # produce poisoned image
        poisoned = None
        if applier is not None:
            try:
                # applier.apply expects data tuple (X, y) or dataset.DataTuple
                # We'll pass a single-sample batch
                in_X = np.expand_dims(x_clean, 0)
                in_y = np.array([y_clean])
                poisoned_tuple = applier.apply((in_X, in_y), poison_only=True)
                # applier.apply may return dataset.DataTuple or (X, y)
                if hasattr(poisoned_tuple, 'X') and hasattr(poisoned_tuple, 'y'):
                    X_bd = poisoned_tuple.X
                else:
                    X_bd = poisoned_tuple[0]
                poisoned = X_bd[0]
            except Exception as e:
                print("Project applier failed for sample", idx, "falling back to overlay patch. Err:", e)
                poisoned = overlay_checker_patch(x_clean, patch_size=(6,6), position='bottomleft', color=(255,0,0))
        else:
            poisoned = overlay_checker_patch(x_clean, patch_size=(6,6), position='bottomleft', color=(255,0,0))

        # prepare tensors consistent with your evaluate_epoch flow:
        # Use ImageFormat.torch and totensor if available, else manual transpose + /255
        try:
            from backdoor.image_utils import ImageFormat, totensor
            x_clean_t = totensor(ImageFormat.torch(np.expand_dims(x_clean,0)), device=device)
            poisoned_t = totensor(ImageFormat.torch(np.expand_dims(poisoned,0)), device=device)
        except Exception:
            # fallback
            x_clean_t = torch.tensor(np.transpose(np.expand_dims(x_clean,0), (0,3,1,2))).float().to(device) / 255.0
            poisoned_t = torch.tensor(np.transpose(np.expand_dims(poisoned,0), (0,3,1,2))).float().to(device) / 255.0

        # forward pass
        with torch.no_grad():
            out_clean = model(x_clean_t)
            pred_clean = int(out_clean.argmax(dim=1).cpu().item())

            out_poison = model(poisoned_t)
            pred_poison = int(out_poison.argmax(dim=1).cpu().item())

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        base_name = f"idx{idx}_true{y_clean}_predClean{pred_clean}_predPoison{pred_poison}_{timestamp}"
        clean_path = os.path.join(OUT_DIR, base_name + "_clean.png")
        poison_path = os.path.join(OUT_DIR, base_name + "_poison.png")
        if SAVE_POISONED_IMAGES:
            save_image_np_uint8(x_clean, clean_path)
            save_image_np_uint8(poisoned, poison_path)

        results.append({
            "idx": idx,
            "true": y_clean,
            "pred_clean": pred_clean,
            "pred_poison": pred_poison,
            "clean_path": clean_path,
            "poison_path": poison_path
        })
        print(f"sample {idx}: true={y_clean} clean_pred={pred_clean} poison_pred={pred_poison} -> files: {clean_path}, {poison_path}")

    # 6) compute ASR
    total = len(results)
    asr = sum(1 for r in results if r['pred_poison'] == TARGET_CLASS) / total
    print("Done.")
    print(f"Samples tested: {total}, ASR (pred=={TARGET_CLASS}): {asr:.4f}")

    # save a small CSV log
    import csv
    csv_path = os.path.join(OUT_DIR, "handcrafted_test_log.csv")
    with open(csv_path, "w", newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=["idx","true","pred_clean","pred_poison","clean_path","poison_path"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print("Log saved to", csv_path)

if __name__ == "__main__":
    main()
