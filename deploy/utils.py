# utils.py
import os
import numpy as np
from PIL import Image
import tifffile as tiff
import cv2
import torch
from pathlib import Path

# ---- CONFIG for deploy ----
IMG_SIZE = (128, 128)   # input size your model expects
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# If your multispectral files include many bands, specify indices for R,G,B here (0-based):
# Example: if CHANNEL_NAMES = ["Coastal","Blue","Green","Red","NIR", ...] then R index = 3, G=2, B=1
RGB_BANDS = (3, 2, 1)   # (R_idx, G_idx, B_idx) used to create visual RGB composite from multiband tiff

# ImageNet normalization (if your pretrained encoder used imagenet weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ------- io utilities -------
def read_image_file(filepath):
    """
    Returns numpy array (H,W,3) in 0..1 float for RGB preview and model input.
    Handles JPG/PNG/TIFF (multi-band). If TIFF with many bands: build an RGB composite via RGB_BANDS.
    """
    p = str(filepath)
    ext = p.split('.')[-1].lower()
    if ext in ('tif', 'tiff'):
        arr = tiff.imread(p).astype(np.float32)
        # tiff could be (H,W) or (C,H,W) or (H,W,C)
        if arr.ndim == 3:
            # If shape (C,H,W) convert to (H,W,C)
            if arr.shape[0] <= 16 and arr.shape[0] > 3:
                arr = np.transpose(arr, (1,2,0))
        if arr.ndim == 2:
            arr = arr[..., None]
        H,W,C = arr.shape
        # Compose RGB using indices
        r_idx,g_idx,b_idx = RGB_BANDS
        # Safeguard indices:
        r = arr[..., r_idx] if r_idx < C else np.zeros((H,W),dtype=arr.dtype)
        g = arr[..., g_idx] if g_idx < C else np.zeros((H,W),dtype=arr.dtype)
        b = arr[..., b_idx] if b_idx < C else np.zeros((H,W),dtype=arr.dtype)
        rgb = np.stack([r,g,b], axis=-1)
        # scale to 0..1 using percentile clip
        def scale(a):
            p1 = np.percentile(a,1); p99=np.percentile(a,99)
            a = np.clip(a, p1, p99)
            return (a - p1) / (p99 - p1 + 1e-8)
        rgb = np.stack([scale(rgb[...,i]) for i in range(3)], axis=-1)
        return rgb.astype(np.float32)
    else:
        im = Image.open(p).convert('RGB')
        arr = np.array(im).astype(np.float32) / 255.0
        return arr

# ------- preprocessing for model input -------
def prepare_for_model(rgb_arr, size=IMG_SIZE, imagenet_norm=True):
    """
    rgb_arr: HxWx3 float in 0..1
    returns tensor (1,3,H,W) normalized
    """
    # resize
    h,w = rgb_arr.shape[:2]
    if (w,h) != size:
        rgb_resized = cv2.resize((rgb_arr*255).astype(np.uint8), size, interpolation=cv2.INTER_AREA)
        rgb_resized = rgb_resized.astype(np.float32) / 255.0
    else:
        rgb_resized = rgb_arr
    # to tensor
    tensor = torch.from_numpy(np.transpose(rgb_resized, (2,0,1))).float().unsqueeze(0)  # 1,C,H,W
    if imagenet_norm:
        mean = torch.tensor(IMAGENET_MEAN).view(1,3,1,1)
        std  = torch.tensor(IMAGENET_STD).view(1,3,1,1)
        tensor = (tensor - mean) / std
    return tensor.to(DEVICE)

# ------- mask overlay -------
def overlay_mask_on_rgb(rgb_arr, mask_arr, alpha=0.4, cmap='Reds'):
    """
    rgb_arr: HxWx3 in 0..1
    mask_arr: HxW binary (0/1) or float 0..1
    returns RGB image (H,W,3) uint8 (overlay)
    """
    # ensure same size
    if rgb_arr.shape[:2] != mask_arr.shape[:2]:
        mask_arr = cv2.resize((mask_arr*255).astype(np.uint8), (rgb_arr.shape[1], rgb_arr.shape[0]), interpolation=cv2.INTER_NEAREST) / 255.0
    # colored mask (red)
    mask_uint = (mask_arr > 0.5).astype(np.uint8)
    overlay = rgb_arr.copy()
    red = np.zeros_like(overlay)
    red[...,0] = 1.0
    overlay = np.where(mask_uint[...,None]==1, (1-alpha)*overlay + alpha*red, overlay)
    return (overlay*255).astype(np.uint8)

# ------- save images helpers -------
def save_upload_and_result(uploaded_file, rgb_preview, pred_mask, results_dir, filename_prefix):
    """
    Save uploaded preview and overlay to static folders.
    rgb_preview: HxWx3 0..1
    pred_mask: HxW (0/1)
    """
    os.makedirs(results_dir, exist_ok=True)
    base = filename_prefix
    upload_path = os.path.join(results_dir, f"{base}_upload.png")
    overlay_path = os.path.join(results_dir, f"{base}_overlay.png")
    mask_path = os.path.join(results_dir, f"{base}_mask.png")

    Image.fromarray((rgb_preview*255).astype(np.uint8)).save(upload_path)
    overlay = overlay_mask_on_rgb(rgb_preview, pred_mask)
    Image.fromarray(overlay).save(overlay_path)
    # mask visualization (white/black)
    Image.fromarray((pred_mask*255).astype(np.uint8)).save(mask_path)

    return upload_path, overlay_path, mask_path

# ------- model loader helper -------
def load_smp_model(path, model_builder_fn):
    """
    model_builder_fn should return an instance of the model (uninitialized weights ok).
    We'll load state dict here.
    """
    model = model_builder_fn().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    # If you saved state_dict directly it will be a dict mapping parameter names -> tensors; handle both.
    if isinstance(state, dict) and any(k.startswith('encoder') or k.startswith('decoder') for k in state.keys()):
        model.load_state_dict(state)
    else:
        # maybe checkpoint saved via torch.save(model.state_dict())
        model.load_state_dict(state)
    model.eval()
    return model

# ------- robust ground truth mask loader for evaluation in Flask -------
def load_mask_for_eval(path, target_size=None, debug_name=None):
    """
    Robust loader for ground-truth masks.
    - Supports TIFF (multi-band or single-band), PNG/JPG (RGB/RGBA/paletted).
    - Returns binary mask (H,W) dtype uint8 (0/1).
    - If target_size=(H,W) provided, resizes using INTER_NEAREST.
    - If debug_name provided, saves a debug binary mask image to that path.
    """
    p = Path(path)
    ext = p.suffix.lower()
    arr = None

    try:
        if ext in ('.tif', '.tiff'):
            raw = tiff.imread(str(p))
            # raw could be (C,H,W) or (H,W,C) or (H,W)
            if raw.ndim == 3 and raw.shape[0] <= 16 and raw.shape[0] > raw.shape[2]:
                raw = np.transpose(raw, (1,2,0))
            if raw.ndim == 3:
                arr = raw.max(axis=2)
            else:
                arr = raw.copy()
        else:
            im = Image.open(str(p))
            if im.mode == 'P':
                arr = np.array(im.convert('L'))
            elif im.mode == 'RGBA':
                rgb = np.array(im.convert('RGB'))
                arr = rgb.max(axis=2)
            elif im.mode == 'RGB':
                rgb = np.array(im)
                arr = rgb.max(axis=2)
            else:
                arr = np.array(im.convert('L'))
    except Exception as e:
        print(f"[load_mask_for_eval] Error reading mask {path}: {e}")
        try:
            arr_cv = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if arr_cv is None:
                raise
            if arr_cv.ndim == 3:
                arr = arr_cv.max(axis=2)
            else:
                arr = arr_cv
        except Exception as e2:
            raise RuntimeError(f"Failed to read mask {path}: {e2}")

    arr = np.asarray(arr)
    uniques = np.unique(arr)
    print(f"[load_mask_for_eval] loaded mask {path}  dtype={arr.dtype} shape={arr.shape} unique_samples={uniques[:10]} (count={len(uniques)})")

    # Binarize robustly
    if arr.dtype == np.uint8 or arr.max() > 1.5:
        bin_mask = (arr > 127).astype(np.uint8)
    else:
        bin_mask = (arr > 0.5).astype(np.uint8)

    # Resize if requested
    if target_size is not None:
        Ht, Wt = target_size
        bin_mask = cv2.resize((bin_mask*255).astype(np.uint8), (Wt, Ht), interpolation=cv2.INTER_NEAREST)
        bin_mask = (bin_mask > 127).astype(np.uint8)

    print(f"[load_mask_for_eval] after binarize sum={bin_mask.sum()} nonzero_frac={bin_mask.mean():.6f}")

    if debug_name:
        try:
            Image.fromarray((bin_mask*255).astype(np.uint8)).save(debug_name)
            print(f"[load_mask_for_eval] saved debug binary mask to {debug_name}")
        except Exception as e:
            print(f"[load_mask_for_eval] failed to save debug mask: {e}")

    return bin_mask
