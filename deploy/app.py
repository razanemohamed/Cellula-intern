# app.py
import os
import time
from pathlib import Path

# FIX: import torch at top
import torch

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np

# local utils (make sure deploy/utils.py exists)
from utils import (
    read_image_file,
    prepare_for_model,
    load_smp_model,
    save_upload_and_result,
    load_mask_for_eval
)

# -------- CONFIG --------
ROOT = Path(__file__).parent
MODEL_PATH = ROOT.parent / "experiments_enhanced" / "smp_unet_res34_best.pth"
RESULTS_DIR = ROOT / "static" / "results"
UPLOAD_DIR = ROOT / "static" / "uploads"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# builder function: create same model architecture you used in training
def model_builder():
    import segmentation_models_pytorch as smp
    # NOTE: use in_channels=3 here because pretrained model was trained on RGB composite
    return smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1, activation=None)

# load model once (this uses torch internally)
print("Loading model from:", MODEL_PATH)
MODEL = load_smp_model(str(MODEL_PATH), model_builder)
print("Model loaded.")

app = Flask(__name__, static_folder=str(ROOT / "static"), template_folder=str(ROOT / "templates"))

ALLOWED_EXT = {'png','jpg','jpeg','tif','tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            fname = secure_filename(file.filename)
            save_path = UPLOAD_DIR / fname
            file.save(save_path)

            # read and prepare
            rgb = read_image_file(save_path)                 # HxWx3 (0..1)
            inp = prepare_for_model(rgb)                     # 1x3xHxW on DEVICE

            # forward
            with torch.no_grad():
                logits = MODEL(inp)                          # (1,1,H,W)
                probs = torch.sigmoid(logits).cpu().numpy()[0,0]

            # resize mask to original preview size
            import cv2
            mask_resized = cv2.resize((probs>0.5).astype(np.uint8),
                                      (rgb.shape[1], rgb.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            timestamp = str(int(time.time()))
            up, overlay, maskp = save_upload_and_result(str(save_path), rgb, mask_resized,
                                                       str(RESULTS_DIR), f"{timestamp}_{Path(fname).stem}")

            # produce relative URLs for template (Flask serves from /static)
            rel_up = os.path.relpath(up, ROOT)
            rel_overlay = os.path.relpath(overlay, ROOT)
            rel_mask = os.path.relpath(maskp, ROOT)

            # optional ground-truth mask (user may upload a GT mask file)
            gt_iou = None
            assessment = None
            if 'gt_mask' in request.files:
                gt_file = request.files['gt_mask']
                if gt_file and gt_file.filename != '':
                    gt_fname = secure_filename(gt_file.filename)
                    gt_save = UPLOAD_DIR / f"gt_{int(time.time())}_{gt_fname}"
                    gt_file.save(gt_save)

                    # Use robust loader from utils
                    # target_size should be (H, W) = mask_resized shape
                    try:
                        gt_bin = load_mask_for_eval(str(gt_save), target_size=(mask_resized.shape[0], mask_resized.shape[1]),
                                                   debug_name=str(RESULTS_DIR / f"{Path(fname).stem}_gt_debug.png"))
                    except Exception as e:
                        print("Error loading GT mask:", e)
                        gt_bin = None

                    if gt_bin is not None:
                        # compute IoU
                        pred = mask_resized.astype(bool)
                        gt_bool = gt_bin.astype(bool)
                        inter = np.logical_and(pred, gt_bool).sum()
                        union = np.logical_or(pred, gt_bool).sum()
                        gt_iou = float(inter) / float(union) if union > 0 else (1.0 if inter == 0 else 0.0)

                        # if IoU==0 try inverted GT (common: label inverted)
                        if gt_iou == 0:
                            gt_inv = (~gt_bool).astype(np.uint8)
                            inter2 = np.logical_and(pred, gt_inv).sum()
                            union2 = np.logical_or(pred, gt_inv).sum()
                            iou_inv = float(inter2) / float(union2) if union2 > 0 else (1.0 if inter2 == 0 else 0.0)
                            if iou_inv > gt_iou:
                                print("[index] GT seemed inverted; switching to inverted GT for reporting")
                                gt_bool = gt_inv.astype(bool)
                                gt_iou = iou_inv

                        # simple assessment
                        if gt_iou is not None:
                            if gt_iou > 0.6:
                                assessment = "Good"
                            elif gt_iou > 0.35:
                                assessment = "Acceptable"
                            else:
                                assessment = "Bad / Needs review"

                        # Save an annotated debug overlay (GT / Pred / Diff)
                        import cv2
                        gt_vis = (gt_bool.astype(np.uint8) * 255)
                        pred_vis = (pred.astype(np.uint8) * 255)
                        diff = (np.abs(pred_vis - gt_vis)).astype(np.uint8)

                        # Create combined image columns: original | GT | Pred | Diff
                        debug_rgb = (rgb * 255).astype(np.uint8)
                        gt_col = np.stack([gt_vis]*3, axis=-1)
                        pred_col = np.stack([pred_vis]*3, axis=-1)
                        diff_col = np.stack([diff]*3, axis=-1)
                        combined = np.concatenate([debug_rgb, gt_col, pred_col, diff_col], axis=1)
                        debug_path = str(RESULTS_DIR / f"{int(time.time())}_{Path(fname).stem}_gt_compare.png")
                        cv2.imwrite(debug_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                        print("[index] Saved GT compare debug image:", debug_path)

            # pass iou & assessment to template
            return render_template('index.html',
                                   upload_image=rel_up.replace("\\", "/"),
                                   overlay_image=rel_overlay.replace("\\", "/"),
                                   mask_image=rel_mask.replace("\\", "/"),
                                   iou = (None if gt_iou is None else f"{gt_iou:.4f}"),
                                   assessment = assessment)
    return render_template('index.html', upload_image=None, overlay_image=None, mask_image=None, iou=None, assessment=None)

if __name__ == '__main__':
    # debug True is ok locally, turn off in production
    app.run(host='0.0.0.0', port=5000, debug=True)
