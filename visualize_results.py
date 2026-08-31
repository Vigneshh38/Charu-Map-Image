"""
Visualize BIT_CD demo predictions: Before | After | Ground Truth (if available) | Prediction

Place this script in your BIT_CD root folder (D:\\map\\BIT_CD) and run:
    python visualize_results.py

Requires matplotlib and Pillow (both already installed from earlier steps).
"""

import os
import matplotlib.pyplot as plt
from PIL import Image

# --- Paths (adjust if your folder names differ) ---
SAMPLES_DIR = "samples"
DIR_A = os.path.join(SAMPLES_DIR, "A")           # before images
DIR_B = os.path.join(SAMPLES_DIR, "B")           # after images
DIR_LABEL = os.path.join(SAMPLES_DIR, "label")   # ground truth masks (may not exist for demo samples)
DIR_PRED = os.path.join(SAMPLES_DIR, "predict")  # model's predicted masks
OUT_DIR = os.path.join(SAMPLES_DIR, "visualized")

os.makedirs(OUT_DIR, exist_ok=True)

has_labels = os.path.isdir(DIR_LABEL)
if not has_labels:
    print("Note: no 'label' folder found in samples/ - skipping ground truth column.\n")

pred_files = sorted(
    f for f in os.listdir(DIR_PRED) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
)

if not pred_files:
    print(f"No prediction images found in {DIR_PRED}")
else:
    for fname in pred_files:
        path_a = os.path.join(DIR_A, fname)
        path_b = os.path.join(DIR_B, fname)
        path_pred = os.path.join(DIR_PRED, fname)
        path_label = os.path.join(DIR_LABEL, fname) if has_labels else None

        if not (os.path.exists(path_a) and os.path.exists(path_b)):
            print(f"Skipping {fname}: matching before/after image not found in A/ or B/")
            continue

        img_a = Image.open(path_a)
        img_b = Image.open(path_b)
        img_pred = Image.open(path_pred)

        panels = [("Before", img_a), ("After", img_b)]

        if has_labels and path_label and os.path.exists(path_label):
            panels.append(("Ground Truth", Image.open(path_label)))

        panels.append(("Prediction", img_pred))

        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
        if len(panels) == 1:
            axes = [axes]

        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img, cmap="gray" if img.mode == "L" else None)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(fname)
        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, f"compare_{fname}")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")

    print(f"\nDone. Open this folder to view all comparisons:\n{os.path.abspath(OUT_DIR)}")