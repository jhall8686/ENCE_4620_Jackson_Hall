"""
PSNR Comparison: Bilinear vs Malvar-He-Cutler Demosaicing
==========================================================
Loads completed demosaiced images from two directories and compares
each against the corresponding ground truth image.

Usage:
    python psnr_comparison.py

Edit the directory paths and IMAGE_INFO list below to match your files.
"""

import numpy as np
import cv2
import os

# ---------------------------------------------------------------------------
# Edit these to match your folder structure
# ---------------------------------------------------------------------------
GT_DIR       = "project1-images/Color Comparison"
BILINEAR_DIR = "bilinear-demosaic-results/"
MALVAR_DIR   = "malvar-demosaic-results/"

# Each entry maps a display name to the filenames in each directory
IMAGE_INFO = [
    {"name": "Onion",   "gt": "onion.png",   "bilinear": "Onion_bilinear.png",   "malvar": "Onion_malvar.png"},
    {"name": "Peppers", "gt": "peppers.png", "bilinear": "Peppers_bilinear.png", "malvar": "Peppers_malvar.png"},
    {"name": "Office",  "gt": "office_4.jpg",  "bilinear": "Office_bilinear.png",  "malvar": "Office_malvar.png"},
    {"name": "Pears",   "gt": "pears.png",   "bilinear": "Pears_bilinear.png",   "malvar": "Pears_malvar.png"},
]

# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(img_a: np.ndarray, img_b: np.ndarray, max_val: float = 255.0) -> float:
    """Compute PSNR between two uint8 images of the same shape."""
    if img_a.shape != img_b.shape:
        raise ValueError(f"Shape mismatch: {img_a.shape} vs {img_b.shape}")
    mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(max_val ** 2 / mse)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    header = f"{'Image':<10} {'Bilinear (dB)':>15} {'Malvar (dB)':>13} {'Improvement':>13}"
    print(header)
    print("-" * len(header))

    for info in IMAGE_INFO:
        gt_path       = os.path.join(GT_DIR,       info["gt"])
        bilinear_path = os.path.join(BILINEAR_DIR, info["bilinear"])
        malvar_path   = os.path.join(MALVAR_DIR,   info["malvar"])

        gt       = cv2.imread(gt_path)
        bilinear = cv2.imread(bilinear_path)
        malvar   = cv2.imread(malvar_path)

        if gt is None:
            print(f"  [{info['name']}] Could not load ground truth: {gt_path}")
            continue
        if bilinear is None:
            print(f"  [{info['name']}] Could not load bilinear image: {bilinear_path}")
            continue
        if malvar is None:
            print(f"  [{info['name']}] Could not load malvar image: {malvar_path}")
            continue

        psnr_bil = psnr(bilinear, gt)
        psnr_mal = psnr(malvar,   gt)
        improvement = psnr_mal - psnr_bil

        print(f"{info['name']:<10} {psnr_bil:>15.2f} {psnr_mal:>13.2f} {improvement:>+13.2f}")