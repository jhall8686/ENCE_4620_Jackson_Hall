import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from part_a_bilinear import bilinear_demosaic_bggr, load_bayer_txt, extract_bggr_channels
# Display settings
plt.rcParams['figure.dpi'] = 110
plt.rcParams['axes.titlesize'] = 10

print(f"NumPy  version : {np.__version__}")
print(f"OpenCV version : {cv2.__version__}")

# ── File catalogue ────────────────────────────────────────────────────────────
# Update DATA_DIR to point to the folder that contains the .txt files
DATA_DIR = "bayer-images-uint8/"   # <-- change this if your files are in a sub-folder

IMAGE_INFO = [
    {"name": "Onion",   "file": "onionBayer.txt",   "rows": 135, "cols": 198},
    {"name": "Peppers", "file": "peppersBayer.txt", "rows": 384, "cols": 512},
    {"name": "Office",  "file": "officeBayer.txt",  "rows": 600, "cols": 903},
    {"name": "Pears",   "file": "pearsBayer.txt",   "rows": 486, "cols": 732},
]

# ── Load Bayer images from text files ───────────────────────────────────────
bayer_images = {}
for info in IMAGE_INFO:
    path = os.path.join(DATA_DIR, info["file"])
    bayer_images[info["name"]] = load_bayer_txt(path, info["rows"], info["cols"])
    bayer=load_bayer_txt(path, info["rows"], info["cols"])
    print(f"  Loaded {info['name']:8s}  shape={bayer.shape}  "
          f"dtype={bayer.dtype}  min={bayer.min()}  max={bayer.max()}")
    

# ── Demosaic using bilinear interpolation ────────────────────────────────



bilinear_demosaiced = {}

for name, bayer in bayer_images.items():
    bilinear_demosaiced[name] = bilinear_demosaic_bggr(bayer)

# ── Gradient-based correction ─────────────────────────────────────────────

def gradient_correction(bilinear_demosaiced, bayer):
    """
    Apply gradient-based correction to the bilinear demosaiced image.
    
    Parameters:
        bilinear_demosaiced (np.ndarray): The bilinear demosaiced RGB image (H x W x 3).
        bayer (np.ndarray): The original Bayer pattern image (H x W).
    Returns:
        gradient_corrected (np.ndarray): The gradient-corrected RGB image (H x W x 3).
    """
    

    rB,gB,bB = extract_rgb_channels(bilinear_demosaiced)

    H, W = bayer.shape
    # Work in float32 for intermediate averages
    src = bayer.astype(np.float32)

    # Pad by 2 pixels (reflect) so we never go out of bounds
    p = np.pad(src, 2, mode='reflect')

    # Allocate output channels
    R_out = np.zeros((H, W), dtype=np.float32)
    G_out = np.zeros((H, W), dtype=np.float32)
    B_out = np.zeros((H, W), dtype=np.float32)

    C = p[2:-2, 2:-2]

    # N,S,E,W correspond to p(i±2, j) and p(i, j±2) <-not AI, I found the alt code for ± it's alt+0177 :)
    
    N =  p[0:-3, 2:-2]
    S =  p[3:  , 2:-2]
    E =  p[2:-2, 3:  ]
    W_=  p[2:-2, 0:-3]
 
    # 4-diagonal neighbours
    NW  = p[0:-2, 0:-2]
    NE  = p[0:-2, 2:  ]
    SW  = p[2:  , 0:-2]
    SE  = p[2:  , 2:  ]

    # Boolean masks for the four Bayer site types
    rows, cols = np.mgrid[0:H, 0:W]
    is_B  = (rows % 2 == 0) & (cols % 2 == 0)   # even row, even col
    is_Gr = (rows % 2 == 0) & (cols % 2 == 1)   # even row, odd col  (Green on Red row)
    is_Gb = (rows % 2 == 1) & (cols % 2 == 0)   # odd row,  even col (Green on Blue row)
    is_R  = (rows % 2 == 1) & (cols % 2 == 1)   # odd row,  odd col
 
    # GRADIENT CALCULATIONS
    
    grad_R = C[is_R] - (N[is_R] + E[is_R] + S[is_R] + W_[is_R]) / 4.0
    grad_B = C[is_B] - (N[is_B] + E[is_B] + S[is_B] + W_[is_B]) / 4.0
    
    # ??


def extract_rgb_channels(img):
    """
    Extract the R, G, B channels from an RGB image.
    
    Parameters:
        img (np.ndarray): The RGB image (H x W x 3).
    Returns:
        ch_R (np.ndarray): The R channel (H x W). 
        ch_G (np.ndarray): The G channel (H x W).
        ch_B (np.ndarray): The B channel (H x W).
    """
    ch_R = img[:, :, 0]
    ch_G = img[:, :, 1]
    ch_B = img[:, :, 2]
    return ch_R, ch_G, ch_B
