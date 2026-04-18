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

# ── Gradient-based correction ─────────────────────────────────────────────

# Gradient kernels

G_AT_R = np.array([
    [ 0,  0, -1,  0,  0],
    [ 0,  0,  2,  0,  0],
    [-1,  2,  4,  2, -1],
    [ 0,  0,  2,  0,  0],
    [ 0,  0, -1,  0,  0],
], dtype=np.float32) / 8.0
 
G_AT_B = G_AT_R  # symmetric
 
# R at green pixel in R-row, B-column  (beta = 5/8)
R_AT_G_RRow_BCol = np.array([
    [ 0,  0,  0.5,  0,  0],
    [ 0, -1,  0,  -1,  0],
    [-1,  4,  5,   4, -1],
    [ 0, -1,  0,  -1,  0],
    [ 0,  0,  0.5, 0,  0],
], dtype=np.float32) / 8.0
 
# R at green pixel in B-row, R-column  (beta = 5/8)
R_AT_G_BRow_RCol = np.array([
    [ 0,  0, -1,  0,  0],
    [ 0, -1,  4, -1,  0],
    [ 0.5, 0, 5,  0,  0.5],
    [ 0, -1,  4, -1,  0],
    [ 0,  0, -1,  0,  0],
], dtype=np.float32) / 8.0
 
# R at B locations (gamma = 3/4)
R_AT_B = np.array([
    [ 0,  0, -1.5,  0,  0],
    [ 0,  2,  0,   2,  0],
    [-1.5, 0, 6,   0, -1.5],
    [ 0,  2,  0,   2,  0],
    [ 0,  0, -1.5,  0,  0],
], dtype=np.float32) / 8.0

# B kernels are symmetric counterparts of R kernels
B_AT_G_BRow_RCol = R_AT_G_RRow_BCol
B_AT_G_RRow_BCol = R_AT_G_BRow_RCol
B_AT_R           = R_AT_B

def gradient_correction(bayer):
    """
    Apply gradient-based correction to the bilinear demosaiced image.
    
    Parameters:
        bayer (np.ndarray): The original Bayer pattern image (H x W).
    Returns:
        gradient_corrected (np.ndarray): The gradient-corrected RGB image (H x W x 3).
    """

    H, W = bayer.shape
    # Work in float32 for intermediate averages
    src = bayer.astype(np.float32)

    R_in, G_in, B_in = extract_bggr_channels(bayer)

    # Allocate output channels
    R_out = np.zeros((H, W), dtype=np.float32)
    G_out = np.zeros((H, W), dtype=np.float32)
    B_out = np.zeros((H, W), dtype=np.float32)

    # Boolean masks for the four Bayer site types- Using different syntax from the given bilinear algorithm, but I think this makes more sense to look at
    is_B = np.zeros((H,W), bool)
    is_Gr = np.zeros((H,W), bool)
    is_Gb = np.zeros((H,W), bool)
    is_R = np.zeros((H,W), bool)

    is_B[0::2, 0::2] = True   # even row, even col 
    is_Gr[0::2, 1::2] = True   # even row, odd col
    is_Gb[1::2, 0::2] = True   # odd row,  even col
    is_R[1::2, 1::2] = True   # odd row,  odd col
    
    # even rows are red rows, odd rows are blue rows
    is_R_row = np.zeros((H, W), bool)
    is_R_row[0::2, :] = True
    is_B_row = ~is_R_row
    is_R_col = np.zeros((H, W), bool)
    is_R_col[:, 0::2] = True
    is_B_col = ~is_R_col

    # convolution helper function using cv2.filter2D
    def conv(kernel):
        return cv2.fliter2D(src, kernel, cv2.BORDER_REFLECT)
 
    conv_G_on_R   = conv(G_AT_R)
    conv_G_on_B   = conv(G_AT_B)
    conv_R_on_GRrowBcol  = conv(R_AT_G_RRow_BCol)
    conv_R_on_GBrowRcol  = conv(R_AT_G_BRow_RCol)
    conv_R_on_B   = conv(R_AT_B)
    conv_B_on_GBrowRcol  = conv(B_AT_G_BRow_RCol)
    conv_B_on_GRrowBcol  = conv(B_AT_G_RRow_BCol)
    conv_B_on_R   = conv(B_AT_R)
    
    # Constructing the filtered blue channel

    B_out = np.where(is_R, conv_B_on_R, B_in)
    B_out = np.where(is_)
    # Constructing the filtered green channel
    G_out = np.where(is_R, conv_G_on_R, G_in)
    G_out = np.where(is_B, conv_G_on_B, G_in)

    

