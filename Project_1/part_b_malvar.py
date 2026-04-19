import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from scipy.ndimage import convolve

# ── Helper: load one text file → 2-D uint8 array ─────────────────────────────
def load_bayer_txt(filepath, rows, cols):
    """
    Read a space-delimited Bayer raw image from a .txt file.

    Parameters
    ----------
    filepath : str   path to the .txt file
    rows     : int   expected number of image rows
    cols     : int   expected number of image columns

    Returns
    -------
    bayer : np.ndarray, shape (rows, cols), dtype uint8
    """
    data = []
    with open(filepath, "r") as fh:
        for line in fh:
            row_vals = list(map(int, line.strip().split()))
            data.append(row_vals)
    bayer = np.array(data, dtype=np.uint8)
    assert bayer.shape == (rows, cols), (
        f"Shape mismatch: expected ({rows},{cols}), got {bayer.shape}"
    )
    return bayer

#── Function to extract separate B, G, R channels from a BGGR Bayer array ─────
def extract_bggr_channels(bayer):
    """
    Extract separate B, G, R channel images from a BGGR Bayer array.

    Unknown pixel locations are set to 0.

    Parameters
    ----------
    bayer : np.ndarray, shape (H, W), dtype uint8

    Returns
    -------
    ch_B, ch_G, ch_R : each np.ndarray shape (H, W), dtype uint8
    """
    H, W = bayer.shape
    ch_B = np.zeros((H, W), dtype=np.uint8)
    ch_G = np.zeros((H, W), dtype=np.uint8)
    ch_R = np.zeros((H, W), dtype=np.uint8)

    # BGGR layout:
    #   B  → even rows, even cols
    #   G  → even rows, odd cols  AND  odd rows, even cols
    #   R  → odd rows,  odd cols

    # --- Blue channel ---
    ch_B[0::2, 0::2] = bayer[0::2, 0::2]   # even row, even col

    # --- Green channel (two interleaved sub-grids) ---
    ch_G[0::2, 1::2] = bayer[0::2, 1::2]   # even row, odd col
    ch_G[1::2, 0::2] = bayer[1::2, 0::2]   # odd row,  even col

    # --- Red channel ---
    ch_R[1::2, 1::2] = bayer[1::2, 1::2] # odd row, odd col

    return ch_B, ch_G, ch_R

# ── Gradient-based correction ─────────────────────────────────────────────

def malvar_demosaic_bggr(bayer):
    """
    Apply the Malvar gradient kernels to a BGGR Bayer array.
    
    Parameters:
        bayer (np.ndarray): The original Bayer pattern image (H x W).
    Returns:
        bgr (np.ndarray): The gradient-corrected demosaiced image (H x W x 3).
    """

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
    ], dtype=np.float32)/ 8.0
    
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

    H, W = bayer.shape
    # Work in float32 for intermediate averages
    src = bayer.astype(np.float32)

    B_in, G_in, R_in = extract_bggr_channels(bayer)

    print(R_in[1::2, 1::2].max())
    print(R_in[0::2, 0::2].max())

    # Allocate output channels
    R_out = np.zeros((H, W), dtype=np.float32)
    G_out = np.zeros((H, W), dtype=np.float32)
    B_out = np.zeros((H, W), dtype=np.float32)

    
    # Boolean masks for the four Bayer site types
    rows, cols = np.mgrid[0:H, 0:W]
    is_B  = (rows % 2 == 0) & (cols % 2 == 0)   # even row, even col
    is_Gb = (rows % 2 == 0) & (cols % 2 == 1)   # even row, odd col  (Green on Red row)
    is_Gr = (rows % 2 == 1) & (cols % 2 == 0)   # odd row,  even col (Green on Blue row)
    is_R  = (rows % 2 == 1) & (cols % 2 == 1)   # odd row,  odd col

    # convolution helper function using cv2.filter2D
    def conv(kernel):
        return convolve(src,kernel,mode="mirror")
 
    conv_G_on_R   = conv(G_AT_R)
    conv_G_on_B   = conv(G_AT_B)
    conv_R_on_GRrow  = conv(R_AT_G_RRow_BCol)
    conv_R_on_GBrow  = conv(R_AT_G_BRow_RCol)
    conv_R_on_B   = conv(R_AT_B)
    conv_B_on_GBrow  = conv(B_AT_G_BRow_RCol)
    conv_B_on_GRrow  = conv(B_AT_G_RRow_BCol)
    conv_B_on_R   = conv(B_AT_R)

    #Blue pixels
    B_out[is_B] = B_in[is_B]
    G_out[is_B] = conv_G_on_B[is_B]
    R_out[is_B] = conv_R_on_B[is_B]

    #Green pixels on red rows
    B_out[is_Gr] = conv_B_on_GRrow[is_Gr]
    G_out[is_Gr] = G_in[is_Gr]
    R_out[is_Gr] = conv_R_on_GRrow[is_Gr]
    
    #Green pixels on blue rows
    B_out[is_Gb] = conv_B_on_GBrow[is_Gb]
    G_out[is_Gb] = G_in[is_Gb]
    R_out[is_Gb] = conv_R_on_GBrow[is_Gb]

    #Red pixels
    B_out[is_R] = conv_B_on_R[is_R]
    G_out[is_R] = conv_G_on_R[is_R]
    R_out[is_R] = R_in[is_R]

    print("R_out max:", R_out.max())
    print("G_out max:", G_out.max())
    print("B_out max:", B_out.max())
    B_out = np.clip(B_out,0,255).astype(np.uint8)
    G_out = np.clip(G_out,0,255).astype(np.uint8)
    R_out = np.clip(R_out,0,255).astype(np.uint8)


    bgr = cv2.merge([B_out, G_out, R_out])
    return bgr

def main():
    # Display settings
    plt.rcParams['figure.dpi'] = 110
    plt.rcParams['axes.titlesize'] = 10

    print(f"NumPy  version : {np.__version__}")
    print(f"OpenCV version : {cv2.__version__}")

    # ── File catalogue ────────────────────────────────────────────────────────────
    # Update DATA_DIR to point to the folder that contains the .txt files
    DATA_DIR = "../bayer-images-uint8/"   # <-- change this if your files are in a sub-folder

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


    demosaiced = {}
    for name, bayer in bayer_images.items():
        bgr = malvar_demosaic_bggr(bayer)
        demosaiced[name] = bgr
        print(f"  {name:8s}  output shape={bgr.shape}  dtype={bgr.dtype}")

    # ── Side-by-side: Raw Bayer (gray) vs Demosaiced with Malvar Kernels (color) ────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(9, 14))
    fig.suptitle("Malvar Demosaicing Results", fontsize=13, fontweight="bold")

    for row_idx, (name, bayer) in enumerate(bayer_images.items()):
        bgr = demosaiced[name]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # convert BGR→RGB for matplotlib

        axes[row_idx, 0].imshow(bayer, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 0].set_title(f"{name} — Raw Bayer")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(rgb)
        axes[row_idx, 1].set_title(f"{name} — Malvar Demosaiced")
        axes[row_idx, 1].axis("off")

    plt.tight_layout()
    plt.show()

    OUTPUT_DIR = "malvar-demosaic-results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, bgr in demosaiced.items():
        filename = os.path.join(OUTPUT_DIR, f"{name}_malvar.png")
        cv2.imwrite(filename, bgr)
        print(f"Saved: {filename}")

    print("\nAll images saved.")
    
if __name__ == "__main__":
    main()