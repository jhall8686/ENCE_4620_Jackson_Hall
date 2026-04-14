import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

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

# ── Load all four images ──────────────────────────────────────────────────────
bayer_images = {}
for info in IMAGE_INFO:
    path = os.path.join(DATA_DIR, info["file"])
    bayer = load_bayer_txt(path, info["rows"], info["cols"])
    bayer_images[info["name"]] = bayer
    print(f"  Loaded {info['name']:8s}  shape={bayer.shape}  "
          f"dtype={bayer.dtype}  min={bayer.min()}  max={bayer.max()}")
    
#── Display the raw Bayer images as grayscale ───────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
fig.suptitle("Raw Bayer Images (grayscale)", fontsize=12, fontweight="bold", y=1.02)

for ax, (name, bayer) in zip(axes, bayer_images.items()):
    ax.imshow(bayer, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{name}\n{bayer.shape[0]}×{bayer.shape[1]}")
    ax.axis("off")

plt.tight_layout()
plt.show()

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

# ── Solution cell (run after completing the function above) ───────────────────

# Quick sanity check on the Onion image
bayer_test = bayer_images["Onion"]
ch_B, ch_G, ch_R = extract_bggr_channels(bayer_test)

# Verify that every pixel is covered exactly once
coverage = (ch_B > 0).astype(int) + (ch_G > 0).astype(int) + (ch_R > 0).astype(int)
print("Coverage check (should all be 1):", np.unique(coverage))

# Count pixels per channel
H, W = bayer_test.shape
print(f"B pixels: {(ch_B > 0).sum():6d}  ({(ch_B > 0).sum()/(H*W)*100:.1f}%)")
print(f"G pixels: {(ch_G > 0).sum():6d}  ({(ch_G > 0).sum()/(H*W)*100:.1f}%)")
print(f"R pixels: {(ch_R > 0).sum():6d}  ({(ch_R > 0).sum()/(H*W)*100:.1f}%)")


# ── Visualise the extracted channels for all four images ─────────────────────
channel_labels = ["Blue (B)", "Green (G)", "Red (R)"]
cmaps          = ["Blues",    "Greens",    "Reds"   ]

fig, axes = plt.subplots(4, 4, figsize=(14, 13))
fig.suptitle("Raw Bayer  |  Blue channel  |  Green channel  |  Red channel",
             fontsize=11, fontweight="bold")

for row_idx, (name, bayer) in enumerate(bayer_images.items()):
    chB, chG, chR = extract_bggr_channels(bayer)

    # Column 0: raw Bayer (grayscale)
    axes[row_idx, 0].imshow(bayer, cmap="gray", vmin=0, vmax=255)
    axes[row_idx, 0].set_title(f"{name} — Raw Bayer", fontsize=9)

    # Columns 1-3: individual channels
    for col_idx, (ch, lbl, cm) in enumerate(zip([chB, chG, chR],
                                                channel_labels, cmaps), start=1):
        axes[row_idx, col_idx].imshow(ch, cmap=cm, vmin=0, vmax=255)
        axes[row_idx, col_idx].set_title(lbl, fontsize=9)

    for ax in axes[row_idx]:
        ax.axis("off")

plt.tight_layout()
plt.show()
print("Notice the sparse, grid-like appearance of each channel — "
      "only 25% of positions are measured for B and R, 50% for G.")