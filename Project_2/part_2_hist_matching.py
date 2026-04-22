from part_1_histeq_from_scratch import hist_eq, read_imgs
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SRC_IMG_DIR = 'source_images'
TGT_IMG_DIR = 'target_images'
extensions = ('*.tif','*.jpg')
levels = 256

def hist_matching(src_imgs, tgt_imgs, L):
    """
    hist_matching
    Args:
        src_imgs (list of np.ndarray): The images to be processed
        tgt_imgs (list of np.ndarray): The images to be matched to
        L (int): The number of levels (bins) in the image
    Returns:
        matched_imgs (list of np.ndarray): The matched images
    """

    # equalize source and target images to get sk = T(rk), vk = G(zk)

    src_eqs = hist_eq(src_imgs, L)
    tgt_eqs = hist_eq(tgt_imgs, L)
    
    # define look up table by finding the closest value in the target cdf (vk) for each pixel in the source cdf (sk)
    luts = []
    for (src_eq, tgt_eq) in zip(src_eqs, tgt_eqs):
        lut = [np.astype(np.argmin(np.abs(tgt_eq - src_eq[i])),np.uint8) for i in range(L-1)]
        luts.append(lut)
    matched_imgs = []
    for (src_img, lut) in zip(src_imgs, luts):
        matched_img = lut[src_img]
        matched_imgs.append(matched_img)
    return matched_imgs


def main():
    # Read images from directory

    src_images = read_imgs(SRC_IMG_DIR)
    tgt_images = read_imgs(TGT_IMG_DIR)
    print('read images')

    matched_images = hist_matching(src_images,tgt_images,levels)
    
    print('matched images')

    fig, axes = plt.subplots(3, len(src_images), figsize=(16, 12))

    for i, (src_img, tgt_img, matched_img) in enumerate(zip(src_images, tgt_images, matched_images)):
        axes[0, i].imshow(src_img, cmap='gray')
        axes[0, i].set_title(f'Source Image {i+1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(tgt_img, cmap='gray')
        axes[1, i].set_title(f'Target Image {i+1}')
        axes[1, i].axis('off')

        axes[2, i].imshow(matched_img, cmap='gray')
        axes[2, i].set_title(f'Histogram Matched Image {i+1}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()