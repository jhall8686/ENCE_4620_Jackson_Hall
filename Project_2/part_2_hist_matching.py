from part_1_histeq_from_scratch import hist_eq, read_imgs
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import match_histograms

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

    src_cdfs, src_eqs = hist_eq(src_imgs, L)
    tgt_cdfs, tgt_eqs = hist_eq(tgt_imgs, L)
    
    # define look up table by finding the closest value in the target cdf (vk) for each pixel in the source cdf (sk)
    matched_imgs = []
    for (src_img, src_cdf, tgt_cdf) in zip(src_imgs, src_cdfs, tgt_cdfs):
        lut = np.array([np.argmin(np.abs(tgt_cdf.flatten() - src_cdf.flatten()[i])) for i in range(L)])
        matched_imgs.append(lut[src_img])
    return matched_imgs

def main():
    # Read images from directory

    src_images = read_imgs(SRC_IMG_DIR)
    tgt_images = read_imgs(TGT_IMG_DIR) # repeated to match the length of src_images, flipped bc of file order
    tgt_images = [img for img in tgt_images for _ in range(2)][::-1]
    print('read images')

    matched_images = hist_matching(src_images,tgt_images,levels)
    sk_matched_images = []
    for src_img, tgt_img in zip(src_images, tgt_images):
        sk_matched_images.append(match_histograms(src_img,tgt_img))

    print('matched images')

    fig, axes = plt.subplots(3, len(src_images), figsize=(16, 12))

    # COMPARISON- Source image, target image, my implemented method
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

    # COMPARISON- Source image, my implemented method, skimage method
    for i, (src_img, matched_img, sk_matched_img) in enumerate(zip(src_images, matched_images, sk_matched_images)):
        axes[0, i].imshow(src_img, cmap='gray')
        axes[0, i].set_title(f'Source Image {i+1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(matched_img, cmap='gray')
        axes[1, i].set_title(f'Target Image {i+1}')
        axes[1, i].axis('off')

        axes[2, i].imshow(sk_matched_img, cmap='gray')
        axes[2, i].set_title(f'Histogram Matched Image {i+1}')
        axes[2, i].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()