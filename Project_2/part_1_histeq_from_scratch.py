import cv2
import os
import glob
import matplotlib.pyplot as plt

IMG_DIR = 'source_images'
levels = 256

def hist_eq(imgs, L):
    """
    hist_eq
    Args:
        imgs (list of numpy.ndarray): A list of input images (grayscale or single-channel).
        L (int): The number of intensity levels (e.g., 256 for 8-bit images).
    Returns:
        tuple: A tuple containing three lists:
            - hists (list of numpy.ndarray): Normalized histograms/PDFs for each image
            - cdfs (list of numpy.ndarray): CDFs for each image
            - eqs (list of numpy.ndarray): Equalized histograms for each image, scaled to [0, L-1].
    """
    hists = []
    cdfs =  []
    eqs =   []
    for img in imgs:
        # Calculating normalized histogram for each image (PDF)
        hist = cv2.calcHist([img],[0],None,[L],[0, L])
        hist /= hist.sum()
        hists.append(hist)

        # Calculating CDF for each image
        cdf = hist.cumsum()
        cdfs.append(cdf)

        # Calculating the equalized for each histogram
        eq = (cdf*(L-1)).round()
        eqs.append(eq)
        
    return hists, cdfs, eqs

def main():
    # Read images from directory
    images = []
    for path in glob.glob(os.path.join(IMG_DIR, '*.tif')):
        src = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if src is not None: #<- I just learned that 'is not' does a memory comparison. kinda cool
            images.append(src)
        else:
            raise FileNotFoundError(f'Could not load image at path {path}')

    # Perform histogram equalization
    hists, cdfs, eqs = hist_eq(images, levels)

    # Apply equalized histogram transformation to each image
    equalized_images = []
    for i, img in enumerate(images):
        img_eq = eqs[i][img]
        equalized_images.append(img_eq)

    # Do the same with the cv2 method
    CV_equalized_images = []
    for img in images:
        img_eq = cv2.equalizeHist(img)
        CV_equalized_images.append(img_eq)

    # PLOTTING

    # Histograms, CDFS, and Equalized Histogram Values
    # labels = ['Circuit', 'Forest']
    # colors = ['black', 'green']
    # data = [hists, cdfs, eqs]
    # titles = ['Normalized Histogram (PDF)', 'CDF', 'Histogram Equalized']
    # ylabels = ['Probability', 'Cumulative Probability', 'Cumulative Probability']

    # fig, axes = plt.subplots(1, 3, figsize=(18,4))

    # for ax, d, title, ylabel in zip(axes, data, titles, ylabels):
    #     for i, (values, label, color) in enumerate(zip(d, labels, colors)):
    #         ax.plot(values, label=label, color=color)
    #     ax.set_title(title)
    #     ax.set_xlabel('Pixel Intensity')
    #     ax.set_ylabel(ylabel)
    #     ax.legend()

    # plt.tight_layout()
    # plt.show()

    # Plot all three sets of images: Original, from-scratch equalization, and OpenCV equalization

    fig, axes = plt.subplots(3, len(images), figsize=(12, 12))

    for i, (img, eq_img, cv_eq_img) in enumerate(zip(images, equalized_images, CV_equalized_images)):
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(eq_img, cmap='gray')
        axes[1, i].set_title(f'Equalized {i+1}')
        axes[1, i].axis('off')

        axes[2, i].imshow(cv_eq_img, cmap='gray')
        axes[2, i].set_title(f'CV Equalized {i+1}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()
    
        
        

if __name__ == '__main__':
    main()