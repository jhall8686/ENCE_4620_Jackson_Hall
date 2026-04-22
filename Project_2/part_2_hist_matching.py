from part_1_histeq_from_scratch import hist_eq
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

SRC_IMG_DIR = 'source_images'
TGT_IMG_DIR = 'target_images'
extensions = ('*.tif','*.jpg')
levels = 256

def main():
    # Read images from directory
    src_images = []
    SRC_IMG_DIR = Path(SRC_IMG_DIR)
    for path in SRC_IMG_DIR.iterdir():
        if path.suffix.lower() in extensions:
            src = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if src is not None: #<- I just learned that 'is not' does a object identity check.
                src_images.append(src)
            else:
                raise FileNotFoundError(f'Could not load source image at path {path}')
    tgt_images = []
    TGT_IMG_DIR = Path(TGT_IMG_DIR)
    for path in TGT_IMG_DIR.iterdir():
        if path.suffix.lower() in extensions:
            tgt = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if tgt is not None:
                tgt_images.append(tgt)
            else:
                raise FileNotFoundError(f'Could not load target image at path {path}')
    
    
    
if __name__ == 'main':
    main()