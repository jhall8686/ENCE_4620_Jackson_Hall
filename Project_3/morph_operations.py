import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(title, img, cmap=None):
    plt.figure(figsize=(6,4))
    plt.imshow(img,cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def process_image(image_path, object_name='object',kernel_size=5,kernel_type=cv2.MORPH_ELLIPSE,min_area=50,max_area=100000,threshold=cv2.THRESH_BINARY):
    # 1. Load image using OpenCV
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')
    
    # Convert to RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    show_image("Original RGB", rgb)
    show_image("Grayscale", gray, cmap="gray")

    # 2. Thresholding: try Otsu's method
    # Depending on the image, you may need THRESH_BINARY or THRESH_BINARY_INV
    otsu_value, binary = cv2.threshold(gray, 0, 255, threshold +cv2.THRESH_OTSU)
    print("Otsu threshold value:", otsu_value)
    show_image("Initial binary mask", binary, cmap="gray")

    # 3. Morphological processing
    # Try different kernel sizes and shapes.
    kernel = cv2.getStructuringElement(kernel_type, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary,cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    out = cv2.morphologyEx(closed, cv2.MORPH_ERODE, kernel)


    show_image("After opening + closing", closed, cmap="gray")
    show_image("After eroding", out, cmap="gray")

    # 4. Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(out, connectivity=8)

    # 5. Filter components by area (now parameters in the function)
    #min_area = 50           # adjust based on the image
    #max_area = 10000       # adjust based on the image
    valid_centers = []
    valid_count = 0

    output = rgb.copy()
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            valid_count += 1
            cx, cy = centroids[label_id]
            valid_centers.append((cx,cy))
            cv2.circle(output, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            cv2.putText(output,str(valid_count), (int(cx)+5, int(cy)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    show_image(f"Detected {object_name} centers", output)
    print(f"Number of detected {object_name}s: ", valid_count)

    return valid_count, valid_centers, closed

def main():
    count1, centers1, mask1 = process_image("neuron.jpg", object_name="neuron", min_area=1000)
    count2, centers2, mask2 = process_image("Blood-cells_12.Red-blood-ce.jpg", object_name="cell",threshold=cv2.THRESH_BINARY,min_area=200,kernel_size=5,kernel_type=cv2.MORPH_ELLIPSE)


if __name__ == '__main__':
    main()