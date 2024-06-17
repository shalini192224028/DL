import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the image
img = cv2.imread(r'C33P1thinF_IMG_20150619_114756a_cell_181.png')

# Convert the image from BGR to RGB for display
b, g, r = cv2.split(img)
rgb_img = cv2.merge([r, g, b])

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Define a kernel for morphological operations
kernel = np.ones((2, 2), np.uint8)

# Apply closing operation to remove small holes within the foreground
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Dilate the result to get the sure background
sure_bg = cv2.dilate(closing, kernel, iterations=3)

# Plotting results
plt.figure(figsize=(12, 8))

plt.subplot(211)
plt.imshow(closing, 'gray')
plt.title("MorphologyEx: Closing (2x2)")
plt.xticks([]), plt.yticks([])

plt.subplot(212)
plt.imshow(sure_bg, 'gray')
plt.title("Dilation")
plt.xticks([]), plt.yticks([])

# Save the dilation result
plt.imsave(r'dilation.png', sure_bg, cmap='gray')

plt.tight_layout()
plt.show()
