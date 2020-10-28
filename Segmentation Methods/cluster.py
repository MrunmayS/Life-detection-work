import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('E:\Life Detection\Gabor Filters\img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pixel_values = img.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

k = 3
_, labels, (centres) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centres = np.uint8(centres)
labels = labels.flatten()
segmented_img = centres[labels.flatten()]
segmented_img = segmented_img.reshape(img.shape)
#plt.imshow(segmented_img)
#plt.show()

img2 = np.copy(img)
img2 = img2.reshape((-1,3))
cluster_number = 1
img2[labels == cluster_number] = [0, 0, 0]
cluster_number = 0
img2[labels == cluster_number] = [0, 0, 0]
img2 = img2.reshape(img.shape)
plt.imshow(img2)
plt.show()