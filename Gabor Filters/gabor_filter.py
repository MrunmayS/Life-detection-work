import numpy as np
import cv2
import matplotlib.pyplot as plt


img = cv2.imread('E:\Life Detection\Gabor Filters\img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_1 = img.reshape(-1)
ksize = 5
sigma = 3
theta = 1*np.pi/4
lam = 1*np.pi/1
gamma = 0.5
phi = 0

g_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma, phi, ktype=cv2.CV_32F)
filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
h, w = g_kernel.shape[:2]
g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
plt.imshow(filtered_img)
plt.show()