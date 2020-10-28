import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img1.jpeg', 0)

fudgefactor = 0.9 
sigma = 21 
kernel = 2*math.ceil(2*sigma)+1
kernel = int(kernel)
img = img/255
blur = cv2.GaussianBlur(img, (kernel, kernel), sigma)
img = cv2.subtract(img, blur)


sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
mag = np.hypot(sobelx, sobely)
ang = np.arctan2(sobely, sobelx)

threshold = 4 * fudgefactor * np.mean(mag)
mag[mag < threshold] = 0

mag = cv2.normalize(mag, 0, 255, cv2.NORM_MINMAX)
kernel = np.ones((5,5),np.uint8)
result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
cv2.imshow('im', result)
cv2.waitKey()
