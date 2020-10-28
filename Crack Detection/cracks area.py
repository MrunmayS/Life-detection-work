import cv2
import numpy as np 

img = cv2.imread('E:\Life Detection\Crack Detection\cracks1.png', 0)
image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#img = cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
cnt = np.array(contours)
area = cv2.contourArea(cnt)
"""
cv2.imshow('im', img)
cv2.waitKey()
"""