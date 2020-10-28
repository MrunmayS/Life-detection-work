import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('E:\Life Detection\Gabor Filters\img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
"""
cv2.namedWindow("Trackbars")

def nothing(x):
    pass

while True:
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    res = cv2.bitwise_and(img, img, mask=mask)
    mask_2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((mask_2, img, res))
    stacked = cv2.resize(stacked,None,fx=0.4,fy=0.4)
    cv2.imshow('video',stacked)
    #cv2.imshow('og', img_og)
    key = cv2.waitKey(1)
    if key ==ord('q'):
        break  
    if key == ord('s'):
    
        arr1 = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
        print(arr10
        
    #np.save('penval_green_01', arr1)

cap.release()
cv2.destroyAllWindows()
"""
max_val = 255

thres =170
ret, output = cv2.threshold(img, thres,max_val,cv2.THRESH_TOZERO )
plt.imshow(output)
plt.show()