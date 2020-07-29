# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 03:15:21 2020

@author: himad
"""

import cv2
import numpy as np
from skimage.filters import threshold_local
from transform import four_point_transform
#step1 : edge detection

image = cv2.imread("bill1.jpg")
cv2.imshow("original", image)
orig = image.copy()
ratio = image.shape[0]/500.0

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 75, 200)
cv2.imshow("edged", edged)

#step2 : find the contour
cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse= True)[:5]
global screenCnt

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    
    if len(approx)==4:
        screenCnt = approx
        break;

cv2.drawContours(image, [screenCnt], -1, (0, 255,0), 2)
cv2.imshow("outline", image)
print(screenCnt)

#perspective transform

warped = four_point_transform(orig, screenCnt.reshape(4,2))

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#th2 = cv2.adaptiveThreshold(warped, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,2)
T= threshold_local(warped, 13, offset=10, method="gaussian") #13 is the block size
warped = (warped>T).astype("uint8")*255

cv2.imshow("warped", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
