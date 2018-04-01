import numpy as np
import cv2

img1 =cv2.imread('test2.jpeg', cv2.IMREAD_COLOR)
img2 =cv2.imread('itsme.jpeg', cv2.IMREAD_COLOR)

rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2_gray, 220, 255, cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)

img2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
dst = cv2.add(img2_bg,img2_fg)
img1[0:rows, 0:cols] = dst

# resulted = cv2.add(img1,img2)
# direct_add = img1+img2;
cv2.imshow('result', img1)
cv2.waitKey(0)
cv2.destoryAllWindows()
