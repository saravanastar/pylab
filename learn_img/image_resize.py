import numpy as np
import cv2

img = cv2.imread('test1.png')
resized = cv2.resize(img, (260,339))
cv2.imshow('resize',resized)
cv2.imwrite('test2.jpeg', resized)
cv2.waitKey(0)
cv2.destoryAllWindows()
