import numpy as np
import cv2

img = cv2.imread('test1.png', cv2.IMREAD_COLOR)

cv2.line(img, (0,0), (150,150), (255,255,255), 15)
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# pts = pts.reshape((-1,1,2))
cv2.rectangle(img, (200,200), (650,650), (0,255,0), 15)
cv2.circle(img, (200,163),155, (0,0,255), -1)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Hello', (200,130), font , 1, (255,255,255), 5,cv2.LINE_AA)
cv2.polylines(img,[pts], True, (0,255,255), 3)
cv2.imshow('image', img)


cv2.waitKey(0)
cv2.destoryAllWindows()
