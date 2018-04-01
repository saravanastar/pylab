import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/jisasa3/Pictures/Saved Pictures/ai_0.jpg',cv2.IMREAD_COLOR)

cv2.imshow('test', img);
cv2.imwrite('test1.png', img)
cv2.waitKey(0)
cv2.destoryAllWindows()
##cv2 = BGR

# plt.imshow(img,cmap='gray', interpolation='bicubic')
# plt.plot([20,200],[50,100], 'c', linewidth=5)
# plt.show()
