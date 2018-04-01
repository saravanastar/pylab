import numpy as np
import cv2
import pandas as pd

img = cv2.imread('itsme.jpeg', cv2.IMREAD_COLOR)

# for i in range(len(img)):
#     print(img[i])
    # for j in range(len(img[i])):
        # print(img[i][j])
df = pd.DataFrame.from_records(img)
watch_face = img[37:211, 107:210]
img[0:174, 0:103] = watch_face
# img[37:211, 87:210] = [255,255,0]
cv2.imshow('test', img)
print(df)
cv2.waitKey(0)
cv2.destoryAllWindows()
