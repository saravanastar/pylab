import cv2

cam_object = cv2.VideoCapture(0)

while cv2.waitKey(1) != 1:
    img = cam_object.read()[1]
    img = cv2.flip(img,1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    cv2.imshow('camera', mask_inv)
    cv2.imshow('camera1', img)
cam_object.release()
cv2.destroyAllWindows()
