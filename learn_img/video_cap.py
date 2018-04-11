import cv2
import numpy as np
# import picamera

cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fourcc = cv2.VideoWriter_fourcc(*'VP80')
# 0x00000021

out = cv2.VideoWriter('C:/Users/jisasa3/git/streamer/output.webm', fourcc, 20.0,(640,480))

while True:
       ret,frame = cap.read()
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       cv2.imshow('video', frame)
       cv2.imshow('grayvideo', gray)
       out.write(frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

cap.release()
out.release()
cv2.destoryAllWindows()
