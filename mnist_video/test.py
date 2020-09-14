import cv2
import numpy as np

cap = cv2.VideoCapture(2)
grayVal = 30
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    frame = frame*0.9
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()