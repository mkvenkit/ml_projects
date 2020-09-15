'''
Program to access the webcam connected to the device and threshold the obtained video
Author: Aryan Mahesh
'''

import cv2
import numpy as np

def on_threshold(x):
    pass

cap = cv2.VideoCapture(2)
frame = cv2.namedWindow('background')
cv2.createTrackbar('threshold', 'background', 0, 255, on_threshold)
background = np.zeros((480, 640), np.uint8)
frameCount = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    frameCount += 1

    frame[0:480, 0:80] = 0
    frame[0:480, 560:640] = 0
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    threshold = cv2.getTrackbarPos('threshold', 'background')
    _, thr = cv2.threshold(grayFrame, threshold, 255, cv2.THRESH_BINARY)
    cv2.putText(background, 'X', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    resizedFrame = thr[240-75:240+75, 320-75:320+75]
    background[240-75:240+75, 320-75:320+75] = resizedFrame

    iconImg = cv2.resize(resizedFrame, (28, 28))
    if frameCount == 20:
        frame[0:480, 0:80] = 0
    
    
    cv2.imshow('background', background)
    # cv2.imshow('resized', resizedFrame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
