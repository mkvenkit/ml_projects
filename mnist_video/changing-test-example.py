import cv2
import numpy as np

win = np.zeros((200, 200, 3), np.uint8)
number = 0
frameCount = 0
while True:
    cv2.putText(win, str(number), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    frameCount += 1
    if number >= 20:
        number = 0

    if frameCount == 70:
        win[:] = 0
        number = number+1
        frameCount = 0

    cv2.imshow('win', win)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()