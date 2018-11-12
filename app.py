import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 127, 255, 1)
    
    im2, contours, hierarchy = cv.findContours(thresh, 1, 2)

    cv2.imshow('original', frame)
    cv2.imshow('Adaptive threshold', th)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.waitKey(0)
cv2.destroyAllWindows()
