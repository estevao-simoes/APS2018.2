import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv2.THRESH_BINARY, 11, 2)
    # ret3, th3 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary = cv2.bitwise_not(gray)

    (_, contours, _) = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        if cv2.contourArea(contour) > 5000:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ret, th = cv2.threshold(gray, 127, 255, 1)

    # im2, contours, hierarchy = cv.findContours(thresh, 1, 2)

    cv2.imshow('original', frame)
    cv2.imshow('threshold', binary)
    # cv2.imshow('threshold2', th3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.waitKey(0)
cv2.destroyAllWindows()
