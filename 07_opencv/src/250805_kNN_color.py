import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():


    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 종료
        break

cv2.destroyAllWindows()