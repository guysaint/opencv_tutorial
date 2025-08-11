import cv2
import numpy as np
import dlib

img = cv2.imread('../img/people.jpg')
img_resized = cv2.resize(img, (755,500))

#hog_face_detector = dlib.get_frontal_face_detector()
#face_detections = hog_face_detector(img_resized, 1)
#print(face_detections)


cv2.imshow('people', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
