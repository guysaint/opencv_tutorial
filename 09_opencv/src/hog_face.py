import cv2
import numpy as np
import dlib

img = cv2.imread('../img/people.jpg')
img_resized = cv2.resize(img, (755,500))

hog_face_detector = dlib.get_frontal_face_detector()
face_detections = hog_face_detector(img_resized, 1)
print(face_detections)
for face_detection in face_detections:
    left, top, right, bottom = face_detection.left(), face_detection.top(), face_detection.right(), face_detection.bottom()
    cv2.rectangle(img_resized, (left, top), (right, bottom), (0, 255, 0), 2)
    
cv2.imshow('people', img_resized)

#cv2.imshow('people', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
