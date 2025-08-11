import cv2
import dlib

img = cv2.imread('../img/people.jpg')
img_resized = cv2.resize(img, (755,500))

cnn_face_detector = dlib.cnn_face_detection_model_v1('../data/mmod_human_face_detector.dat')
# The second parameter is also a scale related parameter.
face_detections = cnn_face_detector(img_resized, 1)

# c is the confidence indicating the reliability of a single detection.
for idx, face_detection in enumerate(face_detections):
    left, top, right, bottom, confidence = face_detection.rect.left(), face_detection.rect.top(), face_detection.rect.right(), face_detection.rect.bottom(), face_detection.confidence
    print(f'confidence{idx+1}: {confidence}')  # print confidence of the detection
    cv2.rectangle(img_resized, (left, top), (right, bottom), (0, 255, 0), 2)

cv2.imshow('people', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
