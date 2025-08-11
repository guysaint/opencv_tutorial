import numpy as np
import cv2

# 얼굴 검출을 위한 케스케이드 분류기 생성
face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')

# 눈 검출을 위한 케스케이드 분류기 생성
eye_cascade = cv2.CascadeClassifier('../data/haarcascade_eye.xml')

# 검출할 이미지 읽고 그레이 스케일로 변환
img = cv2.imread('../img/people.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = cv2.equalizeHist(gray)
gray = cv2.convertScaleAbs(gray, alpha=1.1, beta=10)

# 얼굴 검출
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))

# 검출된 얼굴 순회
for (x,y,w,h) in faces:
    # 검출된 얼굴에 사각형 표시
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # 얼굴 영역을 ROI로 설정
    roi = gray[y:y+h, x:x+w]
    # ROI에서 눈 검출
    roi_upper = roi[0:int(h*0.6), :]
    eyes = eye_cascade.detectMultiScale(roi_upper, scaleFactor=1.1, minNeighbors=12, minSize=(20, 20))
    # 검출된 눈에 사각형 표
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img[y:y+h, x:x+w],(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# 결과 출력 
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()