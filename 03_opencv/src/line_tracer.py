import cv2
import matplotlib.pylab as plt
import numpy as np

cap = cv2.VideoCapture(1) 
            # 1번 카메라 장치 연결 ---①
if cap.isOpened():                      # 캡쳐 객체 연결 확인
    while True:
        ret, img = cap.read()
        if ret:
            #BGR -> GRAY
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #그레이 히스토그램 계산
            hist = cv2.calcHist([gray], [0], None, [256], [0,256])
            cv2.normalize(hist, hist, 0, 100, cv2.NORM_MINMAX)

            #히스토그램 시각화용 이미지 생성
            hist_img = np.full((100,256), 255, dtype=np.uint8)
            for x,y in enumerate(hist):
                cv2.line(hist_img, (x, 100), (x, 100-int(y)), 0)

            #화면 출력
            cv2.imshow('camera(gray)', gray)   # GRAY로 카메라 표시
            cv2.imshow('histogram(gray)', hist_img)

            #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #BGR -> HSV로 변환
            #cv2.imshow('camera', hsv) #HSV로 카메라 표시
            key = cv2.waitKey(1) & 0xFF
                        
            if key == ord('q'):
                break
            if key == ord("s"):            # s키를 누르면
                cv2.imwrite('../img/photo.jpg', img)    #phto.jpg로 사진 저장

        else:
            print('no frame')
            break
else:
    print("can't open camera.")

cap.release()    #자원반납
cv2.destroyAllWindows()