import cv2
import matplotlib.pylab as plt

cap = cv2.VideoCapture(1)             # 1번 카메라 장치 연결 ---①
if cap.isOpened():                      # 캡쳐 객체 연결 확인
    while True:
        ret, img = cap.read()
                   # 다음 프레임 읽기
        if ret:
            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #BGR -> GRAY 변환
            cv2.imshow('camera', img)   # GRAY로 카메라 표시
            #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #BGR -> HSV로 변환
            #cv2.imshow('camera', hsv) #HSV로 카메라 표시
            key = cv2.waitKey(1) & 0xFF
            hist = cv2.calcHist([img], [0], None, [256], [0,256])
            plt.plot(hist)
            print("hist.shape:", hist.shape) #히스토그램의 shape(256,1)
            print("hist.sum():", hist.sum(), "img.shape:", img.shape) #히스토그램 총 합계와 이미지의 크기
            plt.show()
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