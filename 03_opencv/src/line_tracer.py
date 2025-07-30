import cv2


cap = cv2.VideoCapture(1)             # 1번 카메라 장치 연결 ---①
if cap.isOpened():                      # 캡쳐 객체 연결 확인
    while True:
        ret, img = cap.read()
                   # 다음 프레임 읽기
        if ret:
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #BGR -> GRAY 변환
            # cv2.imshow('camera', gray)   # GRAY로 카메라 표시
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #BGR -> HSV로 변환
            cv2.imshow('camera', hsv) #HSV로 카메라 표시
            if cv2.waitKey(1)  & 0xFF == ord('q'):    # 1ms 동안 키 입력 대기 ---②
                break                   # 아무 키라도 입력이 있으면 중지
        else:
            print('no frame')
            break
else:
    print("can't open camera.")

cap.release()             # 자원 반납
cv2.destroyAllWindows()