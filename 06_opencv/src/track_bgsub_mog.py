import numpy as np, cv2

cap = cv2.VideoCapture('../img/walking.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)
# cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)
# history: 과거 프레임의 갯수, 배경을 학습하는데 얼마나 많은 프레임을 기억할지
# varThreshold: 필셀이 객체인 배경인지 구분하는 기준값
fgbg = cv2.createBackgroundSubtractorMOG2(50, 100, detectShadows=False)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame', frame)
    cv2.imshow('bgsub', fgmask)
    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()
