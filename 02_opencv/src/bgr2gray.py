import cv2
import numpy as np

img = cv2.imread('../img/like_lenna.png' )

#계산하는 방법
img2 = img.astype(np.uint16)
b,g,r = cv2.split(img2) #채널별로 분류
gray1 = ((b + g + r) /3).astype(np.uint8) #평균값을 연산후 dtype변경

#함수 사용하는 방법
gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #BGR을 그레이 스케일로 변경
gray3 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)


cv2.imshow('original',img)
cv2.imshow('gray1', gray1)
cv2.imshow('gray2', gray2)
cv2.imshow('gray3', gray3)

cv2.waitKey(0)
cv2.destroyAllWindows()