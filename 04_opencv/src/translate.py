import cv2
import numpy as np


img = cv2.imread('../img/fish.jpg')
row, cols = img.shape[0:2] #가져오는 이미지 크기

dx, dy = 100, 50 #이동할 픽셀거리

#변환 행렬
mtrx = np.float32([[1,0,dx],
                   [0,1,dy]])
#단순 이동
dst = cv2.warpAffine(img, mtrx, (cols+dx, row+dy))

# 탈락된 외각 픽셀을 파랑색으로 보정
dst2 = cv2.warpAffine(img, mtrx, (cols+dx, row+dy), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255,0,0))

#탈락된 외각 픽셀을 원본으로 반사시켜서 보정
dst3 = cv2.warpAffine(img, mtrx, (cols+dx, row+dy),None, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

cv2.imshow('original', img)
cv2.imshow('trans',dst)
cv2.imshow('BORDER_CONSTATNT', dst2)
cv2.imshow('BORDER_REFLECT', dst3)

cv2.waitKey(0)
cv2.destroyAllWindows()