import cv2
import numpy as np
import matplotlib.pylab as plt

#이미지 그레이 스케일로 읽기 및 출력
img = cv2.imread('../img/like_lenna.png')
cv2.imshow('img',img)

#히스토그램 게산 및 그리기
channels = cv2.split(img)
colors = ('b', 'g', 'r')
for (ch, color) in zip (channels, colors):
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    plt.plot(hist, color=color)
plt.show()