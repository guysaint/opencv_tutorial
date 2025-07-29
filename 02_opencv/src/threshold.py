import cv2
import numpy as np
import matplotlib.pylab as plt


img = cv2.imread('../img/like_lenna.png', cv2.IMREAD_GRAYSCALE) # 이미지를 그레이스케일로 불러옴

# Numpy api로 바이너리 이미지
thresh_np = np.zeros_like(img) # 원본과 동일한 크기의 0으로 채워진 이미지
thresh_np[ img > 127 ] = 255 #127보다 큰 값만 255로 변경

# OpenCV API로 바이너리 이미지 만들기
ret, thresh_cv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
print(ret) #127.0, 바이너리 이미지에 사용된 문턱값 반환

# matplot으로 출력하기
imgs = {'Original':img, 'Numpy API':thresh_np, 'cv2.threshold': thresh_cv}
for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(1, 3, i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.show()
