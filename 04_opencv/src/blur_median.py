import cv2
import numpy as np

img = cv2.imread('../img/salt_pepper_noise.jpg')
# 미디언 블러 적용
blur = cv2.medianBlur(img, 5)

merged = np.hstack( (img, blur))
cv2.imshow('gaussian blur', merged)
cv2.waitKey()
cv2.destroyAllWindows()
