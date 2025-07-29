import cv2
import numpy as np

#기본 값
img = cv2.imread('../img/like_lenna.png' )

cv2.imshow('default',img)
cv2.waitKey(0)
cv2.destroyAllWindows()