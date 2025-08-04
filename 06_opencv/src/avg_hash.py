import cv2

img = cv2.imread('../img/pistol.jpg')


cv2.imshow('pistol', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
