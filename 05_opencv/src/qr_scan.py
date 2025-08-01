import cv2
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar


img = cv2.imread('../img/frame.png')

plt.imshow(img)
plt.show()
#cv2.imshow('qr_code', img)
cv2.waitKey(0)
cv2.destroyAllWindows()