import cv2
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar

img = cv2.imread('../img/frame.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#plt.imshow(gray, cmap = 'gray') # cmap = colormap
#plt.show()
#cv2.imshow('qr_code', img)

# 디코딩
decoded = pyzbar.decode(gray)
print(decoded)

for d in decoded:
    print(d.data.decode('utf-8'))
    print(d.type)
    # 사각형 그리기
    cv2.rectangle(img, (d.rect[0], d.rect[1]), (d.rect[0] + d.rect[2], d.rect[1] + d.rect[3]), (0, 255, 0), 20)

plt.imshow(img)
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
