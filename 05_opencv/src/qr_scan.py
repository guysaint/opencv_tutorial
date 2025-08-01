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
    barcode_data = d.data.decode('utf-8')
    print(d.type)
    barcode_type = d.type

    text = '%s (%s)'%(barcode_data, barcode_type)

    # 사각형 그리기
    cv2.rectangle(img, (d.rect[0], d.rect[1]), (d.rect[0] + d.rect[2], d.rect[1] + d.rect[3]), (0, 255, 0), 20)
    cv2.putText(img,text,(d.rect[0], d.rect[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 255), 10, cv2.LINE_AA)

plt.imshow(img)
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
