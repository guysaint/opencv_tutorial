import cv2
import matplotlib.pyplot as plt
import pyzbar.pyzbar as pyzbar

# 이미지 불러오기
#img = cv2.imread('../img/frame.png')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cap = cv2.VideoCapture(1)

# 이미지 캡처 조건 추가
while (cap.isOpened()):
    ret, img = cap.read()

    if not ret:
        continue

    #img = cv2.imread('../img/frame.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    decoded = pyzbar.decode(gray)

    for d in decoded:
        x, y, w, h = d.rect
        barcode_data = d.data.decode('utf-8')
        barcode_type = d.type
        #cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        #print(d.data.decode('utf-8'))
        #print(d.type)
       
        text = '%s (%s)'%(barcode_data, barcode_type)

        # 사각형 그리기
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.rectangle(img, (d.rect[0], d.rect[1]), (d.rect[0] + d.rect[2], d.rect[1] + d.rect[3]), (0, 255, 0), 20)
        cv2.putText(img,text,(d.rect[0], d.rect[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('camera',img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#plt.imshow(gray, cmap = 'gray') # cmap = colormap
#plt.show()
#cv2.imshow('qr_code', img)

'''
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
'''

#cv2.waitKey(0)
#cv2.destroyAllWindows()
