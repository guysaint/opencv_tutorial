import cv2

img = cv2.imread('../img/pistol.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.resize(gray, (16,16))

# 영상의 평균값 구하기
avg = gray.mean()

# 영상의 평균값을 기준으르로 0과 1로 변환
bin = 1* (gray > avg)
print(bin)

# 2진수 문자열을 16진수 문자열로 변환
dhash=[]
for row in bin.tolist():
    s = ''.join([str(i) for i in row])
    dhash.append('%02x'%(int(s,2)))
dhash = ''.join(dhash)
print(dhash)


cv2.namedWindow('pistol', cv2.WINDOW_GUI_NORMAL)
cv2.imshow('pistol', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
