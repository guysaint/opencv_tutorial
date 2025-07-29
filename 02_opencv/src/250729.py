import cv2
import numpy as np

# 이미지 불러오기
img = cv2.imread('../img/test.jpg')
img = cv2.resize(img, (720, 872))
img2 = cv2.imread('../img/couple.png')

# 마스크 반지름
mask_radius = 70
mouse_x, mouse_y = -1, -1

# 마우스 이벤트 콜백 함수
def mouse_move(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

# 윈도우 이름
cv2.namedWindow('mask_follow')
cv2.setMouseCallback('mask_follow', mouse_move)

while True:
    # 원본 이미지 복사
    img_copy = img.copy()

    # 마우스가 유효한 위치일 때 마스킹 적용
    if mouse_x != -1 and mouse_y != -1:
        # 전체 검정 마스크 만들기
        mask = np.zeros_like(img_copy, dtype=np.uint8)

        # 마우스 위치에 하얀 원 그리기 (마스킹 영역)
        cv2.circle(mask, (mouse_x, mouse_y), mask_radius, (255, 255, 255), -1)

        # 마스크를 이용해 이미지 일부분만 보이도록 처리
        masked = cv2.bitwise_and(img_copy, mask)
        img_copy = masked

    # 출력
    cv2.imshow('mask_follow', img_copy)
    cv2.imshow('couple',img2)
    
    # ESC 키 누르면 종료
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
