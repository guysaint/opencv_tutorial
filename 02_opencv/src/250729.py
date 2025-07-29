import cv2
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image

# 이미지 불러오기
img = cv2.imread('../img/test.jpg')
img = cv2.resize(img, (720, 872))

couple = cv2.imread('../img/couple.png', cv2.IMREAD_UNCHANGED)
training = cv2.imread('../img/training.png', cv2.IMREAD_UNCHANGED)

# 오류 확인
if couple is None:
    raise FileNotFoundError("couple 이미지 경로 확인")
if training is None:
    raise FileNotFoundError("training 이미지 경로 확인")

# 크기 조절
scale1 = 0.15
scale2 = 0.15
couple = cv2.resize(couple, (0, 0), fx=scale1, fy=scale1)
training = cv2.resize(training, (0, 0), fx=scale2, fy=scale2)

# 숨길 위치 지정
targets = [
    {'img': couple, 'pos': (365, 757), 'name': 'couple'},
    {'img': training, 'pos': (182, 625), 'name': 'training'},  # 운동하는 사람!
]

# 배경에 두 인물 합성
combined = img.copy()

for target in targets:
    img_rgba = target['img']
    x, y = target['pos']
    h, w = img_rgba.shape[:2]

    b, g, r, a = cv2.split(img_rgba)
    rgb = cv2.merge((b, g, r))
    mask = a

    roi = combined[y:y+h, x:x+w]
    bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    fg = cv2.bitwise_and(rgb, rgb, mask=mask)
    combined[y:y+h, x:x+w] = cv2.add(bg, fg)

# 상태 변수
mouse_x, mouse_y = -1, -1
mask_radius = 70
found = False
wrong = False  # 틀린 클릭 여부

# 한글 폰트
font_path = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(font_path, 32)

# 클릭 여부 확인 함수
def check_click(x, y):
    for target in targets:
        tx, ty = target['pos']
        th, tw = target['img'].shape[:2]
        if tx <= x <= tx + tw and ty <= y <= ty + th:
            return target['name']
    return None

# 마우스 콜백
def mouse_event(event, x, y, flags, param):
    global mouse_x, mouse_y, found, wrong
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        result = check_click(x, y)
        if result == 'training':  # 정답!
            found = True
        else:
            wrong = True  # 틀림 표시

cv2.namedWindow('Sniper Game')
cv2.setMouseCallback('Sniper Game', mouse_event)

while True:
    black_overlay = np.zeros_like(combined)

    if mouse_x != -1 and mouse_y != -1:
        mask = np.zeros((combined.shape[0], combined.shape[1]), dtype=np.uint8)
        cv2.circle(mask, (mouse_x, mouse_y), mask_radius, 255, -1)
        visible = cv2.bitwise_and(combined, combined, mask=mask)
        dark = cv2.bitwise_and(black_overlay, black_overlay, mask=cv2.bitwise_not(mask))
        display = cv2.add(visible, dark)
    else:
        display = black_overlay

    # PIL로 한글 텍스트 처리
    pil_img = Image.fromarray(display)
    draw = ImageDraw.Draw(pil_img)

    # 안내 문구
    instruction = '운동하는 사람을 찾으세요'
    text_size = draw.textbbox((0, 0), instruction, font=font)
    text_x = (display.shape[1] - (text_size[2] - text_size[0])) // 2
    text_y = 20
    draw.text((text_x, text_y), instruction, font=font, fill=(255, 255, 255))

    # 클릭 결과 표시
    if found:
        result_text = '찾았습니다!'
        color = (0, 255, 0)
    elif wrong:
        result_text = '틀렸습니다. 다시 찾아보세요'
        color = (0, 0, 255)
    else:
        result_text = None

    if result_text:
        result_size = draw.textbbox((0, 0), result_text, font=font)
        result_x = (display.shape[1] - (result_size[2] - result_size[0])) // 2
        result_y = (display.shape[0] - (result_size[3] - result_size[1])) // 2
        draw.text((result_x, result_y), result_text, font=font, fill=color)

    # OpenCV로 다시 변환
    display = np.array(pil_img)
    cv2.imshow('Sniper Game', display)

    if found:
        cv2.waitKey(2000)
        break

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
