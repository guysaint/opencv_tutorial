import cv2
import numpy as np
import csv

# 전역 변수 설정
samples = []  # 수집한 색상 샘플들 저장

# 색상 라벨 이름 (1~7 키에 매핑)
label_names = {
    1: 'Red',
    2: 'Blue',
    3: 'Green',
    4: 'Yellow',
    5: 'Black',
    6: 'White',
    7: 'Gray'
}

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Color Collector")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ROI 시각화
    h, w, _ = frame.shape
    cx, cy = w//2, h//2
    size = 20
    cv2.rectangle(frame, (cx-size, cy-size), (cx+size, cy+size), (0,255,0), 1)

    cv2.putText(frame, "1~7: Save Color | S: Save | ESC: Exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Color Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key in map(ord, '1234567'):
        label = int(chr(key))
        b, g, r = frame[cy, cx]
        hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        samples.append([r, g, b, hsv[0], hsv[1], hsv[2], label])
        print(f"[✔] 저장됨: RGB=({r},{g},{b}), HSV={hsv}, 라벨={label_names[label]}")

    elif key == ord('s'):  # S 누르면 저장
        with open("color_dataset.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['R', 'G', 'B', 'H', 'S', 'V', 'label'])
            writer.writerows(samples)
        print(f"💾 총 {len(samples)}개의 샘플이 저장되었습니다.")

    elif key == 27:  # ESC 종료
        break

cap.release()
cv2.destroyAllWindows()
