import cv2
import numpy as np
import pickle

# 모델 불러오기
with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)

# 라벨 번호 → 색상 이름
label_names = {
    1: 'Red',
    2: 'Blue',
    3: 'Green',
    4: 'Yellow',
    5: 'Black',
    6: 'White',
    7: 'Gray'
}

# 웹캠 설정
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ROI (관심영역) 설정 - 화면 중앙 사각형
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    size = 50
    roi = frame[cy-size:cy+size, cx-size:cx+size]

    # ROI 평균 색상 계산 (RGB 평균)
    avg_color = cv2.mean(roi)[:3]
    r, g, b = map(int, avg_color)
    rgb_norm = np.array([[r / 255, g / 255, b / 255]])

    # 예측
    pred_label = model.predict(rgb_norm)[0]
    color_name = label_names.get(pred_label, "Unknown")

    # 시각화
    cv2.rectangle(frame, (cx-size, cy-size), (cx+size, cy+size), (0,255,0), 1)
    cv2.putText(frame, f'RGB: ({r},{g},{b})', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f'Predicted: {color_name}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    cv2.imshow("Color Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 종료
        break

cap.release()
cv2.destroyAllWindows()
