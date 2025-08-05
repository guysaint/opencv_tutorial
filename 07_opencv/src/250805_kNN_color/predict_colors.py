import cv2
import numpy as np
import pickle

# 색상 라벨 이름
label_names = {
    1: 'Red',
    2: 'Blue',
    3: 'Green',
    4: 'Yellow',
    5: 'Black',
    6: 'White',
    7: 'Gray'
}

# 저장된 모델 로드
with open("knn_model.pkl", "rb") as f:
    knn = pickle.load(f)

# ROI 정보
roi_size = 100

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    cx, cy = w//2, h//2
    x1, y1 = cx - roi_size//2, cy - roi_size//2
    x2, y2 = cx + roi_size//2, cy + roi_size//2

    # ROI 추출
    roi = frame[y1:y2, x1:x2]
    mean_color = np.mean(roi.reshape(-1, 3), axis=0)
    r, g, b = mean_color[::-1]  # BGR → RGB

    # 정규화 후 예측
    input_rgb = np.array([[r/255, g/255, b/255]])
    pred = knn.predict(input_rgb)[0]
    proba = knn.predict_proba(input_rgb)[0]

    # 화면 출력
    label_name = label_names.get(pred, "Unknown")
    confidence = proba[pred - 1] * 100  # 라벨이 1~7이므로 인덱스 보정 필요

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f'{label_name} ({confidence:.1f}%)', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Color Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 종료
        break

cap.release()
cv2.destroyAllWindows()
