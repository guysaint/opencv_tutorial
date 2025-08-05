import cv2
import numpy as np
import csv
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# ==== 학습된 모델 로드 또는 생성 ====

try:
    model = joblib.load("knn_color_model.pkl")
    label_names = joblib.load("label_names.pkl")
    print("저장된 모델을 불러왔습니다.")
except:
    print("모델 파일을 찾을 수 없습니다. color_dataset.csv로부터 새로 학습합니다.")
    # CSV 데이터 불러오기
    data = []
    labels = []
    label_names = {}

    with open("color_dataset.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rgb = [int(row['R']), int(row['G']), int(row['B'])]
            data.append([v/255 for v in rgb])  # 정규화
            labels.append(int(row['label']))

    # 라벨 이름 매핑
    label_names = {
        1: 'Red', 2: 'Blue', 3: 'Green',
        4: 'Yellow', 5: 'Black', 6: 'White', 7: 'Gray'
    }

    # 학습
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)
    print("모델 새로 학습 완료.")

    # 저장
    joblib.dump(model, "knn_color_model.pkl")
    joblib.dump(label_names, "label_names.pkl")
    print("모델 저장 완료.")

# ==== 실시간 예측 시작 ====

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

roi_size = 100

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2

    # ROI 추출
    roi = frame[cy-roi_size//2:cy+roi_size//2, cx-roi_size//2:cx+roi_size//2]
    avg_color = cv2.mean(roi)[:3]  # BGR
    avg_rgb = [avg_color[2], avg_color[1], avg_color[0]]  # RGB
    norm_rgb = [v / 255 for v in avg_rgb]

    # 예측
    prediction = model.predict([norm_rgb])[0]
    proba = model.predict_proba([norm_rgb])[0][prediction-1] * 100

    # 시각화
    label_text = f"{label_names[prediction]} ({proba:.1f}%)"
    cv2.rectangle(frame, (cx-roi_size//2, cy-roi_size//2), (cx+roi_size//2, cy+roi_size//2), (0, 255, 0), 2)
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Color Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
