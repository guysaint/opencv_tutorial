import csv
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# CSV 불러오기
data = []
labels = []

with open('color_dataset.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 헤더 건너뛰기
    for row in reader:
        r, g, b, h, s, v, label = row
        rgb = [int(r)/255.0, int(g)/255.0, int(b)/255.0]  # RGB 정규화
        data.append(rgb)
        labels.append(int(label))

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(f"[INFO] 총 샘플 수: {len(data)}")
print(f"[INFO] 학습용: {len(X_train)}개, 테스트용: {len(X_test)}개")