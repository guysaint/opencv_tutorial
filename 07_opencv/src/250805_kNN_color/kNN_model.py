import csv
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle
from sklearn.neighbors import KNeighborsClassifier


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

# K-NN 함수
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(X_train, y_train, x_test, k=3):
    distances = [euclidean_distance(x_test, x) for x in X_train]
    sorted_indices = np.argsort(distances)[:k]
    nearest_labels = y_train[sorted_indices]
    prediction = Counter(nearest_labels).most_common(1)[0][0]
    return prediction
print("테스트 샘플 예측 결과:", knn_predict(X_train, y_train, X_test[0], k=3))
print("실제 라벨:", y_test[0])

# 예측 & 정확도 측정
correct = 0
for i in range(len(X_test)):
    pred = knn_predict(X_train, y_train, X_test[i], k=3)
    if pred == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)
print(f"[정확도] K=3일 때 정확도: {accuracy:.2%}")

# 여러 k값 실험
for k in [1, 3, 5, 7, 9]:
    correct = 0
    for i in range(len(X_test)):
        pred = knn_predict(X_train, y_train, X_test[i], k=k)
        if pred == y_test[i]:
            correct += 1
    acc = correct / len(X_test)
    print(f"K={k}: 정확도={acc:.2%}")

# 모델 학습
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 모델 저장
import pickle
with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)
print("모델이 knn_model.pkl로 저장되었습니다.")
