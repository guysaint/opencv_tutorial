import cv2
import numpy as np
import os

# ORB 추출기 설정
orb = cv2.ORB_create(nfeatures=500)  # 최대 500개의 특징점 추출

# BOW 훈련 데이터 생성
bow_trainer = cv2.BOWKMeansTrainer(100)  # 100개의 단어로 특징을 구분

# 자동차 이미지 경로
image_folder = '../data/train/traffic_light'  # 예시 폴더 경로
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

features = []
labels = []

# 이미지에서 특징 추출 및 BOW 훈련
for image_file in image_files:
    img_path = os.path.join(image_folder, image_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # ORB 특징점 추출
    keypoints, descriptors = orb.detectAndCompute(img, None)
    
    # 디스크립터의 타입을 np.float32로 변환
    if descriptors is not None:
        descriptors = np.float32(descriptors)  # 데이터 타입을 float32로 변경
        bow_trainer.add(descriptors)
    
    # 라벨 추가 (예시: 'car'로 설정)
    labels.append('car')

# BOW 코드북 생성 (k-means 클러스터링)
dictionary = bow_trainer.cluster()

# BOW 특징 벡터 추출기 설정
bow_extractor = cv2.BOWImgDescriptorExtractor(orb, cv2.BFMatcher(cv2.NORM_HAMMING))
bow_extractor.setVocabulary(dictionary)

# 훈련 데이터 준비
bow_features = []

for image_file in image_files:
    img_path = os.path.join(image_folder, image_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    keypoints = orb.detect(img, None)  # 특징점 검출
    
    # BOW 특징 벡터 생성 (특징점 추출 후, BOW 특징 벡터를 생성)
    bow_descriptor = bow_extractor.compute(img, keypoints)
    
    if bow_descriptor is not None:
        bow_features.append(bow_descriptor.flatten())

# 라벨 인코딩 (예: 모든 이미지가 'car'인 경우)
labels = ['car'] * len(bow_features)

# SVM 모델 학습
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 라벨 인코딩
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# 훈련 데이터와 테스트 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(bow_features, encoded_labels, test_size=0.2, random_state=42)

# SVM 모델 학습
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# 모델 평가
accuracy = svm.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# 모델 저장
import joblib
joblib.dump(svm, 'models/svm_car_model.pkl')