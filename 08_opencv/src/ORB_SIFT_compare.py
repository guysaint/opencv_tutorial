import cv2
import matplotlib.pyplot as plt

# 이미지 경로
img_path = '../data/train/traffic_light/512_ND_0000_CF_001.jpg'

# 이미지 로드
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 이미지가 제대로 로드되었는지 확인
if img is None:
    print(f"이미지를 로드할 수 없습니다: {img_path}")
else:
    print(f"이미지 로드 성공: {img_path}")

# SIFT 특징점 추출
sift = cv2.SIFT_create()  # SIFT 추출기
keypoints_sift, descriptors_sift = sift.detectAndCompute(img, None)

# ORB 특징점 추출
orb = cv2.ORB_create(nfeatures=500)  # ORB 추출기
keypoints_orb, descriptors_orb = orb.detectAndCompute(img, None)

# 특징점 시각화
img_sift = cv2.drawKeypoints(img, keypoints_sift, None)
img_orb = cv2.drawKeypoints(img, keypoints_orb, None)

# 이미지 비교 출력
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('SIFT Features')
plt.imshow(img_sift)

plt.subplot(1, 2, 2)
plt.title('ORB Features')
plt.imshow(img_orb)

plt.show()