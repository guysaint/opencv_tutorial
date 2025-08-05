'''
 250805 오전 실습
 - 시각적 결과: 원본 이미지
 - 색상 팔레트: 추출된 3가지 대표 색상
 - 분포 차트: 각 색상이 차지하는 비율
 - 상세 분석: BGR 값과 픽셀 수/비율 정보

'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

K = 16 # 군집화 갯수
img = cv2.imread('../img/load_line.jpg')
img = cv2.resize(img, (600,396))
data = img.reshape((-1,3)).astype(np.float32)



# 데이터 평균을 구할 때 소수점 이하값을 가질 수 있으므로 변환
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 평균 클러스터링 적용
ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 중심값을 정수형으로 변환

center = np.uint8(center)
print(center)

# 각 레이블에 해당하는 중심값으로 픽셀 값 선택
res = center[label.flatten()]

# 원본 영상의 형태로 변환
res = res.reshape((img.shape))

# 팔레트 시각화용 이미지 만들기
palette = np.zeros((100, 300, 3), dtype=np.uint8) # 팔레트를 그리기 위한 공간 확보
width = 100
for i, color in enumerate(center):
    palette[:, i*width:(i+1)*width] = color

# 각 라벨(=클러스터 인덱스)의 갯수 세기
counts = np.bincount(label.flatten())

# 비율 계산
ratios = counts / counts.sum()

# 색상(BGR -> RGB로 변경해야 matplotlib에서 제대로 보임)
colors_rgb = [center[i][::-1]/255 for i in range(K)] # 0~1로 정규화한 RGB

# 원형 차트 그리기
plt.figure(figsize=(6,6))
plt.pie(ratios, labels=[f'#{i}' for i in range(K)], colors=colors_rgb, autopct='%1.1f%%')
plt.title('Color Distribution (K-Means)')
plt.show()

# 결과 출력
merged = np.hstack((img, res))
cv2.imshow('Kmeans_color', merged)
cv2.imshow('Palette', palette)
cv2.waitKey(0)
cv2.destroyAllWindows()