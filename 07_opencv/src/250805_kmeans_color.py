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
from io import BytesIO
from PIL import Image

K = 6 # 군집화 갯수
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
box_width = 100
palette = np.zeros((100, box_width * K, 3), dtype=np.uint8)

for i, color in enumerate(center):
    palette[:, i * box_width : (i + 1) * box_width] = color

# 각 라벨(=클러스터 인덱스)의 갯수 세기
counts = np.bincount(label.flatten())

# 비율 계산
ratios = counts / counts.sum()

# 색상(BGR -> RGB로 변경해야 matplotlib에서 제대로 보임)
colors_rgb = [center[i][::-1]/255 for i in range(K)] # 0~1로 정규화한 RGB

# 막대 그래프 -> 이미지 변환
fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
bars = ax.bar([f'#{i}' for i in range(K)], ratios, color=colors_rgb)

ax.set_title('Color Ratio (Bar)', fontsize=14)
ax.set_xlabel('Cluster Index', fontsize=10)
ax.set_ylabel('Ratio', fontsize=10)
ax.set_ylim(0, max(ratios)*1.2)  # 최대값보다 살짝 여유

# y축 눈금을 0.0, 0.1, 0.2, ... 식으로 자동 설정
ax.set_yticks(np.linspace(0, 1, 11))  # 0~1을 0.1 간격으로

# 눈금 스타일
ax.tick_params(axis='both', labelsize=9)

# 그래프 저장 → OpenCV 이미지로 변환
plt.tight_layout()
buf = BytesIO()
plt.savefig(buf, format='png')
plt.close(fig)
buf.seek(0)

bar_chart_pil = Image.open(buf).convert('RGB')
bar_chart_np = np.array(bar_chart_pil)
bar_chart_cv = cv2.cvtColor(bar_chart_np, cv2.COLOR_RGB2BGR)

# [3] 이미지 결합
merged_top = np.hstack((img, res))
h_top, w_top = merged_top.shape[:2]

# 팔레트와 바 차트의 높이를 동일하게 맞춤
bar_chart_resized = cv2.resize(bar_chart_cv, (w_top // 2, 300))
palette_resized = cv2.resize(palette, (w_top // 2, 300))
bottom_combined = np.hstack((palette_resized, bar_chart_resized))

# 최종 이미지 결합 (세로 방향)
final_result = np.vstack((merged_top, bottom_combined))

# 상세 분석 출력
print(f"\n[ 상세 분석 결과 (K={K}) ]")
print(f"클러스터\tB\tG\tR\t픽셀 수\t\t비율(%)")
for i in range(K):
    b, g, r = center[i]
    count = counts[i]
    ratio = ratios[i] * 100
    print(f"#{i}\t\t{b}\t{g}\t{r}\t{count:,}\t\t{ratio:.1f}%")



# 결과 출력
cv2.imshow('KMeans with Palette + Bar Chart', final_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
