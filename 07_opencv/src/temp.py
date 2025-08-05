import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

K = 16
img = cv2.imread('../img/load_line.jpg')
img = cv2.resize(img, (600, 396))
data = img.reshape((-1, 3)).astype(np.float32)

# KMeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()].reshape(img.shape)

# 비율 및 색상 추출
counts = np.bincount(label.flatten())
ratios = counts / counts.sum()
colors_bgr = [center[i] for i in range(K)]
colors_rgb = [c[::-1]/255 for c in colors_bgr]  # for matplotlib

# [1] 색상 팔레트 만들기
box_width = 100
palette_img = np.zeros((100, box_width*K, 3), dtype=np.uint8)
for i, color in enumerate(colors_bgr):
    palette_img[:, i*box_width:(i+1)*box_width] = color

# [2] 막대그래프 → 이미지 변환
fig, ax = plt.subplots(figsize=(8, 4), dpi=100)  # 그래프 크기 더 키움
bars = ax.bar([f'#{i}' for i in range(K)], ratios, color=colors_rgb)

# 제목 및 세분화된 축 설정
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
palette_resized = cv2.resize(palette_img, (w_top // 2, 300))
bottom_combined = np.hstack((palette_resized, bar_chart_resized))

# 최종 이미지 결합 (세로 방향)
final_result = np.vstack((merged_top, bottom_combined))

# 결과 출력
cv2.imshow('KMeans with Palette + Bar Chart', final_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
