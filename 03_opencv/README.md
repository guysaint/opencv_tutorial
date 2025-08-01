# 웹캠을 이용한 라인트레이싱 기능 구현 코드

USB 카메라를 통해 실시간으로 종이에 그려진 검정색 선을 인식하고  
이를 기반으로 선 중심을 추적하여 향후 라인트레이서 로봇 등에 적용하기 위한 실습 코드입니다.

---

## 📌 사용 기술

- Python 3
- OpenCV (cv2)
- Matplotlib (실시간 히스토그램 디버깅용)
- NumPy

---

## 🛠 진행 단계 요약

### 1. USB 카메라 연결 및 실시간 영상 출력

```python
cap = cv2.VideoCapture(1)
ret, img = cap.read()
cv2.imshow('camera', img)
```

---

### 2. BGR → GRAY 변환 + 전체 히스토그램 출력

- `cv2.cvtColor()`로 GRAY 영상 생성  
- `cv2.calcHist()` + `matplotlib`으로 픽셀값 분포 시각화  
- 밝기 분포를 기반으로 선/배경 분석

---

### 3. ROI(관심영역) 설정 및 좌우 분할 비교

- 영상 중앙 또는 특정 구간을 `gray[y1:y2, :]`로 잘라 ROI 설정  
- ROI를 좌/우로 나누고 각각의 히스토그램 비교  
- 검정선 존재 여부를 `픽셀값 < 50` 기준으로 판단

---

### 4. 선 중심 추출: 픽셀 가중 평균(center of mass) 방식

- ROI 이진화 후 각 열의 검정 픽셀 수 계산  
- `np.sum(x * weight) / np.sum(weight)` 공식으로 중심 추정  
- `cv2.circle()`로 중심 위치 시각화

---

### 5. 선 중심 추출: `cv2.findContours()` + `cv2.moments()` 방식

- ROI 이진화 후 윤곽선 추출 (`cv2.findContours`)  
- 가장 큰 윤곽선에 대해 `cv2.moments()`로 무게중심 계산  
- 중심좌표를 원본 이미지에 표시  
- 선 꺾임, 흔들림 대응을 위해 ROI를 더 넓게 설정

---

### 6. 인식률 향상 방법 적용

- 이진화 임계값 증가: 흐릿한 선도 인식 가능하도록 `threshold=50 → 80`  
- `cv2.GaussianBlur()` 적용: 선 경계 부드럽게  
- `cv2.dilate()` 적용: 얇은 선을 굵게 확장

---

### 7. 윤곽선 시각화

- `cv2.drawContours()`로 ROI 내 윤곽선을 초록색으로 표시  
- ROI 내부 좌표를 원본 영상 좌표에 맞게 `cnt + [0, roi_top]`으로 보정

---

## ✅ 결과

- 실시간으로 검정 선 인식 성공  
- 흐리거나 굽은 선에도 안정적인 중심 추적 가능  
- 이후 라인트레이서 모터 제어 등으로 확장 가능한 기반 확보
