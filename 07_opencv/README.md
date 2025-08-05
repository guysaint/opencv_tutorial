# 07_opencv

---

### 머신러닝 - K-평균 클러스터링

- 클러스터: **비슷한 특성을 가진 데이터끼리의 묶음.**
    - 비슷한 특성 → 가까운 위치
- 클러스터링: **어떤 데이터들이 주어졌을때, 그 데이터들을 클러스터로 그루핑 시켜주는 것.**
  
    
    ![클러스터끼리 그루핑 됨](./result_img/cluster.png)
    
    클러스터끼리 그루핑 됨
    
    ---
    

코드: [k-means_random.py](./src/k-means_random.py)

- ret,label,center=cv2.kmeans(data,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
- data: 처리 대상 데이터
- 2: 원하는 묶음 갯수
- None: 초기 레이블 정보(일반적으로 None사용)
- criteria: 종료 조건
- 10: 다양한 초기 중심값으로 반복 시도 횟수
- cv2.KMEANS_RANDOM_CENTERS: 초기 중심 값 선택 방법
    
    ![k-means_random.py 결과](./result_img/k-means_random_result.png)
    
    k-means_random.py 결과
    

---

코드: [k-means_color.py](./src/k-means_color.py)

### 결과

![k-means_color.py](./result_img/k-means_color_result.png)

k-means_color.py 결과

![k-means_color_result2.png](./result_img/k-means_color_result2.png)

k-means_color.py 터미널 창 결과

---

코드: [mnist.py](./src/mnist.py) - mnist 모듈

코드: [k-means_handwritten.py](./src/k-means_handwritten.py)

- 미리 만들어둔 모듈(mnist.py)을 사용함.

### 결과
  
![k-means_handwritten.py 결과](./result_img/k-means_handwritten_result.png) 
---
### 오전 실습

- 실습 차선 색상 분류
    - 시각적 결과: 원본 이미지
    - 색상 팔레트: 추출된 3가지 대표 색상
    - 분포 차트: 각 색상이 차지하는 비율
    - 상세 분석: BGR 값과 픽셀 수/비율 정보

코드: [250805_kmeans_color.py](./src/250805_kmeans_color.py)

### 결과

![250805_kmeans_color.py 결과](./result_img/250805_k-means_color_result.png)

250805_kmeans_color.py 결과

![터미널 창 결과(상세 분석)](./result_img/250805_k-means_color_terminal_result.png)

터미널 창 결과(상세 분석)

---
