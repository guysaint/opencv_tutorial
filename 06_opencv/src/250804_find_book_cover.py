import cv2
import numpy as np
import os
import glob
import time

# 초기 설정
img1 = None
win_name = 'Camera Matching'
MIN_MATCH = 10

# ORB 및 FLANN 매처 설정
detector = cv2.ORB_create(5000)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# 이미지 경로 설정
search_dir = '../img/books'
img_paths = glob.glob(os.path.join(search_dir, '*.jpg'))

# 카메라 연결
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    res = frame.copy()

    if img1 is None:
        # 고정된 ROI 박스 그리기
        h_frame, w_frame = frame.shape[:2]
        roi_w, roi_h = 250, 350
        x = (w_frame - roi_w) // 2
        y = (h_frame - roi_h) // 2
        cv2.rectangle(res, (x, y), (x + roi_w, y + roi_h), (255, 255, 255), 2)

    cv2.imshow(win_name, res)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC 종료
        break

    elif key == ord(' '):
        # ROI 추출 및 특징점 추출
        img1 = frame[y:y+roi_h, x:x+roi_w]
        kp1, des1 = detector.detectAndCompute(img1, None)

        if des1 is None:
            print("특징점 없음")
            img1 = None
            continue

        # 매칭 시작 시간 기록
        start_time = time.time()

        best_score = 0
        best_img = None
        best_path = None
        best_matches = None
        best_kp = None

        # 폴더 이미지들과 매칭
        for path in img_paths:
            img2 = cv2.imread(path)
            if img2 is None:
                continue

            kp2, des2 = detector.detectAndCompute(img2, None)
            if des2 is None:
                continue

            matches = matcher.knnMatch(des1, des2, k=2)

            good = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            if len(good) > best_score:
                best_score = len(good)
                best_img = img2
                best_path = path
                best_matches = good
                best_kp = kp2

        elapsed_time = time.time() - start_time  # 검색 완료 시간

        # 결과 출력
        if best_img is not None:
            roi_resized = cv2.resize(img1, (300, 400))
            match_resized = cv2.resize(best_img, (300, 400))
            combined = np.hstack((roi_resized, match_resized))

            if des1 is not None and best_kp is not None:
                match_percent = (best_score / len(des1)) * 100
            else:
                match_percent = 0

            text = f"Matching: {match_percent:.2f}%  |  Time: {elapsed_time:.2f}s"
            cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Matching Result', combined)
            print(f'Best match: {best_path}, 매칭률: {match_percent:.2f}%, 시간: {elapsed_time:.2f}s')

        # 다음 검색을 위해 ROI 초기화
        img1 = None

cv2.destroyAllWindows()
