import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture(1) 

plt.ion() #실시간 플롯을 위해 인터렉티브 모드 on
           
if cap.isOpened():    
    while True:
        ret, img = cap.read()
        if ret:
            #BGR -> GRAY
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 스캔라인 ROI 설정(세로 10px 범위, 중앙 수평선 부근)
            roi_top = 0
            roi_bottom = 500
            roi = gray[roi_top:roi_bottom, :]

            # ROI를 절반으로 나눔
            h, w = roi.shape
            left_roi = roi[:, :w//2] #배경 영역
            right_roi = roi[:, w//2:] #선 추정 영역
            
            # 이진화 및 윤곽선 추출
            _, binary = cv2.threshold(roi, 80, 255, cv2.THRESH_BINARY_INV)  # 검정선 강조
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                #가장 큰 윤곽선 선택
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] !=0:
                    cx = int(M["m10"] / M["m00"])
                    # ROI는 gray[240:250, :] -> y=240~250 중간인 245로 설정
                    cv2.circle(img, (cx, 245), 5, (0, 0, 255), -1)
                    print(f"Contour center x: {cx}")
                # drawContours: ROI 내부 contour를 전체 이미지로 보정해서 시각화
            for cnt in contours:
                cnt_shifted = cnt + [0, roi_top]  # y 좌표 보정
                cv2.drawContours(img, [cnt_shifted], -1, (0, 255, 0), 1)
            '''
            #-------이 부분은 1차 이진화 계산
            # 이진화 및 중심 추출
            binary = (roi < 50).astype(np.uint8)
            black_count = np.sum(binary, axis=0)

            # → 검정 픽셀이 존재할 때만 중심 표시
            if np.sum(black_count) > 0:
                x_coords = np.arange(black_count.shape[0])
                center_x = int(np.sum(x_coords * black_count) / np.sum(black_count))
                cv2.circle(img, (center_x, 245), 5, (0, 0, 255), -1)

                # 5. 원본 영상에 중심 좌표 시각화
                cv2.circle(img, (center_x, 245), 5, (0, 0, 255), -1)
            '''
            
            # 화면 출력
            cv2.imshow('camera (with line center)', img)



            # 각각의 히스토그램 계산
            hist_left = cv2.calcHist([left_roi], [0], None, [256], [0, 256])
            hist_right = cv2.calcHist([right_roi], [0], None, [256], [0, 256])

            cv2.normalize(hist_left, hist_left, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_right, hist_right, 0, 1, cv2.NORM_MINMAX)

            #디버깅용 ROI 시각 표시
            cv2.line(img, (0, 240), (img.shape[1], 240), (255, 0, 0), 2)
            cv2.line(img, (0, 249), (img.shape[1], 249), (255, 0, 0), 2)
            #cv2.imshow('camera(gray)', gray)
            
            # 검정색(0~50) 구간의 총 빈도 비교
            black_left = np.sum(hist_left[:50])
            black_right = np.sum(hist_right[:50])

            print(f"Black Pixel (Left ROI): {black_left:.4f}")
            print(f"Black Pixel (Right ROI): {black_right:.4f}")
            black_pixels = np.sum((right_roi < 50) & (right_roi > 0))  # 0은 순수 배경 노이즈일 수 있음
            total_pixels = right_roi.size
            print(f"Black pixel ratio in Right ROI: {black_pixels / total_pixels:.4f}")



            #ROI에 검정선이 실제로 들어오는지 확인
            cv2.imshow('ROI view', roi)
            
            
            #그레이 히스토그램 계산
            #hist = cv2.calcHist([gray], [0], None, [256], [0,256])
            #cv2.normalize(hist, hist, 0, 100, cv2.NORM_MINMAX)
            #화면 출력
            #cv2.imshow('camera(gray)', gray)   # GRAY로 카메라 표시
            
            #Matplotlib 히스토그램
            plt.cla()
            plt.plot(hist_left, color='green', label='Left ROI (Background)')
            plt.plot(hist_right, color='black', label='Right ROI (Line)')
            plt.title('Gray Histogram(ROI-based)')
            plt.xlabel('Pixel Value (0~255)')
            plt.ylabel('Normalized Frequency')
            plt.xlim([0, 255])
            plt.legend()
            plt.pause(0.01)


            '''
            #히스토그램 시각화용 이미지 생성
            hist_img = np.full((100,256), 255, dtype=np.uint8)
            for x,y in enumerate(hist):
                cv2.line(hist_img, (x, 100), (x, 100-int(y)), 0)
            '''
            
            

            #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #BGR -> HSV로 변환
            #cv2.imshow('camera', hsv) #HSV로 카메라 표시
            key = cv2.waitKey(1) & 0xFF
                        
            if key == ord('q'):
                break
            if key == ord("s"):            # s키를 누르면
                cv2.imwrite('../img/photo.jpg', img)    #phto.jpg로 사진 저장

        else:
            print('no frame')
            break
else:
    print("can't open camera.")

cap.release()    #자원반납
cv2.destroyAllWindows()