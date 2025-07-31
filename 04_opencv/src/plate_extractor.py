import cv2
import numpy as np
import datetime
import os


img_paths = {'car_01': '../img/car_01.jpg',
             'car_02': '../img/car_02.jpg',
             'car_03': '../img/car_03.jpg',
             'car_04': '../img/car_04.jpg',
             'car_05': '../img/car_05.jpg',}

imgs = {}

for name, path in img_paths.items():
    img = cv2.imread(path)
    if img is None:
        print(f"{name} 로드 실패")
        continue
    imgs[name] = {
        'img': img,
        'draw': img.copy(),
        'pts': np.zeros((4, 2), dtype=np.float32),
        'pts_cnt': 0,
        'win_name': f'Plate_Scanning_{name}',
        'scanned': None
    }
current_name = None  # 현재 처리 중인 이미지 이름


def onMouse(event, x, y, flags, param):
    global current_name
    img_data = imgs[name]

    if  img_data['pts_cnt']>=4:                     # 이미 4개 클릭한 상태라면 무시
        return
    if event == cv2.EVENT_LBUTTONDOWN:              # 마우스 왼쪽 클릭시
        
        cv2.circle(img_data['draw'], (x,y), 10, (0,255,0), -1) # 좌표에 초록색 동그라미 표시
        cv2.imshow(img_data['win_name'], img_data['draw'])

        img_data['pts'][img_data['pts_cnt']] = [x,y]
        img_data['pts_cnt'] += 1
        

        if img_data['pts_cnt'] == 4:            # 4개의 좌표가 수집 되면
            # 좌표 4개 중 상하 좌우 찾기
            pts = img_data['pts']
            sm = pts.sum(axis=1)                # 4쌍의 좌표 각각 x+y 계산
            diff = np.diff(pts, axis = 1)       # 4쌍의 좌표 각각 x-y 계산

            topLeft = pts[np.argmin(sm)]        # 좌상단 좌표
            bottomRight = pts[np.argmax(sm)]    # 우하단 좌표               
            topRight = pts[np.argmin(diff)]     # 우상단 좌표
            bottomLeft = pts[np.argmax(diff)]   # 좌하단 좌표

            # 변환 전 4개 좌표
            pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

            # 변환 후 영상에 사용할 서류의 폭과 높이 계산
            w1 = abs(bottomRight[0]) - bottomLeft[0]    # 상단 좌우 좌표간의 거리
            w2 = abs(topRight[0] - topLeft[0])          # 하단 좌우 좌표간의 거리
            h1 = abs(topRight[1] - bottomRight[1])      # 우측 상하 좌표간의 거리
            h2 = abs(topLeft[1] - bottomLeft[1])        # 좌측 상하 좌표간의 거리
            width = int(max([w1, w2]))                  # 두 좌우 거리간의 최대값이 서류의 폭
            height = int(max([h1, h2]))                 # 두 상하 거리간의 최대값이 서류의 높이
            
            # 변환 후 4개 좌표
            pts2 = np.float32([[0,0], [width-1, 0], [width-1, height-1], [0, height-1]])

            # 변환 행렬 계산
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)

            #원근 변환 적용
            result = cv2.warpPerspective(imgs[name]['img'], mtrx, (width, height))
            img_data['scanned'] = result
            # 결과 이미지 보여주기
            scanned_win = f'scanned_{current_name}'
            cv2.imshow(scanned_win, result)
            
            # 이미지 저장
            save_dir = "../extracted_plates"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"../extracted_plates/plate_{timestamp}.jpg"
            save_path = f'../extracted_plates/{current_name}_scanned.jpg'
            cv2.imwrite(filename, result)
            print(f'{save_path} 저장 완료')
        
    
# 이미지 하나씩 처리
for name in imgs:
    current_name = name
    img_data = imgs[name] 
    
    # 초기화
    img_data['draw'] = img_data['img'].copy()
    img_data['pts_cnt'] = 0
    img_data['pts'] = np.zeros((4,2), dtype=np.float32)
    
    cv2.imshow(img_data['win_name'], img_data['draw'])
    cv2.setMouseCallback(img_data['win_name'], onMouse)

    print(f'{name} 변호판 꼭지점 4곳을 클릭 후 아무 키나 누르세요...')

    cv2.waitKey(0)
    cv2.destroyWindow(img_data['win_name'])
    if img_data['scanned'] is not None:
        cv2.destroyWindow(f'scanned_{name}')

cv2.destroyAllWindows()
                 