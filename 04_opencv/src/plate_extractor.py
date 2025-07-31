import cv2
import numpy as np

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
    rows, cols = img.shape[:2]
    draw = img.copy()
    win_name = f'Plate_Scanning_{name}'

    imgs[name] = {
        'img': img,
        'draw': draw,
        'rows': rows,
        'cols': cols,
        'win_name': win_name
    }
pts_cnt = 0
pts = np.zeros((4,2), dtype = np.float32)


    


                 