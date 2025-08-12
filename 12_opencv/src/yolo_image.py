from ultralytics import YOLO

#yolo 모델 설정
model = YOLO('yolo11s.pt')

#results = model('https://ultralytics.com/images/bus.jpg')

test_images = ['https://ultralytics.com/images/zidan.jpg''https://ultralytics.com/images/bus.jpg']

#results[0].show()
for img in test_images:
    results = model(img)
    print(results)
    print(f'검출된 객체 수: {len(results[0].boxes)}')
    
results[0].show()