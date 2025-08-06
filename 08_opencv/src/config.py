# config.py: 설정 변수 정의

# 이미지 카테고리 설정 (객체 인식할 클래스들)
CATEGORIES = ['traffic_light', 'stop_sign']  # 예시로 traffic_light와 stop_sign 카테고리 설정

# BOW (Bag of Words) 사전 크기 설정
DICTIONARY_SIZE = 50  # BOW 모델에서 사용할 단어(특징) 수

# 데이터 경로 설정
BASE_PATH = './data/'  # 데이터가 저장된 기본 경로
MODEL_PATH = './models/'  # 훈련된 모델이 저장될 경로

# 라벨 경로 설정
LABELS_PATH = './labels/'  # 라벨 파일들이 저장된 경로