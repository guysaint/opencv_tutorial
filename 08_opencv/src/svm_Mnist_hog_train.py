from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import cv2 
import numpy as np
import mnist
import time

# 기울어진 숫자를 바로 세우기 위한 함수 ---①
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*20*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(20, 20),flags=affine_flags)
    return img

# HOGDescriptor를 위한 파라미터 설정 및 생성---②
winSize = (20,20)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (5,5)
nbins = 9
hogDesc = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

# MNIST 이미지에서 학습용 이미지와 테스트용 이미지 가져오기
train_data, train_label = mnist.getTrain(reshape=False)
test_data, test_label = mnist.getTest(reshape=False)

# 학습 이미지 글씨 바로 세우기
deskewed = [list(map(deskew, row)) for row in train_data]

# 학습 이미지 HOG 계산
hogdata = [list(map(hogDesc.compute, row)) for row in deskewed]
train_data = np.float32(hogdata)
train_data = train_data.reshape(-1, train_data.shape[2])

# 그리드 서치를 위한 파라미터 설정
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3)  # 3겹 교차 검증

# 그리드 서치로 최적의 파라미터 찾기
grid_search.fit(train_data, train_label)
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# 최적의 C와 gamma 값으로 SVM 모델 훈련
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_RBF)
svm.setC(best_params['C'])
svm.setGamma(best_params['gamma'])

# SVM 학습
startT = time.time()
svm.train(train_data, cv2.ml.ROW_SAMPLE, train_label)
endT = time.time() - startT
print(f"SVM training complete. Time taken: {endT/60:.2f} minutes")

# 훈련된 모델 저장
svm.save('svm_mnist.xml')

# 테스트 이미지 글씨 바로 세우기 및 HOG 계산
deskewed = [list(map(deskew, row)) for row in test_data]
hogdata = [list(map(hogDesc.compute, row)) for row in deskewed]
test_data = np.float32(hogdata)
test_data = test_data.reshape(-1, test_data.shape[2])

# 테스트 데이터 예측
ret, result = svm.predict(test_data)



'''
if __name__ =='__main__':
    # MNIST 이미지에서 학습용 이미지와 테스트용 이미지 가져오기 ---③
    train_data, train_label  = mnist.getTrain(reshape=False)
    test_data, test_label = mnist.getTest(reshape=False)
    # 학습 이미지 글씨 바로 세우기 ---④
    deskewed = [list(map(deskew,row)) for row in train_data]
    # 학습 이미지 HOG 계산 ---⑤
    hogdata = [list(map(hogDesc.compute,row)) for row in deskewed]
    train_data = np.float32(hogdata)
    print('SVM training started...train data:', train_data.shape)
    # 학습용 HOG 데이타 재배열  ---⑥
    train_data = train_data.reshape(-1,train_data.shape[2])
    # SVM 알고리즘 객체 생성 및 훈련 ---⑦
    svm = cv2.ml.SVM_create()
    startT = time.time()
    
    # 커널 설정(RBF 커널 사용)
    svm.setKernel(cv2.ml.SVM_RBF)

    # C와 gamma 값 설정
    svm.setC(1.0)
    svm.setGamma(0.5)

    # svm 학습
    svm.train(train_data, cv2.ml.ROW_SAMPLE, train_label)
    # svm.trainAuto(train_data, cv2.ml.ROW_SAMPLE, train_label)
    endT = time.time() - startT
    print('SVM training complete. %.2f Min'%(endT/60))  
    # 훈련된  결과 모델 저장 ---⑧
    svm.save('svm_mnist.xml')

    # 테스트 이미지 글씨 바로 세우기 및 HOG 계산---⑨
    deskewed = [list(map(deskew,row)) for row in test_data]
    hogdata = [list(map(hogDesc.compute,row)) for row in deskewed]
    test_data = np.float32(hogdata)
    # 테스트용 HOG 데이타 재배열 ---⑩
    test_data = test_data.reshape(-1,test_data.shape[2])
    # 테스트 데이타 결과 예측 ---⑪
    ret, result = svm.predict(test_data)
    # 예측 결과와 테스트 레이블이 맞은 갯수 합산 및 정확도 출력---⑫
    correct = (result==test_label).sum()
    print('Accuracy: %.2f%%'%(correct*100.0/result.size))
    '''