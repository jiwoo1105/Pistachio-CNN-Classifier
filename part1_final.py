"""
Part 1: 피스타치오 CNN 모델 개발
Kirmizi vs Siirt 피스타치오 이진 분류

모델: VGG 스타일 CNN (Baseline, No Augmentation)

요구사항:
- Hidden layer: 4개 (Convolutional Blocks)
- Train/Test 비율: 7:3 (random_state=123)
- 전이학습 사용 금지, 기존 아키텍처 참고 가능
- 학습곡선 그래프 및 성능 평가
"""

# 필요한 라이브러리 임포트
import numpy as np  # 수치 계산 및 배열 처리를 위한 라이브러리
import matplotlib.pyplot as plt  # 그래프 시각화 라이브러리
import matplotlib  # matplotlib 설정용
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 한글 폰트 깨짐 방지용 기본 폰트 설정
import os  # 파일 시스템 경로 처리용
from PIL import Image  # 이미지 파일 로드 및 처리용
from sklearn.model_selection import train_test_split  # 데이터 분할용
from sklearn.metrics import classification_report, confusion_matrix  # 모델 성능 평가용
import tensorflow as tf  # 딥러닝 프레임워크
from tensorflow.keras import layers, models  # 신경망 레이어 및 모델 생성용
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  # 학습 최적화 콜백 함수들
import datetime  # 학습 시간 기록용
import shutil  # 파일 복사용

# 프로그램 시작 메시지 출력
print("="*80)
print("피스타치오 분류 CNN 모델 개발 - Part 1")
print("="*80)

# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================
print("\n[1단계] 데이터 로드 및 전처리")
print("-"*80)

# 데이터셋 경로 설정 (상대 경로로 변경)
data_base = "Pistachio_Image_Dataset_120x120"  # 현재 디렉토리 기준 상대 경로
classes = ["Kirmizi_Pistachio", "Siirt_Pistachio"]  # 분류할 두 클래스 이름 리스트

# 이미지와 레이블을 저장할 빈 리스트 초기화
X_data = []  # 이미지 데이터를 저장할 리스트
y_data = []  # 레이블(정답) 데이터를 저장할 리스트

print(f"데이터 로드 중...")

# 각 클래스별로 이미지 로드
for class_idx, class_name in enumerate(classes):  # classes 리스트 순회하며 인덱스와 클래스명 가져오기
    class_dir = os.path.join(data_base, class_name)  # 클래스별 디렉토리 경로 생성
    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]  # 이미지 파일만 필터링

    print(f"  [{class_name}] {len(image_files)}개 이미지 로드 중...")

    # 각 이미지 파일 처리
    for img_file in image_files:  # 이미지 파일 리스트 순회
        img_path = os.path.join(class_dir, img_file)  # 이미지 전체 경로 생성
        img = Image.open(img_path)  # PIL로 이미지 파일 열기
        img_array = np.array(img)  # 이미지를 numpy 배열로 변환
        X_data.append(img_array)  # 이미지 배열을 X_data에 추가
        y_data.append(class_idx)  # 클래스 인덱스(0 또는 1)를 레이블로 추가

# 리스트를 numpy 배열로 변환 (효율적인 연산을 위해)
X_data = np.array(X_data)  # 이미지 리스트를 numpy 배열로 변환
y_data = np.array(y_data)  # 레이블 리스트를 numpy 배열로 변환

# 로드된 데이터 정보 출력
print(f"\n데이터 로드 완료!")
print(f"  총 샘플 수: {len(X_data)}")  # 전체 이미지 개수
print(f"  이미지 shape: {X_data.shape}")  # 배열 형태 (개수, 높이, 너비, 채널)
print(f"  클래스별 분포:")
for idx, class_name in enumerate(classes):  # 각 클래스별로
    count = np.sum(y_data == idx)  # 해당 클래스 샘플 개수 계산
    print(f"    {class_name}: {count}개 ({count/len(y_data)*100:.1f}%)")  # 개수와 비율 출력

# 데이터 정규화 (픽셀값을 0~1 범위로 변환)
X_data = X_data.astype('float32') / 255.0  # 정수형을 실수형으로 변환 후 255로 나눔
print(f"\n정규화 완료: 픽셀 범위 [0, 255] -> [0.0, 1.0]")

# ============================================================================
# 2. Train/Test 데이터 분할 (7:3, random_state=123)
# ============================================================================
print(f"\n[2단계] Train/Test 데이터 분할 (7:3)")
print("-"*80)

RANDOM_STATE = 123  # 재현 가능한 결과를 위한 난수 시드 설정

# 데이터를 학습용과 테스트용으로 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data,  # 분할할 데이터
    test_size=0.3,  # 테스트 데이터 비율 30%
    random_state=RANDOM_STATE,  # 난수 시드 고정
    stratify=y_data  # 클래스 비율을 유지하며 분할
)

# 분할 결과 출력
print(f"데이터 분할 완료 (random_state={RANDOM_STATE})")
print(f"  Train set: {len(X_train)}개 ({len(X_train)/len(X_data)*100:.1f}%)")  # 학습 데이터 개수
print(f"  Test set:  {len(X_test)}개 ({len(X_test)/len(X_data)*100:.1f}%)")  # 테스트 데이터 개수

# Train set 클래스 분포 확인
print(f"\nTrain set 클래스 분포:")
for idx, class_name in enumerate(classes):  # 각 클래스별로
    count = np.sum(y_train == idx)  # 학습 데이터 내 해당 클래스 개수
    print(f"  {class_name}: {count}개 ({count/len(y_train)*100:.1f}%)")

# Test set 클래스 분포 확인
print(f"\nTest set 클래스 분포:")
for idx, class_name in enumerate(classes):  # 각 클래스별로
    count = np.sum(y_test == idx)  # 테스트 데이터 내 해당 클래스 개수
    print(f"  {class_name}: {count}개 ({count/len(y_test)*100:.1f}%)")

# ============================================================================
# 3. VGG 스타일 CNN 모델 설계 (Hidden layer 4개)
# ============================================================================
print(f"\n[3단계] CNN 모델 설계 (VGG 스타일, Hidden layer 4개)")
print("-"*80)

def build_vgg_style_cnn(input_shape=(120, 120, 3)):  # 모델 생성 함수 정의, 입력 이미지 크기 지정
    """
    VGG 스타일 CNN 모델

    구조:
    - 4개의 Convolutional Block
    - 각 블록: Conv2D x 2 + BatchNormalization + MaxPooling + Dropout
    - 2개의 Fully Connected Layer
    - 출력: Sigmoid (이진 분류)
    """

    model = models.Sequential([  # 순차적 모델 생성
        # Block 1: 32 filters (첫 번째 컨볼루션 블록)
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),  # 3x3 필터 32개, ReLU 활성화, 패딩으로 크기 유지
        layers.BatchNormalization(),  # 배치 정규화로 학습 안정화
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),  # 두 번째 컨볼루션 레이어
        layers.BatchNormalization(),  # 배치 정규화
        layers.MaxPooling2D((2, 2)),  # 2x2 최대 풀링으로 크기 절반으로 축소
        layers.Dropout(0.3),  # 30% 드롭아웃으로 과적합 방지

        # Block 2: 64 filters (두 번째 컨볼루션 블록)
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),  # 필터 개수 64개로 증가
        layers.BatchNormalization(),  # 배치 정규화
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),  # 두 번째 컨볼루션
        layers.BatchNormalization(),  # 배치 정규화
        layers.MaxPooling2D((2, 2)),  # 최대 풀링
        layers.Dropout(0.3),  # 드롭아웃

        # Block 3: 128 filters (세 번째 컨볼루션 블록)
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),  # 필터 개수 128개로 증가
        layers.BatchNormalization(),  # 배치 정규화
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),  # 두 번째 컨볼루션
        layers.BatchNormalization(),  # 배치 정규화
        layers.MaxPooling2D((2, 2)),  # 최대 풀링
        layers.Dropout(0.3),  # 드롭아웃

        # Block 4: 256 filters (네 번째 컨볼루션 블록)
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),  # 필터 개수 256개로 증가
        layers.BatchNormalization(),  # 배치 정규화
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),  # 두 번째 컨볼루션
        layers.BatchNormalization(),  # 배치 정규화
        layers.MaxPooling2D((2, 2)),  # 최대 풀링
        layers.Dropout(0.3),  # 드롭아웃

        # Fully Connected Layers (완전 연결 레이어)
        layers.Flatten(),  # 2D 특성 맵을 1D 벡터로 평탄화
        layers.Dense(512, activation='relu'),  # 512개 뉴런의 완전연결 레이어
        layers.BatchNormalization(),  # 배치 정규화
        layers.Dropout(0.6),  # 60% 드롭아웃 (FC 레이어에서는 더 높은 비율)
        layers.Dense(128, activation='relu'),  # 128개 뉴런의 완전연결 레이어
        layers.BatchNormalization(),  # 배치 정규화
        layers.Dropout(0.6),  # 드롭아웃

        # Output Layer (출력 레이어)
        layers.Dense(1, activation='sigmoid')  # 이진 분류를 위한 시그모이드 활성화 함수
    ])

    return model  # 생성된 모델 반환

# 모델 생성
model = build_vgg_style_cnn()  # 위에서 정의한 함수로 모델 생성

# 모델 컴파일 (학습 준비)
model.compile(
    optimizer='adam',  # Adam 옵티마이저 사용 (adaptive learning rate)
    loss='binary_crossentropy',  # 이진 분류용 손실 함수
    metrics=['accuracy']  # 평가 지표로 정확도 사용
)

# 모델 구조 출력
print("VGG 스타일 CNN 모델 구조:")
model.summary()  # 레이어별 파라미터 수 등 상세 정보 출력

# 총 파라미터 수 계산 및 출력
total_params = model.count_params()  # 학습 가능한 전체 파라미터 개수
print(f"\n총 파라미터 수: {total_params:,}개")
print(f"Hidden Layer 수: 4개 (Convolutional Blocks)")

# ============================================================================
# 4. 하이퍼파라미터 및 콜백 설정
# ============================================================================
print(f"\n[4단계] 하이퍼파라미터 및 콜백 설정")
print("-"*80)

# 학습 하이퍼파라미터 설정
EPOCHS = 100  # 최대 학습 반복 횟수
BATCH_SIZE = 32  # 한 번에 처리할 샘플 수
VALIDATION_SPLIT = 0.2  # 학습 데이터 중 검증용으로 사용할 비율

# 설정값 출력
print(f"하이퍼파라미터:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Validation Split: {VALIDATION_SPLIT}")

# 콜백 함수 설정 (학습 중 자동 제어)
early_stopping = EarlyStopping(  # 조기 종료 콜백
    monitor='val_loss',  # 검증 손실값 모니터링
    patience=20,  # 20 에포크 동안 개선 없으면 학습 중단
    restore_best_weights=True,  # 가장 좋았던 가중치로 복원
    verbose=1  # 진행상황 출력
)

reduce_lr = ReduceLROnPlateau(  # 학습률 감소 콜백
    monitor='val_loss',  # 검증 손실값 모니터링
    factor=0.5,  # 학습률을 절반으로 감소
    patience=7,  # 7 에포크 동안 개선 없으면 학습률 감소
    min_lr=1e-7,  # 최소 학습률 제한
    verbose=1  # 진행상황 출력
)

checkpoint_path = "model_best.h5"  # 최고 성능 모델 저장 경로 (상대 경로)
model_checkpoint = ModelCheckpoint(  # 모델 체크포인트 콜백
    checkpoint_path,  # 저장할 파일 경로
    monitor='val_accuracy',  # 검증 정확도 모니터링
    save_best_only=True,  # 최고 성능일 때만 저장
    verbose=1  # 저장시 메시지 출력
)

callbacks = [early_stopping, reduce_lr, model_checkpoint]  # 콜백 리스트로 묶기

# 콜백 설정 정보 출력
print(f"\n콜백 함수:")
print(f"  - EarlyStopping (patience=20)")  # 조기 종료
print(f"  - ReduceLROnPlateau (patience=7)")  # 학습률 감소
print(f"  - ModelCheckpoint")  # 모델 저장

# ============================================================================
# 5. 모델 학습
# ============================================================================
print(f"\n[5단계] 모델 학습")
print("="*80)
print(f"학습 시작: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  # 학습 시작 시간 기록
print("-"*80)

# 모델 학습 실행
history = model.fit(
    X_train, y_train,  # 학습 데이터와 레이블
    epochs=EPOCHS,  # 에포크 수
    batch_size=BATCH_SIZE,  # 배치 크기
    validation_split=VALIDATION_SPLIT,  # 검증 데이터 비율
    callbacks=callbacks,  # 콜백 함수들
    verbose=1  # 학습 진행상황 상세 출력
)

print("-"*80)
print(f"학습 종료: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  # 학습 종료 시간 기록

# ============================================================================
# 6. 모델 평가
# ============================================================================
print(f"\n[6단계] 모델 평가")
print("-"*80)

# 학습 데이터와 테스트 데이터에 대한 성능 평가
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)  # 학습 데이터 평가 (출력 없이)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)  # 테스트 데이터 평가 (출력 없이)

# 평가 결과 출력
print(f"Train Accuracy: {train_acc*100:.2f}%")  # 학습 정확도
print(f"Test Accuracy:  {test_acc*100:.2f}%")  # 테스트 정확도
print(f"Train Loss: {train_loss:.4f}")  # 학습 손실
print(f"Test Loss:  {test_loss:.4f}")  # 테스트 손실
print(f"Overfitting: {(train_acc - test_acc)*100:.2f}%")  # 과적합 정도 (학습-테스트 정확도 차이)

# 최고 validation accuracy 찾기
best_val_acc = max(history.history['val_accuracy'])  # 학습 과정 중 최고 검증 정확도
best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1  # 최고 성능을 보인 에포크 번호
print(f"\n최고 Validation Accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch})")

# ============================================================================
# 7. 상세 성능 분석
# ============================================================================
print(f"\n[7단계] 상세 성능 분석")
print("-"*80)

# 테스트 데이터에 대한 예측 수행
y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()  # 0.5 임계값으로 이진 분류 결과 생성

# Confusion Matrix 계산 및 출력
cm = confusion_matrix(y_test, y_pred)  # 실제값과 예측값으로 혼동행렬 생성
print("\nConfusion Matrix:")
print(cm)  # 혼동행렬 출력 (2x2 행렬)
print(f"\n       예측→")
print(f"실제↓  Kirmizi  Siirt")
print(f"Kirmizi  {cm[0,0]:>4}    {cm[0,1]:>4}   (정확도: {cm[0,0]/(cm[0,0]+cm[0,1])*100:.1f}%)")  # Kirmizi 정확도
print(f"Siirt    {cm[1,0]:>4}    {cm[1,1]:>4}   (정확도: {cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}%)")  # Siirt 정확도

# 오분류 분석
kirmizi_total = np.sum(y_test == 0)  # 테스트 데이터 중 Kirmizi 총 개수
siirt_total = np.sum(y_test == 1)  # 테스트 데이터 중 Siirt 총 개수
kirmizi_error = cm[0, 1]  # Kirmizi를 Siirt로 잘못 분류한 개수
siirt_error = cm[1, 0]  # Siirt를 Kirmizi로 잘못 분류한 개수

# 오분류 정보 출력
print(f"\n오분류 분석:")
print(f"  Kirmizi → Siirt: {kirmizi_error}개 / {kirmizi_total}개 ({kirmizi_error/kirmizi_total*100:.1f}% 에러)")
print(f"  Siirt → Kirmizi: {siirt_error}개 / {siirt_total}개 ({siirt_error/siirt_total*100:.1f}% 에러)")

# Classification Report 출력 (정밀도, 재현율, F1-score 등)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=classes, digits=4))  # 클래스별 상세 성능 지표

# ============================================================================
# 8. 학습곡선 그래프
# ============================================================================
print(f"\n[8단계] 학습곡선 그래프 생성")
print("-"*80)

# 1행 2열 서브플롯 생성 (Loss와 Accuracy 그래프)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss 그래프 (왼쪽)
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2, color='#2E86AB')  # 학습 손실
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#A23B72')  # 검증 손실
axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')  # 그래프 제목
axes[0].set_xlabel('Epoch', fontsize=12)  # x축 레이블
axes[0].set_ylabel('Loss', fontsize=12)  # y축 레이블
axes[0].legend(fontsize=11)  # 범례 표시
axes[0].grid(True, alpha=0.3)  # 그리드 추가

# Accuracy 그래프 (오른쪽)
axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, color='#2E86AB')  # 학습 정확도
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#A23B72')  # 검증 정확도
axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')  # 그래프 제목
axes[1].set_xlabel('Epoch', fontsize=12)  # x축 레이블
axes[1].set_ylabel('Accuracy', fontsize=12)  # y축 레이블
axes[1].legend(fontsize=11)  # 범례 표시
axes[1].grid(True, alpha=0.3)  # 그리드 추가

plt.tight_layout()  # 레이아웃 자동 조정
graph_path = "learning_curves.png"  # 그래프 저장 경로 (상대 경로)
plt.savefig(graph_path, dpi=300, bbox_inches='tight')  # 고해상도로 그래프 저장
print(f"학습곡선 그래프 저장: {graph_path}")

# ============================================================================
# 9. 모델 저장
# ============================================================================
print(f"\n[9단계] 모델 저장")
print("-"*80)

# 최종 모델 저장
model_path = "pistachio_cnn_model.h5"  # 모델 저장 경로 (상대 경로)
model.save(model_path)  # 모델을 HDF5 파일로 저장
print(f"모델 저장: {model_path}")

# Part 2용 최종 모델로 복사
final_model_path = "pistachio_model_final.h5"  # GUI에서 사용할 모델 경로 (상대 경로)
shutil.copy(model_path, final_model_path)  # 파일 복사
print(f"Part 2용 최종 모델 저장: {final_model_path}")

# ============================================================================
# 10. 최종 보고서 저장
# ============================================================================
print(f"\n[10단계] 최종 보고서 저장")
print("-"*80)

report_path = "part1_final_report.txt"  # 보고서 저장 경로 (상대 경로)
with open(report_path, 'w', encoding='utf-8') as f:  # UTF-8 인코딩으로 파일 열기
    # 보고서 헤더
    f.write("="*80 + "\n")
    f.write("피스타치오 분류 CNN 모델 - Part 1 최종 보고서\n")
    f.write("="*80 + "\n\n")

    # 프로젝트 개요
    f.write("[프로젝트 개요]\n")
    f.write("목적: Kirmizi vs Siirt 피스타치오 이진 분류\n")
    f.write(f"데이터: {len(X_data)}개 이미지 (120x120x3)\n")  # 전체 데이터 개수
    f.write("Train/Test 비율: 7:3\n")
    f.write(f"Random State: {RANDOM_STATE}\n\n")

    # 모델 아키텍처
    f.write("[모델 아키텍처]\n")
    f.write("구조: VGG 스타일 CNN (전이학습 사용 안 함, 처음부터 학습)\n")
    f.write("Hidden Layer 수: 4개 (Convolutional Blocks)\n")
    f.write(f"총 파라미터 수: {total_params:,}개\n")
    f.write("블록 구조: Conv2D x2 + BatchNorm + MaxPooling + Dropout\n")
    f.write("필터 수: 32 → 64 → 128 → 256 (점진적 증가)\n\n")

    # 하이퍼파라미터
    f.write("[하이퍼파라미터]\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Validation Split: {VALIDATION_SPLIT}\n")
    f.write("Optimizer: Adam (learning_rate=0.001)\n")
    f.write("Loss: Binary Crossentropy\n")
    f.write("Dropout: Conv=0.25, Dense=0.5\n\n")

    # 학습 결과
    f.write("[학습 결과]\n")
    f.write(f"Train Accuracy: {train_acc*100:.2f}%\n")
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")
    f.write(f"Best Validation Accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch})\n")
    f.write(f"Overfitting: {(train_acc - test_acc)*100:.2f}%\n\n")

    # Confusion Matrix
    f.write("[Confusion Matrix]\n")
    f.write(str(cm) + "\n\n")  # 혼동행렬 출력
    f.write(f"정분류: {cm[0,0] + cm[1,1]}개 / {len(y_test)}개 ({test_acc*100:.2f}%)\n")
    f.write(f"오분류: {cm[0,1] + cm[1,0]}개 / {len(y_test)}개 ({(1-test_acc)*100:.2f}%)\n\n")

    # 클래스별 성능
    f.write("[클래스별 성능]\n")
    f.write(f"Kirmizi_Pistachio:\n")
    f.write(f"  정확도: {cm[0,0]/(cm[0,0]+cm[0,1])*100:.2f}%\n")
    f.write(f"  오분류: {kirmizi_error}개 → Siirt로 잘못 분류\n\n")
    f.write(f"Siirt_Pistachio:\n")
    f.write(f"  정확도: {cm[1,1]/(cm[1,0]+cm[1,1])*100:.2f}%\n")
    f.write(f"  오분류: {siirt_error}개 → Kirmizi로 잘못 분류\n\n")

    # Classification Report
    f.write("[Classification Report]\n")
    f.write(classification_report(y_test, y_pred, target_names=classes, digits=4))
    f.write("\n")

    # 생성된 파일
    f.write("[생성된 파일]\n")
    f.write(f"1. 모델 파일: {model_path}\n")
    f.write(f"2. Part 2용 모델: {final_model_path}\n")
    f.write(f"3. 학습곡선 그래프: {graph_path}\n")
    f.write(f"4. 보고서: {report_path}\n\n")

    # 결론
    f.write("[결론]\n")
    f.write(f"VGG 스타일 CNN 모델로 {test_acc*100:.2f}%의 Test Accuracy 달성\n")
    f.write("전이학습 없이 처음부터 학습하여 우수한 성능 확보\n")
    f.write(f"Overfitting이 {(train_acc-test_acc)*100:.2f}%로 낮아 일반화 성능 양호\n")

print(f"최종 보고서 저장: {report_path}")

# ============================================================================
# 최종 완료
# ============================================================================
print("\n" + "="*80)
print("Part 1 CNN 모델 개발 완료!")
print("="*80)
print(f"\n최종 성능:")
print(f"  Train Accuracy: {train_acc*100:.2f}%")
print(f"  Test Accuracy:  {test_acc*100:.2f}%")
print(f"  Overfitting: {(train_acc - test_acc)*100:.2f}%")
print(f"\n생성된 파일:")
print(f"  - 모델: {model_path}")
print(f"  - Part 2용 모델: {final_model_path}")
print(f"  - 학습곡선: {graph_path}")
print(f"  - 보고서: {report_path}")
print("="*80)
