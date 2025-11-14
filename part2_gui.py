"""
Part 2: 피스타치오 분류 GUI 애플리케이션
"""


# PyQt5 GUI 라이브러리 임포트
from PyQt5 import QtCore, QtWidgets  # Qt의 핵심 기능과 위젯들
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog  # 애플리케이션, 다이얼로그, 파일 선택 창
from PyQt5.QtGui import QPixmap, QFont  # 이미지 표시와 폰트 설정
import os  # 파일 존재 여부 확인용
import sys  # 시스템 관련 기능 (종료 등)
from tensorflow.keras.models import load_model  # 저장된 모델 불러오기
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # 이미지 로드 및 배열 변환
import numpy as np  # 수치 연산


class Ui_Dialog(object):  # GUI 레이아웃을 정의하는 클래스
    def setupUi(self, Dialog):  # UI 구성 요소를 설정하는 메서드
        Dialog.setObjectName("Dialog")  # 다이얼로그 객체 이름 설정
        Dialog.resize(900, 750)  # 창 크기를 900x750 픽셀로 설정

        # Title Label (제목 레이블)
        self.titleLabel = QtWidgets.QLabel(Dialog)  # 레이블 위젯 생성
        self.titleLabel.setGeometry(QtCore.QRect(0, 10, 900, 40))  # 위치(x, y, 너비, 높이) 설정
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)  # 텍스트 중앙 정렬
        font = QFont()  # 폰트 객체 생성
        font.setPointSize(18)  # 폰트 크기 18로 설정
        font.setBold(True)  # 폰트를 굵게 설정
        self.titleLabel.setFont(font)  # 레이블에 폰트 적용

        # Load Image Button (이미지 불러오기 버튼)
        self.pushButton = QtWidgets.QPushButton(Dialog)  # 푸시 버튼 위젯 생성
        self.pushButton.setGeometry(QtCore.QRect(380, 60, 140, 35))  # 버튼 위치와 크기 설정
        font = QFont()  # 버튼용 폰트 객체 생성
        font.setPointSize(12)  # 폰트 크기 12
        font.setBold(True)  # 굵게
        self.pushButton.setFont(font)  # 버튼에 폰트 적용

        # Graphics View for Image (이미지 표시 영역)
        self.graphicsView = QtWidgets.QGraphicsView(Dialog)  # 그래픽 뷰 위젯 생성
        self.graphicsView.setGeometry(QtCore.QRect(150, 110, 600, 400))  # 이미지 표시 영역 크기 설정

        # Classification Result Title (분류 결과 제목)
        self.resultTitleLabel = QtWidgets.QLabel(Dialog)  # 결과 제목 레이블 생성
        self.resultTitleLabel.setGeometry(QtCore.QRect(0, 520, 900, 30))  # 위치와 크기 설정
        self.resultTitleLabel.setAlignment(QtCore.Qt.AlignCenter)  # 중앙 정렬
        font = QFont()  # 폰트 객체 생성
        font.setPointSize(14)  # 폰트 크기 14
        font.setBold(True)  # 굵게
        self.resultTitleLabel.setFont(font)  # 폰트 적용

        # Separator (구분선)
        self.separatorLabel = QtWidgets.QLabel(Dialog)  # 구분선 레이블 생성
        self.separatorLabel.setGeometry(QtCore.QRect(0, 555, 900, 20))  # 위치와 크기
        self.separatorLabel.setAlignment(QtCore.Qt.AlignCenter)  # 중앙 정렬

        # Result Text (결과 텍스트 영역)
        self.resultText = QtWidgets.QTextEdit(Dialog)  # 텍스트 편집 위젯 생성
        self.resultText.setGeometry(QtCore.QRect(250, 580, 400, 150))  # 결과 표시 영역 크기
        self.resultText.setReadOnly(True)  # 읽기 전용으로 설정 (사용자가 수정 못함)
        font = QFont()  # 폰트 객체 생성
        font.setPointSize(11)  # 폰트 크기 11
        self.resultText.setFont(font)  # 폰트 적용

        self.retranslateUi(Dialog)  # 텍스트 번역/설정 메서드 호출
        QtCore.QMetaObject.connectSlotsByName(Dialog)  # 시그널과 슬롯 자동 연결

    def retranslateUi(self, Dialog):  # UI 텍스트를 설정하는 메서드
        _translate = QtCore.QCoreApplication.translate  # 번역 함수 가져오기
        Dialog.setWindowTitle(_translate("Dialog", "AI Image Classifier"))  # 창 제목 설정
        self.titleLabel.setText(_translate("Dialog", "AI Image Classifier"))  # 타이틀 레이블 텍스트
        self.pushButton.setText(_translate("Dialog", "Load image"))  # 버튼 텍스트
        self.resultTitleLabel.setText(_translate("Dialog", "Classification result"))  # 결과 제목 텍스트
        self.separatorLabel.setText(_translate("Dialog", "─" * 50))  # 구분선 (─ 기호 50개)


class PistachioClassifierApp(QDialog):  # 피스타치오 분류 애플리케이션 메인 클래스
    def __init__(self):  # 생성자 (초기화 메서드)
        super().__init__()  # 부모 클래스(QDialog) 생성자 호출
        self.ui = Ui_Dialog()  # UI 객체 생성
        self.ui.setupUi(self)  # UI 구성 요소 설정
        self.ui.pushButton.clicked.connect(self.loadImage)  # 버튼 클릭 시 loadImage 메서드 실행

        # 모델 로드
        self.model = None  # 모델 변수 초기화
        self.load_model()  # 모델 로드 메서드 호출

        # 클래스 이름
        self.class_names = ["Kirmizi_Pistachio", "Siirt_Pistachio"]  # 분류할 클래스 이름 리스트

    def load_model(self):  # 학습된 모델을 불러오는 메서드
        """피스타치오 모델 로드"""
        model_path = 'pistachio_model_final.h5'  # 모델 파일 경로 (상대 경로)
        try:  # 예외 처리 시작
            self.model = load_model(model_path)  # Keras 모델 불러오기
            print(f"✓ 모델 로드 완료: {model_path}")  # 성공 메시지 출력
        except Exception as e:  # 에러 발생 시
            print(f"✗ 모델 로드 실패: {e}")  # 에러 메시지 출력
            self.model = None  # 모델을 None으로 설정

    def loadImage(self):  # 이미지 파일을 선택하고 불러오는 메서드
        """이미지 파일 선택 및 표시"""
        # 파일 다이얼로그 열기
        options = QFileDialog.Options()  # 파일 다이얼로그 옵션 객체
        image_path, _ = QFileDialog.getOpenFileName(  # 파일 선택 창 열기
            self,  # 부모 위젯
            "Select Pistachio Image",  # 다이얼로그 제목
            "",  # 시작 디렉토리 (빈 문자열 = 현재 디렉토리)
            "Image Files (*.png *.jpg *.jpeg);;All Files (*)",  # 파일 필터 (이미지 파일만)
            options=options  # 옵션 전달
        )

        if image_path and os.path.isfile(image_path):  # 파일 경로가 유효하고 파일이 존재하면
            # 이미지 표시
            self.displayImage(image_path)  # 이미지를 화면에 표시

            # 이미지 분류
            if self.model is not None:  # 모델이 로드되어 있으면
                self.classifyImage(image_path)  # 이미지 분류 실행

    def displayImage(self, image_path):  # 이미지를 화면에 표시하는 메서드
        """이미지를 Graphics View에 표시 (높이 일정하게 유지)"""
        scene = QtWidgets.QGraphicsScene(self)  # 그래픽 씬 생성 (이미지를 담을 컨테이너)
        pixmap = QPixmap(image_path)  # 이미지 파일을 QPixmap 객체로 로드

        # 높이를 380으로 고정하고 비율 유지
        pixmap = pixmap.scaledToHeight(380, QtCore.Qt.SmoothTransformation)  # 높이 380px로 조정, 부드러운 변환

        item = QtWidgets.QGraphicsPixmapItem(pixmap)  # 픽스맵을 그래픽 아이템으로 변환
        scene.addItem(item)  # 씬에 아이템 추가
        self.ui.graphicsView.setScene(scene)  # 그래픽 뷰에 씬 설정 (화면에 표시)

    def classifyImage(self, image_path):  # 이미지를 분류하고 결과를 표시하는 메서드
        """이미지 분류 및 결과 표시"""
        try:  # 예외 처리 시작
            # 이미지 전처리 (120x120)
            image = load_img(image_path, target_size=(120, 120))  # 이미지를 120x120 크기로 로드
            image = img_to_array(image)  # 이미지를 numpy 배열로 변환 (120, 120, 3)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))  # 배치 차원 추가 (1, 120, 120, 3)
            image = image.astype('float32') / 255.0  # 정규화 (0~255 → 0~1)

            # 예측
            pred = self.model.predict(image, verbose=0)  # 모델로 예측 실행 (출력 없이)

            # 결과 계산 (sigmoid 출력: 0=Kirmizi, 1=Siirt)
            siirt_prob = pred[0][0]  # Siirt일 확률 (모델 출력값)
            kirmizi_prob = 1 - siirt_prob  # Kirmizi일 확률 (1 - Siirt 확률)

            # 확률 순으로 정렬
            results = [  # (클래스명, 확률) 튜플 리스트 생성
                (self.class_names[0], kirmizi_prob),  # Kirmizi 확률
                (self.class_names[1], siirt_prob)  # Siirt 확률
            ]
            results.sort(key=lambda x: x[1], reverse=True)  # 확률 높은 순으로 정렬

            # 결과 텍스트 생성
            result_text = "\n"  # 결과 텍스트 시작 (빈 줄)
            for i, (class_name, prob) in enumerate(results, 1):  # 정렬된 결과 순회 (1부터 번호 매김)
                result_text += f"  {i}. {class_name:20s} ({prob:.2f})\n"  # "1. Siirt_Pistachio   (0.85)" 형식

            result_text += "\n"  # 마지막 빈 줄 추가

            # 결과 표시
            self.ui.resultText.setText(result_text)  # 텍스트 영역에 결과 출력

        except Exception as e:  # 에러 발생 시
            self.ui.resultText.setText(f"\nError: {str(e)}")  # 에러 메시지 출력


if __name__ == '__main__':  # 이 파일이 직접 실행될 때만 실행
    app = QApplication(sys.argv)  # PyQt 애플리케이션 객체 생성 (명령줄 인자 전달)
    window = PistachioClassifierApp()  # 메인 윈도우 생성
    window.show()  # 윈도우 화면에 표시
    sys.exit(app.exec_())  # 애플리케이션 이벤트 루프 실행 (종료될 때까지 대기)
