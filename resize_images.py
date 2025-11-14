"""
이미지 리사이즈 스크립트
600x600 이미지를 120x120으로 변환
"""

from PIL import Image
import os

# 경로 설정
original_base = "Pistachio_Image_Dataset"
resized_base = "Pistachio_Image_Dataset_120x120"

classes = ["Kirmizi_Pistachio", "Siirt_Pistachio"]

# 타겟 이미지 크기
target_size = (120, 120)

print("="*60)
print("피스타치오 이미지 리사이즈 시작")
print(f"원본 크기: 600x600 -> 타겟 크기: {target_size[0]}x{target_size[1]}")
print("="*60)

for class_name in classes:
    print(f"\n[{class_name}] 처리 중...")

    # 원본 및 대상 디렉토리 경로
    original_dir = os.path.join(original_base, class_name)
    resized_dir = os.path.join(resized_base, class_name)

    # 대상 디렉토리가 없으면 생성
    os.makedirs(resized_dir, exist_ok=True)

    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(original_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"  총 {len(image_files)}개의 이미지 발견")

    # 각 이미지 리사이즈
    success_count = 0
    error_count = 0

    for idx, img_file in enumerate(image_files, 1):
        try:
            # 원본 이미지 로드
            original_path = os.path.join(original_dir, img_file)
            img = Image.open(original_path)

            # 리사이즈 (고품질 리샘플링 사용)
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)

            # 저장
            resized_path = os.path.join(resized_dir, img_file)
            resized_img.save(resized_path, quality=95)

            success_count += 1

            # 진행상황 표시 (매 100개마다)
            if idx % 100 == 0:
                print(f"  진행: {idx}/{len(image_files)} 완료...")

        except Exception as e:
            print(f"\n  에러 발생: {img_file} - {str(e)}")
            error_count += 1

    print(f"  ✓ 성공: {success_count}개, 실패: {error_count}개")

print("\n" + "="*60)
print("리사이즈 작업 완료!")
print("="*60)

# 결과 확인
print("\n[결과 확인]")
for class_name in classes:
    resized_dir = os.path.join(resized_base, class_name)
    count = len([f for f in os.listdir(resized_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  {class_name}: {count}개 이미지")

# 첫 번째 이미지의 크기 확인
print("\n[크기 검증]")
for class_name in classes:
    resized_dir = os.path.join(resized_base, class_name)
    image_files = [f for f in os.listdir(resized_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if image_files:
        sample_path = os.path.join(resized_dir, image_files[0])
        with Image.open(sample_path) as img:
            print(f"  {class_name} 샘플 이미지 크기: {img.size}")
