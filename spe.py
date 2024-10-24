import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 데이터셋의 경로 설정
dataset_path = 'C:/Users/user/Documents/Dataset'
train_path = os.path.join(dataset_path, 'train')
validation_path = os.path.join(dataset_path, 'validation')
test_path = os.path.join(dataset_path, 'test')

# 스펙트로그램을 저장할 경로 설정
train_spec_path = os.path.join(dataset_path, 'train_ho')
validation_spec_path = os.path.join(dataset_path, 'validation_spec')
test_spec_path = os.path.join(dataset_path, 'test_spec')

# 스펙트로그램을 생성하고 저장하는 함수
def create_spectrogram(src_path, dest_path):
    # STFT 파라미터
    n_fft = 1024 # 주파수 해상도를 높이기 위해 n_fft를 크게 설정
    hop_length = 512  # 시간 해상도 조정
    win_length = 800

    # 디렉토리 생성
    os.makedirs(dest_path, exist_ok=True)

    # 상위 폴더(예: class1, class2 등)만 탐색
    for class_dir in os.listdir(src_path):
        class_dir_path = os.path.join(src_path, class_dir)

        # 클래스 폴더 안의 파일들 처리
        if os.path.isdir(class_dir_path):
            spec_class_dir_path = os.path.join(dest_path, class_dir)
            os.makedirs(spec_class_dir_path, exist_ok=True)

            # 클래스 폴더 안에 있는 오디오 파일들 처리
            for filename in os.listdir(class_dir_path):
                if filename.endswith('.wav'):
                    audio_path = os.path.join(class_dir_path, filename)
                    process_file(audio_path, spec_class_dir_path, filename, n_fft, hop_length, win_length)

def process_file(audio_path, dest_path, filename, n_fft, hop_length, win_length):
    # 오디오 파일 로드
    y, sr = librosa.load(audio_path)

    # Mel 스케일 스펙트로그램으로 변환
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=128)
    S_dB = librosa.amplitude_to_db(S, ref=np.max)  # dB 스케일로 변환

    # 스펙트로그램 이미지로 저장 (축과 레이블 없이)
    plt.figure(figsize=(16, 10))  # 그림 크기 조정
    plt.axis('off')  # 축 숨기기
    librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='inferno')  # 컬러맵 변경

    # 파일 경로 설정
    spec_filename = filename.replace('.wav', '.png')
    spec_path = os.path.join(dest_path, spec_filename)

    # 스펙트로그램을 이미지 파일로 저장 (DPI 조정)
    plt.savefig(spec_path, bbox_inches='tight', pad_inches=0)  # DPI 300으로 이미지 해상도 개선
    plt.close()

# train, validation, test 데이터에 대해 스펙트로그램 생성
create_spectrogram(train_path, train_spec_path)
create_spectrogram(validation_path, validation_spec_path)
create_spectrogram(test_path, test_spec_path)

print("Spectrogram conversion completed.")
