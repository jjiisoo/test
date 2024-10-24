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
train_spec_path = os.path.join(dataset_path, 'train_s_sr')
validation_spec_path = os.path.join(dataset_path, 'validation_spec')
test_spec_path = os.path.join(dataset_path, 'test_spec')

# 스펙트로그램을 생성하고 저장하는 함수
def create_spectrogram(src_path, dest_path, sr=None):
    # STFT 파라미터 설정
    n_fft = 1024 # 주파수 해상도를 높이기 위한 설정
    hop_length = 512  # 시간 해상도 조정
    win_length = 1024

    # 디렉토리 생성
    os.makedirs(dest_path, exist_ok=True)

    # 상위 폴더(예: class1, class2 등) 탐색
    for class_dir in os.listdir(src_path):
        class_dir_path = os.path.join(src_path, class_dir)

        if os.path.isdir(class_dir_path):
            spec_class_dir_path = os.path.join(dest_path, class_dir)
            os.makedirs(spec_class_dir_path, exist_ok=True)

            # 클래스 폴더 안에 있는 오디오 파일들 처리
            for filename in os.listdir(class_dir_path):
                if filename.endswith('.wav'):
                    audio_path = os.path.join(class_dir_path, filename)
                    process_file(audio_path, spec_class_dir_path, filename, n_fft, hop_length, win_length, sr)

def process_file(audio_path, dest_path, filename, n_fft, hop_length, win_length, sr):
    # 오디오 파일 로드 (sr 값이 None이면 원본 샘플레이트 유지)
    y, sr = librosa.load(audio_path, sr=sr)  
    
    # STFT 수행하여 스펙트로그램 생성
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    D_dB = librosa.amplitude_to_db(D, ref=np.max)

    # 스펙트로그램 이미지로 저장
    plt.figure(figsize=(16, 10))  # 그림 크기 설정
    librosa.display.specshow(D_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='magma')  # 컬러맵 설정

    # 파일 경로 설정
    spec_filename = filename.replace('.wav', '.png')
    spec_path = os.path.join(dest_path, spec_filename)

    # 스펙트로그램을 이미지 파일로 저장
    plt.axis('off')  # 축 제거
    plt.savefig(spec_path, bbox_inches='tight', pad_inches=0, dpi=400)  # DPI 400으로 해상도 개선
    plt.close()

# 샘플레이트 설정
sample_rate = 18000  # 원하는 샘플레이트로 설정 (예: 22050 Hz)

# train, validation, test 데이터에 대해 스펙트로그램 생성
create_spectrogram(train_path, train_spec_path, sr=sample_rate)
create_spectrogram(validation_path, validation_spec_path, sr=sample_rate)
create_spectrogram(test_path, test_spec_path, sr=sample_rate)

print("Spectrogram conversion completed.")
