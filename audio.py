import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

# 오디오 파일 읽기
def load_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    samples = np.array(audio.get_array_of_samples())
    return samples, audio.frame_rate

# 오디오 신호 시각화
def plot_waveform(signal, frame_rate):
    time = np.arange(len(signal)) / frame_rate  # 시간 축 계산
    plt.figure(figsize=(15, 5))
    plt.plot(time, signal)
    plt.title("Waveform of Audio Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

# 메인 함수
def main(input_file):
    # 오디오 파일 로드
    input_signal, frame_rate = load_audio(input_file)

    # 오디오 신호 시각화
    plot_waveform(input_signal, frame_rate)

# 사용 예시
input_file = "input.wav"  # 입력 오디오 파일
main(input_file)