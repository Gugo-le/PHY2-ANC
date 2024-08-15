import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from pydub import AudioSegment

# 오디오 파일 읽기
def load_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    samples = np.array(audio.get_array_of_samples())
    return samples, audio.frame_rate

# 푸리에 변환 및 삼각함수 성분 계산
def decompose_to_sinusoids(signal, frame_rate):
    # Fourier 변환 수행
    signal_freq = rfft(signal)
    frequencies = rfftfreq(len(signal), d=1/frame_rate)

    # 진폭, 위상, 주파수를 사용해 각 성분을 삼각함수로 나타냄
    amplitudes = np.abs(signal_freq) / len(signal)
    phases = np.angle(signal_freq)
    
    # 삼각함수 성분들을 합쳐서 원래 신호 재구성
    t = np.arange(len(signal)) / frame_rate
    reconstructed_signal = np.zeros_like(signal, dtype=float)
    
    for i, freq in enumerate(frequencies):
        if amplitudes[i] > 0:  # 작은 성분들은 무시
            reconstructed_signal += amplitudes[i] * np.cos(2 * np.pi * freq * t + phases[i])
    
    return t, reconstructed_signal

# 신호와 삼각함수 성분 시각화
def plot_signals(original_signal, t, reconstructed_signal):
    plt.figure(figsize=(15, 10))

    # 원본 신호
    plt.subplot(3, 1, 1)
    plt.plot(t, original_signal[:len(t)])
    plt.title("Original Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # 삼각함수로 재구성된 신호
    plt.subplot(3, 1, 2)
    plt.plot(t, reconstructed_signal)
    plt.title("Reconstructed Signal from Sinusoids")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # 차이
    plt.subplot(3, 1, 3)
    plt.plot(t, original_signal[:len(t)] - reconstructed_signal)
    plt.title("Difference (Original - Reconstructed)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude Difference")

    plt.tight_layout()
    plt.show()

# 메인 함수
def main(input_file):
    # 오디오 파일 로드
    input_signal, frame_rate = load_audio(input_file)

    # 삼각함수 성분으로 신호 재구성
    t, reconstructed_signal = decompose_to_sinusoids(input_signal, frame_rate)

    # 신호와 삼각함수 성분 시각화
    plot_signals(input_signal, t, reconstructed_signal)

# 사용 예시
input_file = "input.wav"  # 입력 오디오 파일
main(input_file)