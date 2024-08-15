import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft, rfftfreq
from pydub import AudioSegment

# 오디오 파일 읽기
def load_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    samples = np.array(audio.get_array_of_samples(), dtype=float)
    return samples, audio.frame_rate

# 간단한 주파수 필터를 사용한 노이즈 제거
def noise_cancellation(input_signal, frame_rate, cutoff_frequency=1000):
    # Fourier Transform을 사용하여 주파수 영역으로 변환
    signal_freq = rfft(input_signal)
    frequencies = rfftfreq(len(input_signal), d=1/frame_rate)

    # 특정 주파수 이상의 성분을 제거
    signal_freq[frequencies > cutoff_frequency] = 0

    # 역 Fourier Transform을 사용하여 시간 영역으로 변환
    output_signal = irfft(signal_freq)
    return output_signal

# 결과 그래프 그리기
def plot_signals(original_signal, output_signal, frame_rate):
    plt.figure(figsize=(15, 10))

    # 원본 신호의 주파수 스펙트럼
    plt.subplot(3, 1, 1)
    plt.plot(np.linspace(0, len(original_signal) / frame_rate, num=len(original_signal)), original_signal)
    plt.title("Original Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # 노이즈 캔슬링 신호의 주파수 스펙트럼
    plt.subplot(3, 1, 2)
    plt.plot(np.linspace(0, len(output_signal) / frame_rate, num=len(output_signal)), output_signal)
    plt.title("Noise Cancelled Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # 주파수 도메인 시각화
    signal_freq = rfft(original_signal)
    output_freq = rfft(output_signal)
    frequencies = rfftfreq(len(original_signal), d=1/frame_rate)
    
    plt.subplot(3, 1, 3)
    plt.plot(frequencies, np.abs(signal_freq), label='Original Signal Spectrum')
    plt.plot(frequencies, np.abs(output_freq), label='Noise Cancelled Signal Spectrum', linestyle='--')
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()

    plt.tight_layout()
    plt.show()

# 노이즈 캔슬링된 오디오 파일 저장
def save_audio(output_signal, frame_rate, output_file_path):
    output_signal = np.int16(output_signal / np.max(np.abs(output_signal)) * 32767)  # 신호를 16비트로 스케일링
    output_audio = AudioSegment(
        output_signal.tobytes(), 
        frame_rate=frame_rate, 
        sample_width=2,  # 16-bit audio
        channels=1
    )
    output_audio.export(output_file_path, format="wav")

# 메인 함수
def main(input_file, output_file, cutoff_frequency=1000):
    input_signal, frame_rate = load_audio(input_file)
    output_signal = noise_cancellation(input_signal, frame_rate, cutoff_frequency)
    
    plot_signals(input_signal, output_signal, frame_rate)
    save_audio(output_signal, frame_rate, output_file)

# 사용 예시
input_file = "input.wav"  # 입력 오디오 파일
output_file = "output1.wav"  # 출력 오디오 파일
cutoff_frequency = 1000  # 필터링할 주파수(Hz)

main(input_file, output_file, cutoff_frequency)