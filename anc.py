import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft, rfftfreq
from pydub import AudioSegment

# 오디오 파일 읽기 함수
def load_audio(file_path):
    # AudioSegment를 사용해 오디오 파일을 로드
    audio = AudioSegment.from_file(file_path)
    
    # 오디오 데이터를 NumPy 배열로 변환
    samples = np.array(audio.get_array_of_samples(), dtype=float)
    
    # 샘플 배열과 샘플링 레이트를 반환
    return samples, audio.frame_rate

# 노이즈 제거 함수 (주파수 필터링을 통해)
def noise_cancellation(input_signal, frame_rate, cutoff_frequency=1000):
    # Fourier Transform을 사용하여 시간 영역 신호를 주파수 영역으로 변환
    signal_freq = rfft(input_signal)
    
    # 주파수 축 생성 (각 주파수 성분이 대응하는 주파수 값)
    frequencies = rfftfreq(len(input_signal), d=1/frame_rate)

    # cutoff_frequency 이상인 주파수 성분 제거 (저역통과 필터)
    signal_freq[frequencies > cutoff_frequency] = 0

    # 역 Fourier Transform을 사용하여 주파수 영역에서 다시 시간 영역으로 변환
    output_signal = irfft(signal_freq)
    
    # 노이즈가 제거된 신호 반환
    return output_signal

# 신호를 시각화하는 함수
def plot_signals(original_signal, output_signal, frame_rate):
    plt.figure(figsize=(15, 10))  # 플롯 크기 설정

    # 원본 신호의 시간 도메인 플롯
    plt.subplot(3, 1, 1)
    plt.plot(np.linspace(0, len(original_signal) / frame_rate, num=len(original_signal)), original_signal)
    plt.title("Original Signal")  # 플롯 제목
    plt.xlabel("Time (s)")  # X축 레이블
    plt.ylabel("Amplitude")  # Y축 레이블

    # 노이즈 제거된 신호의 시간 도메인 플롯
    plt.subplot(3, 1, 2)
    plt.plot(np.linspace(0, len(output_signal) / frame_rate, num=len(output_signal)), output_signal)
    plt.title("Noise Cancelled Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # 주파수 도메인에서의 스펙트럼 비교
    signal_freq = rfft(original_signal)  # 원본 신호의 주파수 스펙트럼
    output_freq = rfft(output_signal)  # 노이즈 제거된 신호의 주파수 스펙트럼
    frequencies = rfftfreq(len(original_signal), d=1/frame_rate)
    
    plt.subplot(3, 1, 3)
    plt.plot(frequencies, np.abs(signal_freq), label='Original Signal Spectrum')
    plt.plot(frequencies, np.abs(output_freq), label='Noise Cancelled Signal Spectrum', linestyle='--')
    plt.title("Frequency Spectrum")  # 플롯 제목
    plt.xlabel("Frequency (Hz)")  # X축 레이블
    plt.ylabel("Magnitude")  # Y축 레이블
    plt.legend()  # 범례 추가

    plt.tight_layout()  # 레이아웃 조정
    plt.show()  # 플롯 출력

# 노이즈 제거된 오디오를 파일로 저장하는 함수
def save_audio(output_signal, frame_rate, output_file_path):
    # 신호를 16비트 정수형으로 스케일링 (오디오 데이터는 일반적으로 16비트 정수로 표현)
    output_signal = np.int16(output_signal / np.max(np.abs(output_signal)) * 32767)
    
    # NumPy 배열을 AudioSegment로 변환
    output_audio = AudioSegment(
        output_signal.tobytes(),  # 신호 데이터를 바이트로 변환
        frame_rate=frame_rate,  # 샘플링 레이트 설정
        sample_width=2,  # 샘플당 바이트 수 (16비트 = 2바이트)
        channels=1  # 모노 오디오
    )
    
    # 파일로 내보내기 (wav 포맷으로 저장)
    output_audio.export(output_file_path, format="wav")

# 메인 함수 (프로그램 실행의 진입점)
def main(input_file, output_file, cutoff_frequency=1000):
    # 오디오 파일 로드
    input_signal, frame_rate = load_audio(input_file)
    
    # 노이즈 제거
    output_signal = noise_cancellation(input_signal, frame_rate, cutoff_frequency)
    
    # 신호 시각화 (원본과 노이즈 제거된 신호 비교)
    plot_signals(input_signal, output_signal, frame_rate)
    
    # 결과를 오디오 파일로 저장
    save_audio(output_signal, frame_rate, output_file)

# 사용 예시 (input_file에서 output_file로 변환)
input_file = "input.wav"  # 입력 오디오 파일 경로
output_file = "output1.wav"  # 출력 오디오 파일 경로
cutoff_frequency = 1000  # 필터링할 주파수(Hz)

# 메인 함수 실행
main(input_file, output_file, cutoff_frequency)