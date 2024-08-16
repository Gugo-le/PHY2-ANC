import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd
from scipy.fft import rfft, rfftfreq, irfft

# 샘플링 레이트 및 차단 주파수 설정
fs = 44100  # 샘플링 레이트 (Hz)
cutoff_frequency = 1000  # 노이즈 필터링할 주파수
blocksize = 1024  # 한 번에 처리할 샘플의 개수

# 노이즈 캔슬링 함수
def noise_cancellation(input_signal, frame_rate, cutoff_frequency):
    signal_freq = rfft(input_signal)
    frequencies = rfftfreq(len(input_signal), d=1/frame_rate)
    signal_freq[frequencies > cutoff_frequency] = 0
    output_signal = irfft(signal_freq)
    return output_signal

# 실시간 그래프 업데이트 함수
def update_plot(frame):
    global indata
    
    # 원본 신호의 주파수 스펙트럼 계산
    original_freq = rfft(indata[:, 0])
    original_magnitude = np.abs(original_freq)
    
    # 노이즈 제거된 신호의 주파수 스펙트럼 계산
    filtered_signal = noise_cancellation(indata[:, 0], fs, cutoff_frequency)
    filtered_freq = rfft(filtered_signal)
    filtered_magnitude = np.abs(filtered_freq)
    
    # 그래프 업데이트
    original_line.set_ydata(original_magnitude)
    filtered_line.set_ydata(filtered_magnitude)
    
    return original_line, filtered_line

# 오디오 콜백 함수
def callback(indata_, outdata, frames, time, status):
    global indata
    indata = indata_.copy()  # 현재 입력 데이터를 전역 변수에 저장
    
    # 노이즈 캔슬링 적용 후 출력
    outdata[:, 0] = noise_cancellation(indata[:, 0], fs, cutoff_frequency)

# 초기 그래프 설정
frequencies = rfftfreq(blocksize, d=1/fs)
indata = np.zeros((blocksize, 1))  # 빈 입력 데이터로 초기화

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 첫 번째 그래프: 원본 신호의 주파수 스펙트럼
original_line, = ax1.plot(frequencies, np.zeros_like(frequencies), label="Original Signal Spectrum")
ax1.set_xlim(0, 10000)  # 주파수 범위를 0Hz에서 10000Hz로 설정
ax1.set_ylim(0, 200)
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Magnitude")
ax1.set_title("Original Signal Spectrum")
ax1.legend()

# 두 번째 그래프: 노이즈 캔슬링된 신호의 주파수 스펙트럼
filtered_line, = ax2.plot(frequencies, np.zeros_like(frequencies), label="Noise Cancelled Spectrum", linestyle='--', color='orange')
ax2.set_xlim(0, 10000)  # 주파수 범위를 0Hz에서 10000Hz로 설정
ax2.set_ylim(0, 200)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")
ax2.set_title("Noise Cancelled Signal Spectrum")
ax2.legend()

# 스트림 설정 및 시작
stream = sd.Stream(callback=callback, channels=1, samplerate=fs, blocksize=blocksize)
stream.start()

# 애니메이션 설정
ani = FuncAnimation(fig, update_plot, blit=True, interval=50)

# 실시간 그래프 표시
plt.tight_layout()
plt.show()

# 스트림 중지 및 종료
stream.stop()
stream.close()