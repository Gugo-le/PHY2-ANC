import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fft import rfft, irfft, rfftfreq
from matplotlib.animation import FuncAnimation

# 샘플링 레이트 및 차단 주파수 설정
fs = 44100  # 샘플링 레이트 (Hz)
blocksize = 1024  # 한 번에 처리할 샘플의 개수
cutoff_frequency = 1000  # 노이즈 캔슬링 필터의 차단 주파수 (Hz)

# 오디오 콜백 함수
def callback(indata_, outdata, frames, time, status):
    global indata
    indata = indata_.copy()  # 현재 입력 데이터를 전역 변수에 저장
    
    # Fourier Transform을 사용하여 주파수 영역으로 변환
    signal_freq = rfft(indata[:, 0])
    frequencies = rfftfreq(len(indata), d=1/fs)

    # 특정 주파수 이상의 성분을 제거 (노이즈 제거)
    signal_freq[frequencies > cutoff_frequency] = 0

    # 역 Fourier Transform을 사용하여 시간 영역으로 변환
    filtered_signal = irfft(signal_freq)

    # 노이즈 캔슬링 처리된 신호를 출력
    outdata[:, 0] = filtered_signal

# 초기 그래프 설정
indata = np.zeros((blocksize, 1))  # 빈 입력 데이터로 초기화
outdata = np.zeros((blocksize, 1))  # 빈 출력 데이터로 초기화
time_axis = np.linspace(0, blocksize/fs, blocksize)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 첫 번째 그래프: 원본 신호의 시간 영역 파형
original_line, = ax1.plot(time_axis, np.zeros_like(time_axis), label="Original Signal")
ax1.set_ylim(-1, 1)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
ax1.set_title("Original Signal Waveform")
ax1.legend()

# 두 번째 그래프: 노이즈 캔슬링된 신호의 시간 영역 파형
filtered_line, = ax2.plot(time_axis, np.zeros_like(time_axis), label="Noise Cancelled Signal", color='orange')
ax2.set_ylim(-1, 1)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude")
ax2.set_title("Noise Cancelled Signal Waveform")
ax2.legend()

# 실시간 그래프 업데이트 함수
def update_plot(frame):
    original_line.set_ydata(indata[:, 0])
    filtered_line.set_ydata(outdata[:, 0])
    return original_line, filtered_line

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