import soundfile
import IPython
import matplotlib.pyplot as plt

# 사운드파일 불러오기
soundfile_path = './audio/iloveyou_kakao.mp3'
sound, sampling_rate = soundfile.read(soundfile_path)

# 역위상 음원 만들기
sound_reverse = sound * (-1)

# 원 음원과 역위상 음원 더하기
noise_canceling = sound + sound_reverse

# 그래프 보기
plt.figure(figsize = (20, 10))
plt.subplot(1, 3, 1)
plt.plot(range(len(sound)), sound)
plt.title('sound_original')
plt.subplot(1, 3, 2)
plt.plot(range(len(sound_reverse)), sound_reverse)
plt.title('sound_reverse')
plt.subplot(1, 3, 3)
plt.plot(range(len(noise_canceling)), noise_canceling)
plt.title('noise_canceling')
plt.show()