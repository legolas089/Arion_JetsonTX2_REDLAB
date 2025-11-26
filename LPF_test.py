import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# --- 1. 시뮬레이션 파라미터 설정 ---
fs = 1000       # 샘플링 주파수 (Hz)
T = 2           # 총 신호 길이 (초)
n_samples = int(fs * T)
t = np.linspace(0, T, n_samples, endpoint=False)

# 필터 파라미터
filter_order = 4   # 필터 차수 (동일하게 유지)

# *** 비교할 두 개의 차단 주파수 ***
cutoff_hz_1 = 100.0  # (성공) 250Hz 노이즈를 제거할 LPF
cutoff_hz_2 = 200.0  # (실패) 250Hz 노이즈를 제거하지 못할 LPF

# 테스트 신호 주파수 (이전과 동일)
freq_low = 50.0    # 통과시킬 신호
freq_mid = 150.0   # 통과시키지 않을 신호
freq_high = 250.0  # 제거할 노이즈

# --- 2. 테스트 신호 생성 ---
sig_low = np.sin(2 * np.pi * freq_low * t)
sig_high = 0.5 * np.sin(2 * np.pi * freq_high * t)
sig_mid = 0.3 * np.sin(2 * np.pi * freq_mid * t)
sig_noisy = sig_low + sig_high + sig_mid

# --- 3. 두 LPF의 "계수" 생성 ---
nyquist = 0.5 * fs

# 필터 1 (100Hz) 계수 생성
wn1 = cutoff_hz_1 / nyquist
b1, a1 = signal.butter(filter_order, wn1, btype='low')
print(f"--- Filter 1 (Cutoff: {cutoff_hz_1}Hz) 계수 ---")
print(f"B1 (분자): {b1}")
print(f"A1 (분모): {a1}\n")

# 필터 2 (300Hz) 계수 생성
wn2 = cutoff_hz_2 / nyquist
b2, a2 = signal.butter(filter_order, wn2, btype='low')
print(f"--- Filter 2 (Cutoff: {cutoff_hz_2}Hz) 계수 ---")
print(f"B2 (분자): {b2}")
print(f"A2 (분모): {a2}\n")
print("==> 두 필터의 계수(숫자 배열)가 완전히 다른 것을 확인하세요.")

# --- 4. 신호에 "두 필터 각각" 적용 ---
sig_filtered_100 = signal.filtfilt(b1, a1, sig_noisy)
sig_filtered_300 = signal.filtfilt(b2, a2, sig_noisy)

# --- 5. FFT 수행 (비교용) ---
# 원본
fft_noisy = np.fft.fft(sig_noisy)
fft_noisy_mag = np.abs(fft_noisy) / n_samples
# 필터 1 (100Hz)
fft_filtered_100 = np.fft.fft(sig_filtered_100)
fft_filtered_100_mag = np.abs(fft_filtered_100) / n_samples
# 필터 2 (300Hz)
fft_filtered_300 = np.fft.fft(sig_filtered_300)
fft_filtered_300_mag = np.abs(fft_filtered_300) / n_samples

# FFT 주파수 축
fft_freq = np.fft.fftfreq(n_samples, 1/fs)
mask = (fft_freq >= 0) & (fft_freq <= fs / 2)


# --- 6. 결과 시각화 (3개 필터 동시 비교) ---
plt.figure(figsize=(12, 10))

# 그래프 1: 시간 영역 비교
plt.subplot(3, 1, 1)
plt.plot(t, sig_noisy, 'b-', label='Original (50+250Hz)', alpha=0.3)
plt.plot(t, sig_filtered_100, 'r-', label=f'Filtered (100Hz LPF) - [Success]', linewidth=2)
plt.plot(t, sig_filtered_300, 'g--', label=f'Filtered (300Hz LPF) - [Fail]', linewidth=2)
plt.title('Time Domain Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.xlim(0, 0.1) # 0.1초만 확대

# 그래프 2: LPF 자체의 주파수 응답 (가장 중요!)
plt.subplot(3, 1, 2)

w1, h1 = signal.freqz(b1, a1, worN=8000)
freq_resp_hz1 = (w1 / np.pi) * nyquist
plt.plot(freq_resp_hz1, np.abs(h1), 'r', label=f'100Hz LPF Response')
plt.axvline(cutoff_hz_1, color='red', linestyle='--')

w2, h2 = signal.freqz(b2, a2, worN=8000)
freq_resp_hz2 = (w2 / np.pi) * nyquist
plt.plot(freq_resp_hz2, np.abs(h2), 'g', label=f'300Hz LPF Response')
plt.axvline(cutoff_hz_2, color='green', linestyle='--')

# 제거 대상인 250Hz 노이즈 위치 표시
plt.axvline(freq_high, color='black', linestyle=':', label=f'Noise Freq ({freq_high}Hz)')
plt.title('Filter Frequency Responses (Why it fails/succeeds)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.legend()
plt.grid(True)
plt.xlim(0, fs / 2)

# 그래프 3: 주파수 영역(FFT) 비교
plt.subplot(3, 1, 3)
plt.plot(fft_freq[mask], fft_noisy_mag[mask], 'b-', label='Original FFT', alpha=0.3)
plt.plot(fft_freq[mask], fft_filtered_100_mag[mask], 'r-', label='Filtered FFT (100Hz) - [Success]', linewidth=2)
plt.plot(fft_freq[mask], fft_filtered_300_mag[mask], 'g--', label='Filtered FFT (300Hz) - [Fail]', linewidth=2)
# 제거 대상인 250Hz 노이즈 위치 표시
plt.axvline(freq_high, color='black', linestyle=':', label=f'Noise Freq ({freq_high}Hz)')
plt.title('Frequency Domain (FFT) Comparison')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()