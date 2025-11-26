import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================
# 1) dat 파일 읽기
# ==========================
def load_dat_as_float(path, dtype=float):
    """
    한 줄에 하나의 실수 값이 들어있는 .dat 파일을
    numpy 1D array로 읽어오는 함수.
    """
    path = Path(path)
    data = np.loadtxt(path, dtype=dtype)
    return data  # shape: (N,)


# ==========================
# 2) STFT + segment 함수
# ==========================
def stft_segments(
    x,
    fs=4000,
    n_fft=512,
    n_frames=10,
    hop=16,
    seg_stride=16,
    max_freq=64.0,
):
    """
    1D 신호 x 에 대해:
      - segment 길이: n_fft + (n_frames-1)*hop
      - segment 시작 간격: seg_stride
      - 각 segment마다 n_frames개의 STFT frame 계산
      - 주파수는 0~max_freq Hz까지만 사용

    반환:
      stft_all : shape (n_segments, n_freq_bins, n_frames)
      freqs    : 사용된 주파수 축 (Hz), shape (n_freq_bins,)
    """
    x = np.asarray(x, dtype=np.float32)
    N = len(x)

    # segment 길이 계산
    seg_len = n_fft + (n_frames - 1) * hop

    if N < seg_len:
        raise ValueError(f"신호 길이 N={N} 가 segment 길이 {seg_len} 보다 짧습니다.")

    # segment 시작 인덱스들
    starts = np.arange(0, N - seg_len + 1, seg_stride)
    n_segments = len(starts)

    print(f"[INFO] signal length: {N}")
    print(f"[INFO] segment length: {seg_len}")
    print(f"[INFO] n_segments: {n_segments}")

    # Hann window
    window = np.hanning(n_fft).astype(np.float32)

    # 주파수 축 (rfft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)  # shape (n_fft//2 + 1,)
    # 0 ~ max_freq Hz 범위만 사용
    freq_mask = freqs <= max_freq
    freqs_used = freqs[freq_mask]
    n_freq_bins = len(freqs_used)

    print(f"[INFO] n_fft={n_fft}, 전체 freq bins={len(freqs)}, 사용 freq bins={n_freq_bins}")
    print(f"[INFO] freq range used: {freqs_used[0]} ~ {freqs_used[-1]} Hz")

    # 결과 배열: (segment, freq, time-frame)
    stft_all = np.zeros((n_segments, n_freq_bins, n_frames), dtype=np.complex64)

    # ==========================
    #  각 segment마다 STFT 계산
    # ==========================
    for si, start in enumerate(starts):
        seg = x[start : start + seg_len]  # shape (seg_len,)

        for ti in range(n_frames):
            t0 = ti * hop
            frame = seg[t0 : t0 + n_fft]  # length n_fft
            if len(frame) < n_fft:
                # 이론상 나오면 안 되지만 방어 코드
                pad = np.zeros(n_fft - len(frame), dtype=np.float32)
                frame = np.concatenate([frame, pad])

            frame_windowed = frame * window
            spec = np.fft.rfft(frame_windowed)  # shape (n_fft//2 + 1,)
            stft_all[si, :, ti] = spec[freq_mask]

    return stft_all, freqs_used


# ==========================
# 3) 예시 실행부
# ==========================
if __name__ == "__main__":
    # 1) dat 파일 경로
    dat_path = "/home/rgblab/Arion_JetsonTX2_REDLAB/Arion_JetsonTX2_REDLAB/20251031-150459-4KHz-CH1.dat"

    # 2) 데이터 읽기
    x = load_dat_as_float(dat_path)
    print(f"[INFO] loaded data shape: {x.shape}")

    # 3) STFT + segment 계산
    stft_all, freqs_used = stft_segments(
        x,
        fs=4000,
        n_fft=512,
        n_frames=10,   # 윈도우 개수
        hop=16,        # 윈도우 간격
        seg_stride=16, # segment 간격
        max_freq=64.0, # 0~64 Hz
    )

    print(f"[INFO] STFT result shape: {stft_all.shape}")  # (n_segments, n_freq_bins, n_frames)

    # 4) 한 segment에 대한 magnitude 스펙트로그램 예시 플롯
    #    (원하면 주석 풀어서 확인)
    seg_idx = 0
    mag = np.abs(stft_all[seg_idx])  # shape (n_freq_bins, n_frames)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.imshow(
        mag,
        origin="lower",
        aspect="auto",
        extent=[0, mag.shape[1], freqs_used[0], freqs_used[-1]],
    )
    plt.colorbar(label="|X(f,t)|")
    plt.xlabel("Frame index (0~9)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Segment {seg_idx} STFT magnitude (0~64 Hz)")
    plt.tight_layout()
    plt.show()
