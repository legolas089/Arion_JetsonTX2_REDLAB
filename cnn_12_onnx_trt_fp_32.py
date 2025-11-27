"""
cnn_12_onnx_trt_fp_32.py

Small CNN (Convolutional Neural Network) classifier for 12-channel STFT (Short-Time Fourier Transform) images
→ ONNX (Open Neural Network Exchange, opset 11, STATIC axes)
→ TensorRT (FP32, 32-bit floating point) engine build via `trtexec`
→ Optional TensorRT Python API (Application Programming Interface) + PyCUDA (Python CUDA) runtime inference

Assumptions from the user spec:
- Input tensor: [N, 12, 64, 50]  (NCHW; 12 channels = 4 sensors × XYZ)
- STFT done on DSP board; amplitude scale + min–max normalization already applied
- Classes: 4
- Target device: Jetson TX2; use FP32; static input size

Usage examples
--------------
# Train → export ONNX (static) → build TRT FP32 engine with trtexec
python cnn_12_onnx_trt_fp_32.py --epochs 5 --export-dir ./artifacts

# Skip training; load checkpoint and export/build
python cnn_12_onnx_trt_fp_32.py --no-train --ckpt ./artifacts/cnn12.pt --export-dir ./artifacts

# Optional: run TRT Python inference comparison (requires tensorrt + pycuda)
python cnn_12_onnx_trt_fp_32.py --trt-python-infer

# On the Jetson TX2 (engine build via CLI):
trtexec --onnx=./artifacts/cnn12_static.onnx \
        --saveEngine=./artifacts/cnn12_fp32.engine \
        --workspace=1024 --verbose

Notes
-----
- STATIC axes: batch and spatial dims are fixed at export time. If you need different shapes, re-export.
- For legacy TensorRT on TX2, opset 11 is typically the safest. Adjust if your JetPack supports newer.
"""

# from __future__ import annotations
import argparse
from pathlib import Path
import os, time, shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

try:
    from sklearn.model_selection import train_test_split
except Exception:
    train_test_split = None

print("torch_version:",torch.__version__)
print("pytorch build in cuda:",torch.version.cuda)

print("cuda available:",torch.cuda.is_available())

# ----------------------
# Model: small CNN for [N,12,64,50]
# ----------------------
class SmallCNN(nn.Module):
    def __init__(self, in_ch: int = 12, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64x50 -> 32x25

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32x25 -> 16x12
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ----------------------
# Utils
# ----------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(1) == y).float().mean().item())


# ----------------------
# Data helpers (sin/cos → STFT → spectrogram [12,64,50])
# ----------------------

FS = 4000  # Hz (downsampled)
WIN = 200  # samples (50 ms)
HOP = 100  # samples (25 ms)
NFFT = 128


def _stft_amplitude(x: np.ndarray, n_fft=NFFT, win_length=WIN, hop_length=HOP) -> np.ndarray:
    """Single-channel STFT amplitude spectrogram.
    Returns shape (F, T) with F = n_fft//2+1, T = frames. Uses Hann window.
    """
    w = np.hanning(win_length).astype(np.float32)
    L = x.shape[0]
    frames = 1 + (L - win_length) // hop_length
    S = np.empty((n_fft // 2 + 1, frames), dtype=np.float32)
    for t in range(frames):
        s = t * hop_length
        frame = x[s:s + win_length]
        if frame.shape[0] < win_length:
            pad = np.zeros(win_length, dtype=np.float32)
            pad[: frame.shape[0]] = frame
            frame = pad
        frame = frame * w
        fft = np.fft.rfft(frame, n=n_fft)
        S[:, t] = np.abs(fft).astype(np.float32)
    return S


def _sincos_signal(length=5100, fs=FS, freqs=(120, 300), phase=None, noise_std=0.01):
    t = np.arange(length, dtype=np.float32) / fs
    f1, f2 = freqs
    if phase is None:
        phase = (np.random.rand()*2*np.pi, np.random.rand()*2*np.pi)
    p1, p2 = phase
    sig = np.sin(2*np.pi*f1*t + p1) + 0.5*np.cos(2*np.pi*f2*t + p2)
    if noise_std > 0:
        sig = sig + np.random.randn(*sig.shape).astype(np.float32) * noise_std
    return sig.astype(np.float32)


# frequency templates per class (Hz)
CLASS_FREQS = [
    (120, 300),  # class 0
    (200, 400),  # class 1
    (280, 600),  # class 2
    (350, 800),  # class 3
]


def make_sincos_dataset(n: int, num_classes: int = 4, in_shape=(12, 64, 50)):
    """Generate dataset by synthesizing 12-channel time-domain signals using sin/cos,
    then computing STFT amplitude spectrograms, min–max normalize per-sample to [0,1].
    Returns X: (n, 12, 64, 50), y: (n,)
    """
    X = np.zeros((n, *in_shape), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    length = 5100  # to get exactly 50 frames with WIN=200, HOP=100
    for i in range(n):
        c = i % num_classes
        base_f = CLASS_FREQS[c]
        spec = np.zeros((in_shape[0], in_shape[1], in_shape[2]), dtype=np.float32)
        for ch in range(in_shape[0]):
            # small random detuning per channel
            df1 = np.random.uniform(-10, 10)
            df2 = np.random.uniform(-10, 10)
            ph = (np.random.rand()*2*np.pi, np.random.rand()*2*np.pi)
            sig = _sincos_signal(length=length, fs=FS, freqs=(base_f[0]+df1, base_f[1]+df2), phase=ph, noise_std=0.01)
            S = _stft_amplitude(sig)  # (65, 50)
            S = S[:64, :50]  # make (64,50)
            spec[ch] = S
        # per-sample min–max to [0,1]
        mn, mx = spec.min(), spec.max()
        if mx > mn:
            spec = (spec - mn) / (mx - mn)
        X[i] = spec
        y[i] = c
    return X, y


# ----------------------
# Train / Eval
# ----------------------

def train(model, dl_tr: DataLoader, dl_va: DataLoader | None, device, epochs=5, lr=1e-3):
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        model.train()
        loss_sum = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * xb.size(0)
        tr_loss = loss_sum / len(dl_tr.dataset)
        va_acc = float('nan')
        if dl_va is not None:
            model.eval()
            n, acc = 0, 0.0
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb, yb = xb.to(device), yb.to(device)
                    acc += accuracy(model(xb), yb) * xb.size(0)
                    n += xb.size(0)
            va_acc = acc / max(1,n)
        print(f"[Epoch {ep:02d}] train_loss={tr_loss:.4f} val_acc={va_acc:.4f}")


# ----------------------
# ONNX export (STATIC axes)
# ----------------------

def export_static_onnx(model: nn.Module, onnx_path: str, batch_size: int = 1, in_shape=(12,64,50), opset: int = 11):
    model = model.eval().cpu()
    dummy = torch.randn(batch_size, *in_shape, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        opset_version=opset, do_constant_folding=True,
        # dynamic_axes=None  # STATIC
    )
    print(f"[ONNX] exported (STATIC) -> {onnx_path}")


# ----------------------
# TensorRT engine build via trtexec (FP32)
# ----------------------

def has_trtexec() -> bool:
    return shutil.which("trtexec") is not None


def build_trt_engine_trtexec(onnx_path: str, engine_path: str, workspace_mb: int = 1024, extra: list[str] | None = None):
    if not has_trtexec():
        print("[TensorRT] trtexec not found. Skipping engine build.")
        return False
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--explicitBatch",
        f"--workspace={workspace_mb}",
        "--verbose",
        # FP32 by default (no --fp16/--int8)
    ]
    if extra:
        cmd += extra
    print("[TensorRT] Building engine via trtexec:\n  $ " + " ".join(cmd))
    import subprocess
    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    ok = os.path.exists(engine_path)
    print(f"[TensorRT] build {'OK' if ok else 'FAILED'} in {time.time()-t0:.1f}s -> {engine_path if ok else 'N/A'}")
    return ok


# ----------------------
# Optional: TRT Python runtime inference (FP32)
# ----------------------

def trt_python_infer(engine_path: str, x: np.ndarray) -> np.ndarray | None:
    try:
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
    except Exception as e:
        print(f"[TRT-Python] Missing tensorrt/pycuda: {e}")
        return None

    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        print("[TRT-Python] Failed to load engine.")
        return None

    context = engine.create_execution_context()
    # Static shape assumed; bindings: [input, output]
    in_idx, out_idx = 0, 1 if engine.num_bindings > 1 else 0

    # Allocate device buffers
    nbytes_in  = x.nbytes
    import numpy as _np
    d_in  = cuda.mem_alloc(nbytes_in)

    out_shape = tuple(context.get_binding_shape(out_idx))
    out_size  = int(_np.prod(out_shape))
    d_out = cuda.mem_alloc(out_size * _np.dtype(_np.float32).itemsize)

    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_in, x, stream)
    context.execute_async_v2([int(d_in), int(d_out)], stream.handle)
    host_out = _np.empty(out_shape, dtype=_np.float32)
    cuda.memcpy_dtoh_async(host_out, d_out, stream)
    stream.synchronize()
    return host_out


# ----------------------
# Real sample evaluation utilities
# ----------------------

def load_real_samples(folder: str, limit: int = 20) -> np.ndarray:
    """Load up to `limit` .npy/.npz files with arrays shaped (12,64,50) float32 in [0,1].
    Returns array of shape (M,12,64,50) with M<=limit.
    """
    paths = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith((".npy", ".npz")):
            paths.append(os.path.join(folder, name))
    arrs = []
    for p in paths[:limit]:
        if p.endswith(".npy"):
            a = np.load(p)
        else:
            z = np.load(p)
            # try common keys
            if isinstance(z, np.lib.npyio.NpzFile):
                for key in ("arr", "data", "x", "spectrogram"):
                    if key in z.files:
                        a = z[key]
                        break
                else:
                    raise ValueError(f"No known array key in {p}")
            else:
                a = z
        if a.shape != (12,64,50):
            raise ValueError(f"Unexpected shape {a.shape} in {p}, expected (12,64,50)")
        if a.dtype != np.float32:
            a = a.astype(np.float32)
        arrs.append(a)
    if not arrs:
        raise FileNotFoundError(f"No .npy/.npz found in {folder}")
    X = np.stack(arrs, axis=0)
    return X


def eval_on_real_samples(onnx_path: str | None, engine_path: str | None, folder: str, limit: int = 20):
    X = load_real_samples(folder, limit)
    print(f"[RealEval] loaded {X.shape[0]} samples from {folder}")

    # ONNXRuntime path
    if onnx_path is not None:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])  # quick CPU check
            logits = sess.run(["logits"], {"input": X})[0]
            pred = logits.argmax(axis=1)
            print("[RealEval:ONNXRuntime] preds:", pred.tolist())
        except Exception as e:
            print(f"[RealEval:ONNXRuntime] skip: {e}")

    # TensorRT path
    if engine_path is not None and os.path.exists(engine_path):
        outs = []
        for i in range(X.shape[0]):
            y = trt_python_infer(engine_path, X[i:i+1])
            if y is None:
                print("[RealEval:TRT] Python inference unavailable.")
                break
            outs.append(y)
        if outs:
            logits = np.concatenate(outs, axis=0)
            pred = logits.argmax(axis=1)
            print("[RealEval:TRT] preds:", pred.tolist())


# ----------------------
# Main
# ----------------------

def main():
    parser = argparse.ArgumentParser(description="CNN12 -> ONNX(static) -> TensorRT FP32")
    parser.add_argument("--export-dir", type=str, default="./artifacts")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-train", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--trt-python-infer", action="store_true")
    parser.add_argument("--real-samples-dir", type=str, default="", help="Folder with .npy/.npz (12,64,50) to test")
    parser.add_argument("--real-samples-limit", type=int, default=20)
    args = parser.parse_args()

    set_seed(42)
    out_dir = Path(args.export_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path  = out_dir / "cnn12.pt"
    onnx_path  = out_dir / "cnn12_static.onnx"
    engine_path= out_dir / "cnn12_fp32.engine"

    in_shape   = (12,64,50)
    num_classes= 4

    # Data (sin/cos → STFT synthetic). Replace with your real [N,12,64,50] and labels when available.
    X, y = make_sincos_dataset(4000, num_classes, in_shape)
    if train_test_split:
        from sklearn.model_selection import train_test_split as _tts
        Xtr, Xva, ytr, yva = _tts(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        # Simple split without sklearn
        n = int(len(X)*0.8)
        Xtr, Xva, ytr, yva = X[:n], X[n:], y[:n], y[n:]

    dl_tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=args.batch_train, shuffle=True)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)), batch_size=64, shuffle=False)

    model = SmallCNN(in_ch=12, num_classes=num_classes)

    if not args.no_train:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[Train] device:", device)
        train(model, dl_tr, dl_va, device=device, epochs=args.epochs, lr=args.lr)
        torch.save({"model": model.state_dict()}, ckpt_path)
        print(f"[Checkpoint] saved -> {ckpt_path}")
    else:
        ck = args.ckpt or str(ckpt_path)
        sd = torch.load(ck, map_location="cpu")
        model.load_state_dict(sd["model"])
        print(f"[Checkpoint] loaded -> {ck}")

    # Quick sanity (PyTorch)
    model.eval()
    xb = torch.from_numpy(Xva[:1])
    with torch.no_grad():
        probs = torch.softmax(model(xb), dim=1).cpu().numpy()
    print("[PyTorch] sample probs:", np.round(probs, 4))

    # Export ONNX (STATIC, batch=1)
    export_static_onnx(model, str(onnx_path), batch_size=1, in_shape=in_shape, opset=11)

    # Optional: ONNXRuntime check
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])  # CPU quick check
        ort_logits = sess.run(["logits"], {"input": Xva[:1].astype(np.float32)})[0]
        ort_probs = torch.softmax(torch.from_numpy(ort_logits), dim=1).numpy()
        print("[ONNXRuntime] sample probs:", np.round(ort_probs, 4))
    except Exception as e:
        print(f"[ONNXRuntime] skip check: {e}")

    # Build TensorRT engine via trtexec (FP32)
    build_trt_engine_trtexec(str(onnx_path), str(engine_path), workspace_mb=1024)

    # Optional: TRT Python inference
    if args.trt_python_infer and os.path.exists(engine_path):
        x1 = Xva[:1].astype(np.float32)
        out = trt_python_infer(str(engine_path), x1)
        if out is not None:
            import numpy.linalg as LA
            trt_probs = torch.softmax(torch.from_numpy(out), dim=1).numpy()
            print("[TensorRT] sample probs:", np.round(trt_probs, 4))
            # Compare cosine similarity (sanity)
            a = probs.reshape(1,-1)
            b = trt_probs.reshape(1,-1)
            cos = (a*b).sum() / (LA.norm(a)*LA.norm(b) + 1e-8)
            print("[Compare] cosine(PyTorch, TRT):", float(cos))

    # Optional: Evaluate on real samples (user-provided)
    if args.real_samples_dir:
        eval_on_real_samples(str(onnx_path), str(engine_path), args.real_samples_dir, limit=args.real_samples_limit)

    print("Done.")


if __name__ == "__main__":
    main()
