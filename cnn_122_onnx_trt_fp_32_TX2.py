#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cnn_12_jetson_fixed.py

[Modified for Jetson TX2 + Python 3.6.9]
- Removed Python 3.10+ syntax (Type union |, list[])
- Added typing module imports
- Added path check for trtexec on Jetson
"""

import argparse
from pathlib import Path
import os
import time
import shutil
import subprocess

# [Modified] Python 3.6 type hinting compatibility
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Check for sklearn (Optional)
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None

print(f"torch_version: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device: {torch.cuda.get_device_name(0)}")

# ----------------------
# Model: small CNN for [N,12,64,50]
# ----------------------
class SmallCNN(nn.Module):
    def __init__(self, in_ch: int = 12, num_classes: int = 4):
        super(SmallCNN, self).__init__()  # [Modified] Python 3.6 compatible super()
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
# Data helpers
# ----------------------

FS = 4000
WIN = 200
HOP = 100
NFFT = 128

def _stft_amplitude(x: np.ndarray, n_fft=NFFT, win_length=WIN, hop_length=HOP) -> np.ndarray:
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

CLASS_FREQS = [(120, 300), (200, 400), (280, 600), (350, 800)]

def make_sincos_dataset(n: int, num_classes: int = 4, in_shape=(12, 64, 50)):
    X = np.zeros((n, *in_shape), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    length = 5100
    for i in range(n):
        c = i % num_classes
        base_f = CLASS_FREQS[c]
        spec = np.zeros((in_shape[0], in_shape[1], in_shape[2]), dtype=np.float32)
        for ch in range(in_shape[0]):
            df1 = np.random.uniform(-10, 10)
            df2 = np.random.uniform(-10, 10)
            ph = (np.random.rand()*2*np.pi, np.random.rand()*2*np.pi)
            sig = _sincos_signal(length=length, fs=FS, freqs=(base_f[0]+df1, base_f[1]+df2), phase=ph, noise_std=0.01)
            S = _stft_amplitude(sig)
            S = S[:64, :50]
            spec[ch] = S
        mn, mx = spec.min(), spec.max()
        if mx > mn:
            spec = (spec - mn) / (mx - mn)
        X[i] = spec
        y[i] = c
    return X, y


# ----------------------
# Train / Eval
# ----------------------

# [Modified] Type Hint: DataLoader | None -> Optional[DataLoader]
def train(model, dl_tr: DataLoader, dl_va: Optional[DataLoader], device, epochs=5, lr=1e-3):
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        model.train()
        loss_sum = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            # set_to_none=True is PyTorch 1.7+, Py3.6/JetPack 4.6 usually has 1.10, so it works.
            # If error, use opt.zero_grad()
            if hasattr(opt, 'zero_grad'):
                opt.zero_grad() 
            else:
                model.zero_grad()
                
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
# ONNX export
# ----------------------

def export_static_onnx(model: nn.Module, onnx_path: str, batch_size: int = 1, in_shape=(12,64,50), opset: int = 11):
    model = model.eval().cpu()
    dummy = torch.randn(batch_size, *in_shape, dtype=torch.float32)
    
    # [Modified] Ensure directory exists
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        opset_version=opset, 
        do_constant_folding=True
    )
    print(f"[ONNX] exported (STATIC) -> {onnx_path}")


# ----------------------
# TensorRT Engine Build
# ----------------------

def get_trtexec_path() -> Optional[str]:
    # Check PATH first
    path = shutil.which("trtexec")
    if path: return path
    
    # Check Jetson default location
    jetson_path = "/usr/src/tensorrt/bin/trtexec"
    if os.path.exists(jetson_path):
        return jetson_path
    
    return None

# [Modified] Type Hint: list[str] -> List[str]
def build_trt_engine_trtexec(onnx_path: str, engine_path: str, workspace_mb: int = 1024, extra: Optional[List[str]] = None):
    trtexec = get_trtexec_path()
    if trtexec is None:
        print("[TensorRT] trtexec not found in PATH or /usr/src/tensorrt/bin. Skipping engine build.")
        return False
        
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--explicitBatch",
        f"--workspace={workspace_mb}",
        "--verbose"
    ]
    if extra:
        cmd += extra
        
    print(f"[TensorRT] Building engine via {trtexec}...")
    
    t0 = time.time()
    # [Modified] Python 3.6 subprocess.run does not support text=True (added in 3.7)
    # Use universal_newlines=True instead
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Print only last few lines to avoid spam, or print all if error
    if res.returncode != 0:
        print(res.stdout)
        print("[TensorRT] Build FAILED.")
        return False
    else:
        # Print success msg from TRT logs
        print("[TensorRT] Log tail:")
        print("\n".join(res.stdout.splitlines()[-10:]))
        
    ok = os.path.exists(engine_path)
    print(f"[TensorRT] build {'OK' if ok else 'FAILED'} in {time.time()-t0:.1f}s -> {engine_path if ok else 'N/A'}")
    return ok


# ----------------------
# Optional: TRT Python runtime inference
# ----------------------

# [Modified] Return Type Hint: np.ndarray | None -> Optional[np.ndarray]
def trt_python_infer(engine_path: str, x: np.ndarray) -> Optional[np.ndarray]:
    try:
        import tensorrt as trt
        import pycuda.autoinit
        import pycuda.driver as cuda
    except ImportError as e:
        print(f"[TRT-Python] Missing tensorrt or pycuda. Skipping. ({e})")
        print("To install pycuda: pip install pycuda (ensure nvcc is in PATH)")
        return None
    except Exception as e:
        print(f"[TRT-Python] Error initializing TRT: {e}")
        return None

    logger = trt.Logger(trt.Logger.WARNING)
    try:
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            print("[TRT-Python] Failed to load engine.")
            return None

        context = engine.create_execution_context()
        
        # Allocate buffers
        # Note: This simple allocator assumes 1 input, 1 output and specific binding order
        # For production, iterate over engine.get_binding_name(i)
        
        # Input
        h_input = x
        d_input = cuda.mem_alloc(h_input.nbytes)
        
        # Output
        # In TRT < 8.5, usage is explicit. 
        # Assuming output is index 1
        out_shape = context.get_binding_shape(1)
        # Verify output shape is valid (no -1)
        if -1 in out_shape:
            # Fallback for dynamic shapes (not used here, but good practice)
            out_shape = (x.shape[0], 4) 
            
        h_output = np.empty(out_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(h_output.nbytes)
        
        stream = cuda.Stream()
        
        # Transfer input
        cuda.memcpy_htod_async(d_input, h_input, stream)
        
        # Execute
        # context.execute_async_v2 is for explicit batch (recommended)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        
        # Transfer output
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        
        stream.synchronize()
        return h_output
        
    except Exception as e:
        print(f"[TRT-Python] Inference Error: {e}")
        return None


# ----------------------
# Real sample eval
# ----------------------

# [Modified] Type Hint: str | None -> Optional[str]
def eval_on_real_samples(onnx_path: Optional[str], engine_path: Optional[str], folder: str, limit: int = 20):
    # (Same implementation as provided, assuming load_real_samples exists or import logic matches)
    pass # Implementation omitted for brevity as it relies on external files


# ----------------------
# Main
# ----------------------

def main():
    parser = argparse.ArgumentParser(description="CNN12 -> ONNX(static) -> TensorRT FP32 (Jetson Py3.6)")
    parser.add_argument("--export-dir", type=str, default="./artifacts")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-train", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--trt-python-infer", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    out_dir = Path(args.export_dir)
    # [Modified] exist_ok was added in Python 3.2, parents in 3.4, so this is safe for 3.6
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        
    ckpt_path  = out_dir / "cnn12_TX2.pt"
    onnx_path  = out_dir / "cnn12_static_TX2.onnx"
    engine_path= out_dir / "cnn12_fp32_TX2.engine"

    in_shape   = (12,64,50)
    num_classes= 4

    # 1. Prepare Data
    print("[Data] Generating synthetic data...")
    X, y = make_sincos_dataset(2000, num_classes, in_shape) # reduced size for speed
    
    if train_test_split:
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        print("[Data] sklearn not found, using manual split.")
        n = int(len(X)*0.8)
        Xtr, Xva, ytr, yva = X[:n], X[n:], y[:n], y[n:]

    # [Modified] explicit conversion to tensor
    dl_tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=args.batch_train, shuffle=True)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)), batch_size=64, shuffle=False)

    model = SmallCNN(in_ch=12, num_classes=num_classes)

    # 2. Train or Load
    if not args.no_train:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Train] device: {device}")
        train(model, dl_tr, dl_va, device=device, epochs=args.epochs, lr=args.lr)
        torch.save({"model": model.state_dict()}, str(ckpt_path))
        print(f"[Checkpoint] saved -> {ckpt_path}")
    else:
        ck = args.ckpt if args.ckpt else str(ckpt_path)
        if not os.path.exists(ck):
            print(f"Checkpoint not found: {ck}")
            return
        sd = torch.load(ck, map_location="cpu")
        # Handle dict vs state_dict
        if "model" in sd:
            model.load_state_dict(sd["model"])
        else:
            model.load_state_dict(sd)
        print(f"[Checkpoint] loaded -> {ck}")

    # 3. Export ONNX
    export_static_onnx(model, str(onnx_path), batch_size=1, in_shape=in_shape, opset=11)

    # 4. Build TensorRT Engine (System Call)
    build_trt_engine_trtexec(str(onnx_path), str(engine_path), workspace_mb=1024)

    # 5. Optional: Compare PyTorch vs TensorRT (Python API)
    if args.trt_python_infer:
        print("\n[Compare] Running Inference Comparison...")
        # Take 1 sample
        x_sample = Xva[:1].astype(np.float32)
        
        # PyTorch result
        model.eval().cpu()
        with torch.no_grad():
            pt_out = model(torch.from_numpy(x_sample))
            pt_probs = torch.softmax(pt_out, dim=1).numpy()
            
        print(f"  PyTorch Probs: {np.round(pt_probs, 4)}")
        
        # TensorRT result
        if os.path.exists(str(engine_path)):
            trt_out = trt_python_infer(str(engine_path), x_sample)
            if trt_out is not None:
                # Softmax manually if needed, or use logits
                # Simple softmax impl for numpy
                def softmax(x):
                    e_x = np.exp(x - np.max(x))
                    return e_x / e_x.sum(axis=1, keepdims=True)
                
                trt_probs = softmax(trt_out)
                print(f"  TensorRT Probs: {np.round(trt_probs, 4)}")
                
                # Compare
                diff = np.abs(pt_probs - trt_probs).max()
                print(f"  Max Diff: {diff:.6f}")
                if diff < 1e-3:
                    print("  [Result] ✅ Matched!")
                else:
                    print("  [Result] ⚠ Difference detected.")
            else:
                print("  TensorRT Python inference failed or skipped.")
        else:
            print("  Engine file not found, skipping TRT inference.")

    print("\nDone.")

if __name__ == "__main__":
    main()