#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_all_models.py

Purpose:
  Load .pt, .onnx, and .engine files and compare their outputs 
  to ensure the conversion process was successful.

Usage:
  python3 validate_all_models.py --model-dir ./artifacts
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn

# Import the model definition from your previous file
# (Assumes cnn_12_jetson_fixed.py is in the same folder)
try:
    from cnn_122_onnx_trt_fp_32_TX2 import SmallCNN
except ImportError:
    print("[Error] Could not import 'SmallCNN' from 'cnn_12_jetson_fixed.py'")
    print("Make sure this script is in the same folder as your main code.")
    exit(1)

# ------------------------------------------------------------------------------
# 1. Helper: Softmax for NumPy (to convert logits to probabilities)
# ------------------------------------------------------------------------------
def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# ------------------------------------------------------------------------------
# 2. PyTorch Inference
# ------------------------------------------------------------------------------
def run_pytorch(ckpt_path, input_tensor):
    print(f"\n[PyTorch] Loading {ckpt_path}...")
    
    # Initialize model structure
    model = SmallCNN(in_ch=12, num_classes=4)
    
    # Load weights
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    except Exception as e:
        print(f"[PyTorch] Error loading model: {e}")
        return None

    # Run inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).numpy()
    
    return probs

# ------------------------------------------------------------------------------
# 3. ONNX Inference (using ONNX Runtime)
# ------------------------------------------------------------------------------
def run_onnx(onnx_path, input_numpy):
    print(f"\n[ONNX] Loading {onnx_path}...")
    
    try:
        import onnxruntime as ort
        # Use CPU provider for safety/compatibility
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        
        # Run inference
        logits = sess.run([output_name], {input_name: input_numpy})[0]
        probs = softmax(logits)
        return probs
        
    except ImportError:
        print("[ONNX] 'onnxruntime' not installed. Skipping ONNX check.")
        print("To install: python3 -m pip install onnxruntime")
        return None
    except Exception as e:
        print(f"[ONNX] Error: {e}")
        return None

# ------------------------------------------------------------------------------
# 4. TensorRT Inference (using TRT Python API + PyCUDA)
# ------------------------------------------------------------------------------
def run_tensorrt(engine_path, input_numpy):
    print(f"\n[TensorRT] Loading {engine_path}...")
    
    if not os.path.exists(engine_path):
        print(f"[TensorRT] Engine file not found: {engine_path}")
        return None

    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("[TensorRT] Missing 'tensorrt' or 'pycuda'. Skipping TRT check.")
        return None

    logger = trt.Logger(trt.Logger.WARNING)
    
    try:
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if not engine:
            print("[TensorRT] Failed to deserialize engine.")
            return None

        context = engine.create_execution_context()
        
        # Allocate device memory
        # Note: This assumes 1 input and 1 output for simplicity
        input_shape = input_numpy.shape # (1, 12, 64, 50)
        output_shape = (1, 4)           # 4 classes
        
        h_input = input_numpy
        h_output = np.empty(output_shape, dtype=np.float32)
        
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        stream = cuda.Stream()

        # Inference
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        
        probs = softmax(h_output)
        return probs

    except Exception as e:
        print(f"[TensorRT] Error: {e}")
        return None

# ------------------------------------------------------------------------------
# Main Validation Logic
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Validate .pt, .onnx, .engine files")
    parser.add_argument("--dir", type=str, default="./artifacts", help="Directory containing models")
    args = parser.parse_args()

    # Paths
    # Note: Filenames match those generated by your previous script
    pt_path = os.path.join(args.dir, "cnn12_TX2.pt")
    onnx_path = os.path.join(args.dir, "cnn12_static_TX2.onnx")
    trt_path = os.path.join(args.dir, "cnn12_fp32_TX2.engine")

    # Generate random test input (Batch size 1)
    # Shape: [1, 12, 64, 50]
    print("Generating random test input...")
    dummy_numpy = np.random.randn(1, 12, 64, 50).astype(np.float32)
    dummy_tensor = torch.from_numpy(dummy_numpy)

    # 1. Run PyTorch
    pt_probs = run_pytorch(pt_path, dummy_tensor)
    if pt_probs is None:
        print("Cannot continue without PyTorch baseline.")
        return

    print(f"PyTorch Output: {np.round(pt_probs, 4)}")

    # 2. Run ONNX and Compare
    onnx_probs = run_onnx(onnx_path, dummy_numpy)
    if onnx_probs is not None:
        diff = np.abs(pt_probs - onnx_probs).max()
        print(f"ONNX Output:    {np.round(onnx_probs, 4)}")
        print(f"Max Difference (PT vs ONNX): {diff:.6f}")
        if diff < 1e-4:
            print("✅ ONNX matches PyTorch!")
        else:
            print("❌ ONNX mismatch!")

    # 3. Run TensorRT and Compare
    trt_probs = run_tensorrt(trt_path, dummy_numpy)
    if trt_probs is not None:
        diff = np.abs(pt_probs - trt_probs).max()
        print(f"TRT Output:     {np.round(trt_probs, 4)}")
        print(f"Max Difference (PT vs TRT):  {diff:.6f}")
        if diff < 1e-4:
            print("✅ TensorRT matches PyTorch!")
        else:
            print("❌ TensorRT mismatch!")

if __name__ == "__main__":
    main()