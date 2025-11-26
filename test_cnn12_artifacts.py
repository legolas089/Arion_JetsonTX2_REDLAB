#!/usr/bin/env python
"""
cnn12.pt & cnn12_static.onnx 검증 스크립트 (Jetson용, CPU 기준)

- PyTorch .pt 로드 → 랜덤 입력에 대한 출력 확인
- ONNX 모델 구조 검사 (onnx.checker)
- ONNX Runtime으로 같은 입력에 대한 출력 확인
- PyTorch vs ONNX 출력 차이(MAE, Max diff, cosine similarity) 계산
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import onnx


def load_pytorch_model(ckpt_path: Path):
    """
    cnn_12_onnx_trt_fp_32.py에 정의된 SmallCNN을 불러와서
    ckpt(.pt)의 state_dict를 로드합니다.
    """
    from cnn_12_onnx_trt_fp_32 import SmallCNN  # 같은 폴더에 있다고 가정

    print(f"[PyTorch] checkpoint load: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck["model"] if "model" in ck else ck

    model = SmallCNN(in_ch=12, num_classes=4)
    model.load_state_dict(state)
    model.eval().cpu()
    return model


def test_pt_and_onnx(ckpt_path: Path, onnx_path: Path, batch_size: int = 1):
    # -----------------------------
    # 1) PyTorch 모델 로드
    # -----------------------------
    model = load_pytorch_model(ckpt_path)

    # 랜덤 입력 생성 (학습 코드와 동일한 shape: [N, 12, 64, 50])
    x_np = np.random.randn(batch_size, 12, 64, 50).astype(np.float32)
    x_t = torch.from_numpy(x_np)

    with torch.no_grad():
        logits_pt = model(x_t).cpu().numpy()
        probs_pt = torch.softmax(torch.from_numpy(logits_pt), dim=1).numpy()

    print("[PyTorch] logits shape:", logits_pt.shape)
    print("[PyTorch] sample probs:", np.round(probs_pt[0], 4))

    # -----------------------------
    # 2) ONNX 구조 검증
    # -----------------------------
    print(f"[ONNX] load & check: {onnx_path}")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("[ONNX] checker: OK (model structure valid)")

    # -----------------------------
    # 3) ONNX Runtime로 추론 비교
    # -----------------------------
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ONNXRuntime] onnxruntime이 설치되어 있지 않습니다. `pip install onnxruntime` 후 다시 실행하세요.")
        return

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"]  # Jetson에서 CPU로만 테스트
    )

    ort_logits = sess.run(["logits"], {"input": x_np})[0]
    probs_ort = torch.softmax(torch.from_numpy(ort_logits), dim=1).numpy()

    print("[ONNXRuntime] logits shape:", ort_logits.shape)
    print("[ONNXRuntime] sample probs:", np.round(probs_ort[0], 4))

    # -----------------------------
    # 4) PyTorch vs ONNX 결과 비교
    # -----------------------------
    diff = probs_pt - probs_ort
    mae = float(np.mean(np.abs(diff)))
    max_diff = float(np.max(np.abs(diff)))

    # cosine similarity
    a = probs_pt.reshape(batch_size, -1)
    b = probs_ort.reshape(batch_size, -1)
    num = float((a * b).sum())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    cos_sim = num / denom

    print("\n[Compare PyTorch vs ONNX]")
    print("  Mean abs diff :", mae)
    print("  Max  abs diff :", max_diff)
    print("  Cosine similarity:", cos_sim)

    if mae < 1e-3 and cos_sim > 0.999:
        print("\n[Result] ONNX가 PyTorch와 거의 완전히 일치합니다. (정상)")
    else:
        print("\n[Result] 차이가 다소 큽니다. export 옵션이나 모델을 다시 확인해보세요.")


def main():
    p = argparse.ArgumentParser(description="cnn12.pt & cnn12_static.onnx 검증 스크립트")
    p.add_argument(
        "--export-dir",
        type=str,
        default="./artifacts",
        help="cnn_12_onnx_trt_fp_32.py에서 사용한 export 디렉토리 (기본: ./artifacts)",
    )
    p.add_argument(
        "--ckpt-name",
        type=str,
        default="cnn12.pt",
        help="체크포인트 파일 이름 (기본: cnn12.pt)",
    )
    p.add_argument(
        "--onnx-name",
        type=str,
        default="cnn12_static.onnx",
        help="ONNX 파일 이름 (기본: cnn12_static.onnx)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="랜덤 테스트 배치 크기 (기본: 1)",
    )
    args = p.parse_args()

    export_dir = Path(args.export_dir)
    ckpt_path = export_dir / args.ckpt_name
    onnx_path = export_dir / args.onnx_name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트(.pt)를 찾을 수 없습니다: {ckpt_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX 파일을 찾을 수 없습니다: {onnx_path}")

    test_pt_and_onnx(ckpt_path, onnx_path, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
