#!/usr/bin/env python
"""
cnn12.pt & cnn12_static.onnx 검증 스크립트 (Jetson, 같은 폴더 기준)

- cnn_12_onnx_trt_fp_32.py 안의 SmallCNN을 불러서 .pt 로드
- 랜덤 입력으로 PyTorch 출력 확인
- ONNX 모델 구조 검사
- ONNX Runtime으로 같은 입력에 대해 출력 계산
- 둘의 차이를 수치로 비교 (MAE, max diff, cosine similarity)
"""

from pathlib import Path

import numpy as np
import torch
import onnx


def load_pytorch_model(ckpt_path: Path):
    """같은 폴더에 있는 cnn_12_onnx_trt_fp_32.py에서 SmallCNN을 가져와 .pt 로드"""
    import cnn_12_onnx_trt_fp_32 as m  # 같은 디렉토리에 있다고 가정

    print(f"[PyTorch] checkpoint load: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu")

    # ck가 {'model': state_dict, ...} 형태일 수도 있고, 곧바로 state_dict일 수도 있음
    if isinstance(ck, dict) and "model" in ck:
        state = ck["model"]
    else:
        state = ck

    # 이 설정은 cnn_12_onnx_trt_fp_32.py에서 사용한 것과 동일해야 함
    model = m.SmallCNN(in_ch=12, num_classes=4)
    model.load_state_dict(state)
    model.eval().cpu()

    return model


def main():
    base_dir = Path(__file__).resolve().parent
    ckpt_path = base_dir / "cnn12.pt"
    onnx_path = base_dir / "cnn12_static.onnx"

    if not ckpt_path.exists():
        raise FileNotFoundError(f".pt 파일을 찾을 수 없습니다: {ckpt_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f".onnx 파일을 찾을 수 없습니다: {onnx_path}")

    # -----------------------------
    # 1) PyTorch 모델 로드 + 추론
    # -----------------------------
    model = load_pytorch_model(ckpt_path)

    # cnn_12_onnx_trt_fp_32.py에서 사용한 입력 크기와 동일하게 맞춤
    batch_size = 1
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

    # 입력/출력 이름 자동 추출
    graph = onnx_model.graph
    input_name = graph.input[0].name
    output_name = graph.output[0].name
    print(f"[ONNX] input name : {input_name}")
    print(f"[ONNX] output name: {output_name}")

    # -----------------------------
    # 3) ONNX Runtime로 추론
    # -----------------------------
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ONNXRuntime] onnxruntime이 설치되어 있지 않습니다.")
        print("  → `pip install onnxruntime` 후 다시 실행하세요.")
        return

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],  # Jetson에서는 CPU로만 테스트
    )

    ort_logits = sess.run([output_name], {input_name: x_np})[0]
    probs_ort = torch.softmax(torch.from_numpy(ort_logits), dim=1).numpy()

    print("[ONNXRuntime] logits shape:", ort_logits.shape)
    print("[ONNXRuntime] sample probs:", np.round(probs_ort[0], 4))

    # -----------------------------
    # 4) PyTorch vs ONNX 결과 비교
    # -----------------------------
    diff = probs_pt - probs_ort
    mae = float(np.mean(np.abs(diff)))
    max_diff = float(np.max(np.abs(diff)))

    a = probs_pt.reshape(batch_size, -1)
    b = probs_ort.reshape(batch_size, -1)
    num = float((a * b).sum())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    cos_sim = num / denom

    print("\n[Compare PyTorch vs ONNX]")
    print("  Mean abs diff   :", mae)
    print("  Max  abs diff   :", max_diff)
    print("  Cosine similarity:", cos_sim)

    if mae < 1e-3 and cos_sim > 0.999:
        print("\n[Result] ✅ ONNX가 PyTorch와 거의 완전히 일치합니다. (정상)")
    else:
        print("\n[Result] ⚠ 차이가 다소 큽니다. export 옵션이나 모델/입출력 형태를 다시 확인해보세요.")


if __name__ == "__main__":
    main()
