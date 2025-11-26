 #!/usr/bin/env python

"""

Jetson에서 onnxruntime 없이

PyTorch(.pt) vs ONNX(.onnx) 차이를 PyTorch만으로 검증하는 스크립트.


- cnn12.pt   → SmallCNN (원본 PyTorch 모델)

- cnn12_static.onnx → onnx2pytorch로 다시 PyTorch 모델로 변환

- 같은 랜덤 입력 x에 대해 두 모델의 출력 차이를 비교

"""


from pathlib import Path

import numpy as np

import torch

import onnx

from onnx2pytorch import ConvertModel



def load_pytorch_model(ckpt_path: Path):

    """같은 폴더의 cnn_12_onnx_trt_fp_32.py에서 SmallCNN 불러와 .pt 로드"""

    import cnn_12_onnx_trt_fp_32 as m


    print(f"[PyTorch] checkpoint load: {ckpt_path}")

    ck = torch.load(ckpt_path, map_location="cpu")


    # ck가 {'model': state_dict, ...} 또는 곧바로 state_dict일 수 있음

    if isinstance(ck, dict) and "model" in ck:

        state = ck["model"]

    else:

        state = ck


    model = m.SmallCNN(in_ch=12, num_classes=4)

    model.load_state_dict(state)

    model.eval().cpu()

    return model



def load_onnx_as_pytorch(onnx_path: Path):

    """ONNX를 onnx2pytorch로 다시 PyTorch 모델로 변환"""

    print(f"[ONNX] load: {onnx_path}")

    onnx_model = onnx.load(str(onnx_path))

    # 구조 검사

    onnx.checker.check_model(onnx_model)

    print("[ONNX] checker: OK (model structure valid)")


    print("[ONNX2PyTorch] converting ONNX → PyTorch...")

    model_pt = ConvertModel(onnx_model)  # 기본 옵션 사용

    model_pt.eval().cpu()

    return model_pt



def main():

    base_dir = Path(__file__).resolve().parent

    ckpt_path = base_dir / "cnn12.pt"

    onnx_path = base_dir / "cnn12_static.onnx"


    if not ckpt_path.exists():

        raise FileNotFoundError(f".pt 파일을 찾을 수 없습니다: {ckpt_path}")

    if not onnx_path.exists():

        raise FileNotFoundError(f".onnx 파일을 찾을 수 없습니다: {onnx_path}")


    print("\n--- 1. PyTorch 원본 모델 로드 ---")

    model_orig = load_pytorch_model(ckpt_path)


    print("\n--- 2. ONNX → PyTorch 변환 모델 로드 ---")

    model_from_onnx = load_onnx_as_pytorch(onnx_path)


    # 랜덤 입력 (학습 코드와 동일한 shape 가정: [N, 12, 64, 50])

    batch_size = 1

    x_np = np.random.randn(batch_size, 12, 64, 50).astype(np.float32)

    x_t = torch.from_numpy(x_np)


    print("\n--- 3. 두 모델에 대해 추론 수행 (CPU) ---")

    with torch.no_grad():

        logits_orig = model_orig(x_t).cpu().numpy()

        logits_onnx = model_from_onnx(x_t).cpu().numpy()


        probs_orig = torch.softmax(torch.from_numpy(logits_orig), dim=1).numpy()

        probs_onnx = torch.softmax(torch.from_numpy(logits_onnx), dim=1).numpy()


    print("[Orig PyTorch] logits shape:", logits_orig.shape)

    print("[From ONNX ]  logits shape:", logits_onnx.shape)

    print("[Orig PyTorch] sample probs:", np.round(probs_orig[0], 4))

    print("[From ONNX ]  sample probs:", np.round(probs_onnx[0], 4))


    # 차이 계산

    diff = probs_orig - probs_onnx

    mae = float(np.mean(np.abs(diff)))

    max_diff = float(np.max(np.abs(diff)))


    a = probs_orig.reshape(batch_size, -1)

    b = probs_onnx.reshape(batch_size, -1)

    num = float((a * b).sum())

    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    cos_sim = num / denom


    print("\n--- 4. PyTorch vs ONNX(Pytorch 변환) 비교 ---")

    print("  Mean abs diff   :", mae)

    print("  Max  abs diff   :", max_diff)

    print("  Cosine similarity:", cos_sim)


    if mae < 1e-3 and cos_sim > 0.999:

        print("\n[Result] ✅ ONNX가 PyTorch와 거의 완전히 일치합니다. (Jetson, CPU 기준 검증 OK)")

    else:

        print("\n[Result] ⚠ 차이가 다소 큽니다. export 과정이나 연산자 지원 여부를 다시 체크해보세요.")



if __name__ == "__main__":

    main() 