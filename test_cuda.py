# import os
# # Jetson TX2 OpenBLAS 충돌 방지 코드
# os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

import numpy as np
import torch
print("torch_version:",torch.__version__)
print("pytorch build in cuda:",torch.version.cuda)

print("cuda available:",torch.cuda.is_available())

print(np.random.normal(0,1,(3,3)))
print("cuda Check")

