# 1. 환경변수 설정 (Jetson 필수!)
import os
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print(f"Seaborn Version: {sns.__version__}")

# 2. 더미 데이터 생성
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'group': np.random.choice(['A', 'B'], 100)
})

# 3. 그래프 그리기 (산점도)
sns.set_theme(style="darkgrid")
sns.scatterplot(data=data, x='x', y='y', hue='group')

# 4. 저장 (화면이 없는 터미널 환경일 수 있으므로 파일로 저장)
plt.savefig('seaborn_test.png')
print("Test image saved to 'seaborn_test.png'")