import os
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. 路径管理 (自动识别当前路径，跨平台兼容)
# ==========================================
BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS_DATA = BASE_DIR / "results" / "data"
RESULTS_FIG = BASE_DIR / "results" / "figures"

# 自动创建不存在的文件夹
for p in [DATA_RAW, DATA_PROCESSED, RESULTS_DATA, RESULTS_FIG]:
    p.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. 绘图风格统一设置 (在这里改，全篇生效)
# ==========================================
# ### [自定义区域] 修改字体和配色 ###
plt.rcParams['font.family'] = 'Times New Roman'  # 论文标准字体
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True     # 默认开启网格
plt.rcParams['grid.alpha'] = 0.5     # 网格透明度
plt.rcParams['savefig.dpi'] = 300    # 图片保存分辨率
# 如果有中文需求，解开下面这行：
# plt.rcParams['font.sans-serif'] = ['SimHei'] 

# 定义一组学术配色 (蓝色系，橙色系等)
COLORS = {
    'primary': '#1f77b4',  # 经典蓝
    'secondary': '#ff7f0e', # 经典橙
    'accent': '#2ca02c',    # 经典绿
    'gray': '#7f7f7f'
}