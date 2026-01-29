import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import config

# ==========================================
# 日志打印 (带时间戳，方便排查程序卡在哪)
# ==========================================
def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# ==========================================
# 数据存取 (支持 pickle 和 csv)
# ==========================================
def save_result(data, filename):
    """保存计算结果到 results/data"""
    filepath = config.RESULTS_DATA / filename
    
    # ### [自定义区域] 根据数据类型自动选择保存方式 ###
    if filename.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif filename.endswith('.csv') and isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
    elif filename.endswith('.npy'):
        np.save(filepath, data)
    
    log(f"结果已保存: {filepath}")

def load_result(filename):
    """从 results/data 读取结果"""
    filepath = config.RESULTS_DATA / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件未找到: {filepath}")
        
    if filename.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif filename.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filename.endswith('.npy'):
        return np.load(filepath)