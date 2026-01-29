import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
import config
from utils import log, save_result

# ==========================================
# 1. 定义模型 (微分方程 / 算法逻辑)
# ==========================================
def model_equation(y, t, params):
    """
    ### [自定义区域] 在这里定义你的微分方程或核心算法 ###
    示例：简单的SIR模型 (传染病/信息传播)
    y: 状态向量 [S, I, R]
    t: 时间
    params: 参数包
    """
    S, I, R = y
    beta, gamma = params
    
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    
    return [dSdt, dIdt, dRdt]

# ==========================================
# 2. 主求解流程
# ==========================================
def run_simulation():
    log("开始模型计算...")
    
    # ### [自定义区域] 设置初始条件和参数 ###
    t = np.linspace(0, 100, 1000)  # 时间轴
    y0 = [0.99, 0.01, 0.0]         # 初始状态
    params = (0.3, 0.1)            # beta, gamma
    
    # 调用求解器 (可以是 odeint, 也可以是你手写的算法)
    solution = odeint(model_equation, y0, t, args=(params,))
    
    # 整理结果
    results = {
        'time': t,
        'S': solution[:, 0],
        'I': solution[:, 1],
        'R': solution[:, 2],
        'params': params
    }
    
    log("计算完成，正在保存...")
    
    # 关键一步：保存结果，供 plotter 使用
    # 建议使用 .pkl 保存包含字典的复杂结果，或 .csv 保存纯表格
    save_result(results, 'simulation_result_v1.pkl') 

if __name__ == "__main__":
    run_simulation()