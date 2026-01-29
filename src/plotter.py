import matplotlib.pyplot as plt
import numpy as np
import config
from utils import log, load_result

def plot_simulation_results():
    log("正在加载数据用于绘图...")
    
    # 1. 读取 Solver 算好的数据
    # ### [自定义区域] 确保文件名与 solver.py 中保存的一致 ###
    data = load_result('simulation_result_v1.pkl')
    
    t = data['time']
    S = data['S']
    I = data['I']
    R = data['R']
    
    # 2. 开始绘图
    plt.figure(figsize=(10, 6))
    
    # ### [自定义区域] 绘图逻辑 ###
    plt.plot(t, S, label='Susceptible', color=config.COLORS['primary'], linewidth=2)
    plt.plot(t, I, label='Infected', color=config.COLORS['secondary'], linewidth=2)
    plt.plot(t, R, label='Recovered', color=config.COLORS['accent'], linewidth=2)
    
    # 3. 美化与标注
    plt.title('SIR Model Simulation', fontsize=16, pad=15)
    plt.xlabel('Time (days)', fontsize=14)
    plt.ylabel('Proportion of Population', fontsize=14)
    plt.legend(frameon=True, shadow=True)
    plt.xlim(0, 100)
    plt.ylim(0, 1.1)
    
    # 4. 保存图片
    # ### [自定义区域] 修改输出图片的文件名 ###
    output_path = config.RESULTS_FIG / 'sir_model_trajectory.png'
    plt.savefig(output_path, bbox_inches='tight')
    log(f"图片已保存: {output_path}")
    
    # 可选：显示图片
    # plt.show()

if __name__ == "__main__":
    plot_simulation_results()