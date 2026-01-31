import matplotlib.pyplot as plt
import numpy as np
import config
from utils import log, load_result
import seaborn as sns
import os
import pandas as pd
TASK_1_FILE = 'votes_estimation_CI.csv'
OUTPUT_FILE = config.RESULTS_FIG
TARGET_SEASON = 1
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

def plot_fan_vote_trends(df, season):
    """
    图表 1: 观众投票趋势图 (带 95% 置信区间)
    展示每个选手每周的得票率变化，阴影表示不确定性范围。
    """
    plt.figure(figsize=(14, 8))
    
    # 获取该赛季的所有选手
    contestants = df['celebrity_name'].unique()
    
    # 为每个选手分配颜色
    palette = sns.color_palette("tab20", len(contestants))
    color_map = dict(zip(contestants, palette))
    
    # 绘制每个选手的曲线
    for name in contestants:
        subset = df[df['celebrity_name'] == name].sort_values('week')
        
        # 绘制均值线
        plt.plot(subset['week'], subset['vote_mean'], label=name, 
                 color=color_map[name], linewidth=2.5, marker='o', markersize=4)
        
        # 绘制置信区间 (Ribbon)
        plt.fill_between(subset['week'], 
                         subset['vote_CI_lower'], 
                         subset['vote_CI_upper'], 
                         color=color_map[name], alpha=0.15)
        
        # 标记淘汰点
        elim_point = subset[subset['is_eliminated']]
        if not elim_point.empty:
            plt.scatter(elim_point['week'], elim_point['vote_mean'], 
                        color='red', s=100, zorder=5, marker='X', edgecolors='white')

    plt.title(f'Season {season}: Estimated Fan Vote Share Over Time (with 95% CI)', fontsize=16, fontweight='bold')
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Estimated Vote Share (0-1)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Celebrity')
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE / f'S{season}_vote_trends.png', dpi=300)
    print(f"图表已保存: S{season}_vote_trends.png")
    plt.close()

def plot_judge_vs_fan(df, season):
    """
    图表 2: 评委分 vs 观众票 散点图
    分析“叫好”与“叫座”的关系。红色 X 代表该周被淘汰。
    """
    plt.figure(figsize=(10, 8))
    
    # 归一化评委分 (为了和得票率在同一尺度对比，或者直接用原始分)
    # 这里我们用原始分，但在颜色映射上区分
    
    # 绘制散点
    # 使用 hue 表示是否被淘汰
    sns.scatterplot(data=df, x='judge_score', y='vote_mean', 
                    hue='is_eliminated', style='is_eliminated',
                    palette={False: 'royalblue', True: 'red'},
                    markers={False: 'o', True: 'X'}, s=100, alpha=0.7)
    
    # 添加标签 (只标记极值点或淘汰点，防止太乱)
    for idx, row in df.iterrows():
        if row['is_eliminated'] or row['vote_mean'] > 0.25: # 标记淘汰者和高人气者
            plt.text(row['judge_score']+0.2, row['vote_mean'], 
                     f"{row['celebrity_name']} (W{row['week']})", 
                     fontsize=8, alpha=0.8)

    plt.title(f'Season {season}: Judge Score vs. Estimated Fan Votes', fontsize=16)
    plt.xlabel('Weekly Judge Score', fontsize=12)
    plt.ylabel('Estimated Fan Vote Share', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE / f'S{season}_judge_vs_fan.png', dpi=300)
    print(f"图表已保存: S{season}_judge_vs_fan.png")
    plt.close()

def plot_vote_composition(df, season):
    """
    图表 3: 得票构成堆叠图 (Stacked Area Chart)
    展示每周总票仓是如何在幸存选手之间分配的。
    """
    # 数据透视: 行=Week, 列=Name, 值=Vote_Mean
    pivot_df = df.pivot(index='week', columns='celebrity_name', values='vote_mean')
    
    # 填充NaN为0 (已淘汰)
    pivot_df = pivot_df.fillna(0)
    
    plt.figure(figsize=(14, 8))
    pivot_df.plot.area(stacked=True, cmap='tab20', alpha=0.85, figsize=(14, 8))
    
    plt.title(f'Season {season}: Evolution of Vote Composition', fontsize=16)
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Cumulative Vote Share', fontsize=12)
    plt.margins(0, 0) # 去除空白边缘
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Celebrity')
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE / f'S{season}_vote_composition.png', dpi=300)
    print(f"图表已保存: S{season}_vote_composition.png")
    plt.close()

def task1():
    data = load_result(TASK_1_FILE)

    full_df = pd.DataFrame(data)
    # 读取数据
    # full_df = pd.read_csv(TASK_1_FILE)
    
    # 过滤特定赛季
    df = full_df[full_df['season'] == TARGET_SEASON].copy()
    
    if df.empty:
        print(f"警告: 数据中没有 Season {TARGET_SEASON} 的记录。")
        return
        
    print(f"正在为 Season {TARGET_SEASON} 生成图表...")
    
    # 生成三个图表
    plot_fan_vote_trends(df, TARGET_SEASON)
    plot_judge_vs_fan(df, TARGET_SEASON)
    plot_vote_composition(df, TARGET_SEASON)
    
    print("所有图表生成完毕！")
if __name__ == "__main__":
    task1()