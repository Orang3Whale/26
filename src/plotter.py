import matplotlib.pyplot as plt
import numpy as np
import config
from utils import log, load_result
import seaborn as sns
import os
import pandas as pd
TASK_1_FILE = 'votes_estimation_CI.csv'
OUTPUT_FILE = config.RESULTS_FIG
TARGET_SEASON = 27
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

def check_and_create_dummy_data(filepath):
    """
    检查数据文件是否存在，如果不存在则生成符合结构的演示数据。
    这是为了防止您没有上一步的运行结果而导致报错。
    """
    if os.path.exists(filepath):
        return

    print(f"提示: 未找到 {filepath}，正在生成演示数据以展示绘图效果...")
    seasons = [27, 28] # 选取两个代表性赛季（27为争议赛季，28为回归赛季）
    weeks = range(1, 11)
    data = []
    
    for s in seasons:
        # 模拟选手名
        contestants = [f'Celebrity_{s}_{chr(65+i)}' for i in range(4)]
        for w in weeks:
            # 模拟 S27 后期一致性下降（争议大），S28 保持平稳
            base_consistency = 0.9 if s == 28 else max(0.05, 1.0 - w * 0.12)
            
            # 模拟投票分布 (Dirichlet)
            alpha = np.random.rand(len(contestants)) * 10
            votes = np.random.dirichlet(alpha)
            priors = np.random.dirichlet(alpha * 1.2) # 先验略有不同
            
            for i, name in enumerate(contestants):
                mean = votes[i]
                std = mean * 0.15 # 假设标准差
                
                # 构造一行数据
                row = {
                    'season': s,
                    'week': w,
                    'celebrity_name': name,
                    'vote_mean': mean,
                    'prior_mean': priors[i], # 先验值
                    'vote_std': std,
                    'vote_CI_lower': max(0, mean - 1.96*std), # 95% CI
                    'vote_CI_upper': min(1, mean + 1.96*std),
                    'consistency_acceptance': base_consistency * np.random.uniform(0.8, 1.0),
                    'consistency_mse': (mean - priors[i])**2,
                    'consistency_kl': np.sum(mean * np.log(mean/priors[i] + 1e-9)),
                    'is_eliminated': (i == 0) # 假设第一个人被淘汰
                }
                data.append(row)
    
    df_dummy = pd.DataFrame(data)
    df_dummy.to_csv(filepath, index=False)
    print(f"演示数据已生成: {filepath}")

def plot_voting_results(data_path):
    """
    核心绘图函数：读取分析结果并生成三张关键图表
    """
    print(f"正在读取数据: {data_path} ...")
    data =load_result(data_path)
    df = pd.DataFrame(data)
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid", context="talk")
    
    # ==========================================
    # 图表 1: 一致性趋势分析 (Model Consistency)
    # 作用: 识别哪个赛季、哪一周出现了“不可思议”的结果
    # ==========================================
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df, 
        x='week', 
        y='consistency_acceptance', 
        hue='season', 
        palette='tab10', 
        marker='o',
        linewidth=2.5
    )
    plt.title('Season Consistency Trend (Acceptance Rate)', fontsize=16)
    plt.ylabel('Acceptance Rate (Low = High Controversy)', fontsize=14)
    plt.xlabel('Competition Week', fontsize=14)
    plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE / 'consistency_trend.png', dpi=300)
    plt.close()
    print("✅ 已生成: consistency_trend.png (一致性趋势图)")

    # ==========================================
    # 图表 2: 潜在得票率演变 (Vote Share Evolution)
    # 作用: 展示第一名是如何通过投票建立优势的，包含置信区间
    # ==========================================
    # 选取数据最多的一个赛季进行展示
    target_season = df['season'].value_counts().idxmax()
    season_df = df[df['season'] == target_season]
    
    plt.figure(figsize=(14, 8))
    
    # 绘制均值线
    ax = sns.lineplot(
        data=season_df, 
        x='week', 
        y='vote_mean', 
        hue='celebrity_name', 
        marker='o', 
        palette='viridis',
        linewidth=2
    )
    
    # 手动添加置信区间阴影 (Error Bands)
    unique_celebs = season_df['celebrity_name'].unique()
    # 获取当前调色板颜色
    colors = sns.color_palette('viridis', n_colors=len(unique_celebs))
    
    for i, name in enumerate(unique_celebs):
        subset = season_df[season_df['celebrity_name'] == name].sort_values('week')
        plt.fill_between(
            subset['week'], 
            subset['vote_CI_lower'], 
            subset['vote_CI_upper'], 
            color=colors[i], 
            alpha=0.15 # 透明度
        )
    
    plt.title(f'Estimated Fan Vote Share with 95% CI (Season {target_season})', fontsize=16)
    plt.ylabel('Estimated Vote Share (0-100%)', fontsize=14)
    plt.xlabel('Week', fontsize=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE / f'vote_share_season_{target_season}.png', dpi=300)
    plt.close()
    print(f"✅ 已生成: vote_share_season_{target_season}.png (得票率演变图)")

    # ==========================================
    # 图表 3: 争议度象限分析 (Controversy Quadrant)
    # 作用: 也就是"异常检测"。右上角的点代表极度不合理的淘汰结果。
    # ==========================================
    plt.figure(figsize=(10, 8))
    
    # 绘制散点图
    scatter = sns.scatterplot(
        data=df, 
        x='consistency_mse', 
        y='consistency_kl', 
        hue='season', 
        size='week', 
        sizes=(50, 400), 
        alpha=0.7,
        palette='magma'
    )
    
    plt.title('Controversy Analysis: Distortion (MSE) vs Surprise (KL)', fontsize=16)
    plt.xlabel('MSE Distortion (Posterior vs Prior)', fontsize=14)
    plt.ylabel('KL Divergence (Information Surprise)', fontsize=14)
    
    # 添加辅助线：区分"正常区域"和"争议区域"
    plt.axhline(y=df['consistency_kl'].mean() + df['consistency_kl'].std(), color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=df['consistency_mse'].mean() + df['consistency_mse'].std(), color='red', linestyle='--', alpha=0.5)
    plt.text(df['consistency_mse'].max()*0.8, df['consistency_kl'].max()*0.9, 'High Controversy\nZone', color='red', fontsize=12, ha='center')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE / 'controversy_scatter.png', dpi=300)
    plt.close()
    print("✅ 已生成: controversy_scatter.png (争议度散点图)")

#---------------------------------------------任务二绘图-----------------------------------------------------
TASK2_DATA = config.RESULTS_DATA / 'task2_analysis_results.csv'

def check_data_exists(filepath):
    """简单检查文件是否存在，不存在则提示"""
    if not os.path.exists(filepath):
        print(f"警告: 未找到 {filepath}。请先运行 q2_solver.py 生成分析数据。")
        return False
    return True

def plot_task2(data_path):
    """
    任务二图片绘制代码
    读取赛制对比数据，生成可视化图表
    """
    if not check_data_exists(data_path): return

    print(f"正在读取数据: {data_path} ...")
    df = pd.read_csv(data_path)
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid", context="talk") # 大字体适合论文
    
    # ==========================================
    # 图表 1: 历史还原度 (Accuracy Comparison)
    # 逻辑: 谁能更准确地复现历史(包含人气偏见)，谁就更偏向观众
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    acc_rank = df['rank_correct'].mean()
    acc_pct = df['pct_correct'].mean()
    
    acc_df = pd.DataFrame({
        'System': ['Rank System', 'Percentage System'],
        'Historical Fidelity': [acc_rank, acc_pct]
    })
    
    # 绘制柱状图
    ax = sns.barplot(x='System', y='Historical Fidelity', data=acc_df, palette='viridis')
    plt.ylim(0, 1.1)
    plt.title('Historical Fidelity: Which System Matches Reality?', fontsize=15, pad=20)
    plt.ylabel('Match Rate with Actual Eliminations', fontsize=12)
    plt.xlabel('')
    
    # 标注数值
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1%}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 10), textcoords = 'offset points', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE / 'comparison_accuracy.png', dpi=300)
    print("✅ 图表生成: comparison_accuracy.png")

    # ==========================================
    # 图表 2: 影响力方差比 (Variance Ratio - Log Scale)
    # 逻辑: 比较 Var(Fan)/Var(Judge)，量化"观众权力是评委的多少倍"
    # ==========================================
    # 计算比率
    df['Ratio_Rank'] = df['var_rank_fan'] / (df['var_rank_judge'] + 1e-9)
    df['Ratio_Pct'] = df['var_pct_fan'] / (df['var_pct_judge'] + 1e-9)
    
    # 转换长格式以便绘图
    melted = pd.melt(df, value_vars=['Ratio_Rank', 'Ratio_Pct'], 
                    var_name='System', value_name='Variance Ratio')
    melted['System'] = melted['System'].map({'Ratio_Rank': 'Rank System', 'Ratio_Pct': 'Percentage System'})
    
    plt.figure(figsize=(10, 8))
    
    # 关键: 使用对数坐标，因为百分比制的比率是天文数字
    ax = sns.boxplot(x='System', y='Variance Ratio', data=melted, palette='coolwarm', width=0.5)
    plt.yscale('log') 
    
    plt.title('Magnitude of Fan Bias: Variance Ratio (Fan/Judge)', fontsize=15, pad=20)
    plt.ylabel('Variance Ratio (Log Scale)', fontsize=12)
    plt.xlabel('')
    
    # 添加基准线
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    plt.text(0.5, 1.1, 'Balanced Influence (Ratio=1)', color='black', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE / 'comparison_variance_ratio.png', dpi=300)
    print("✅ 图表生成: comparison_variance_ratio.png")

    # ==========================================
    # 图表 3: 主导权归属 (Dominance Analysis)
    # 逻辑: 系统到底听谁的？听观众的还是听评委的？
    # ==========================================
    # 准备数据
    dom_df = pd.DataFrame({
        'System': ['Rank System', 'Rank System', 'Percentage System', 'Percentage System'],
        'Controller': ['Fan Dominance', 'Judge Dominance', 'Fan Dominance', 'Judge Dominance'],
        'Rate': [
            df['rank_obeys_fan'].mean(), df['rank_obeys_judge'].mean(),
            df['pct_obeys_fan'].mean(), df['pct_obeys_judge'].mean()
        ]
    })
    
    plt.figure(figsize=(12, 7))
    sns.barplot(x='System', y='Rate', hue='Controller', data=dom_df, palette='Set2')
    
    plt.ylim(0, 1.15)
    plt.title('System Control Analysis: Who Decides the Outcome?', fontsize=15, pad=20)
    plt.ylabel('Obedience Rate (Probability)', fontsize=12)
    plt.legend(title='Dominant Party', loc='upper center')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE / 'comparison_dominance.png', dpi=300)
    print("✅ 图表生成: comparison_dominance.png")
    print("所有图表绘制完成。")
#---------------------------------------------任务二绘图结束-------------------------------------------------

if __name__ == "__main__":
    # task1()
    # plot_voting_results(TASK_1_FILE)
    plot_task2(TASK2_DATA)