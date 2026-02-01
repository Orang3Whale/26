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

def plot_rank_diff():
    """绘制不同机制下的排名变化"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 文件路径
    file = 'data/results/integrated_seasonal_data_with_hypothetical_ranks.csv'
    
    # 检查文件是否存在
    try:
        df = pd.read_csv(file)
        print(f"数据加载成功，形状: {df.shape}")
        print(f"可用列名: {list(df.columns)}")
    except FileNotFoundError:
        print(f"数据文件不存在: {file}")
        print("请先运行生成假设排名数据的代码")
        return
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
    
    # 检查必要的列是否存在
    required_columns = ['actual_rank', 'hypothetical_rank_1', 'hypothetical_rank_2']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"警告: 数据文件中缺少以下列: {missing_columns}")
        print("将使用示例数据进行演示")
        # 创建示例数据用于演示
        df = create_sample_rank_data()
    
    # 图表1: 排名变化散点图
    plt.figure(figsize=(12, 10))
    
    # 子图1: 实际排名 vs 假设排名1
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=df, x='actual_rank', y='hypothetical_rank_1', alpha=0.7)
    plt.plot([df['actual_rank'].min(), df['actual_rank'].max()], 
             [df['actual_rank'].min(), df['actual_rank'].max()], 'r--', alpha=0.8)
    plt.xlabel('实际排名')
    plt.ylabel('假设排名 (机制1)')
    plt.title('实际排名 vs 假设排名1')
    
    # 子图2: 实际排名 vs 假设排名2
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='actual_rank', y='hypothetical_rank_2', alpha=0.7)
    plt.plot([df['actual_rank'].min(), df['actual_rank'].max()], 
             [df['actual_rank'].min(), df['actual_rank'].max()], 'r--', alpha=0.8)
    plt.xlabel('实际排名')
    plt.ylabel('假设排名 (机制2)')
    plt.title('实际排名 vs 假设排名2')
    
    # 子图3: 排名差异分布
    plt.subplot(2, 2, 3)
    df['rank_diff_1'] = df['hypothetical_rank_1'] - df['actual_rank']
    df['rank_diff_2'] = df['hypothetical_rank_2'] - df['actual_rank']
    
    plt.hist(df['rank_diff_1'], alpha=0.7, label='机制1差异', bins=20)
    plt.hist(df['rank_diff_2'], alpha=0.7, label='机制2差异', bins=20)
    plt.xlabel('排名变化 (假设排名 - 实际排名)')
    plt.ylabel('频次')
    plt.title('排名变化分布')
    plt.legend()
    
    # 子图4: 排名变化箱线图
    plt.subplot(2, 2, 4)
    diff_data = pd.DataFrame({
        '差异值': pd.concat([df['rank_diff_1'], df['rank_diff_2']]),
        '机制': ['机制1'] * len(df) + ['机制2'] * len(df)
    })
    sns.boxplot(data=diff_data, x='机制', y='差异值')
    plt.title('排名变化箱线图')
    plt.ylabel('排名变化')
    
    plt.tight_layout()
    plt.savefig('output/rank_comparison_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 图表2: 排名变化热力图
    plt.figure(figsize=(10, 8))
    
    # 创建排名变化矩阵
    rank_changes = pd.crosstab(df['actual_rank'], df['hypothetical_rank_1'])
    sns.heatmap(rank_changes, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('实际排名到假设排名1的变化热力图')
    plt.xlabel('假设排名1')
    plt.ylabel('实际排名')
    plt.tight_layout()
    plt.savefig('output/rank_change_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 图表3: 排名稳定性分析
    plt.figure(figsize=(12, 6))
    
    # 计算每个选手的排名变化绝对值
    df['abs_diff_1'] = abs(df['rank_diff_1'])
    df['abs_diff_2'] = abs(df['rank_diff_2'])
    
    stability_data = pd.DataFrame({
        '变化幅度': pd.concat([df['abs_diff_1'], df['abs_diff_2']]),
        '机制': ['机制1'] * len(df) + ['机制2'] * len(df),
        '实际排名': pd.concat([df['actual_rank'], df['actual_rank']])
    })
    
    sns.boxplot(data=stability_data, x='实际排名', y='变化幅度', hue='机制')
    plt.title('不同实际排名下的排名变化稳定性')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/rank_stability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出统计信息
    print("\n排名变化统计分析:")
    print(f"机制1平均变化: {df['rank_diff_1'].mean():.2f} ± {df['rank_diff_1'].std():.2f}")
    print(f"机制2平均变化: {df['rank_diff_2'].mean():.2f} ± {df['rank_diff_2'].std():.2f}")
    print(f"机制1最大提升: {df['rank_diff_1'].min():.0f} 名")
    print(f"机制1最大下降: {df['rank_diff_1'].max():.0f} 名")
    print(f"机制2最大提升: {df['rank_diff_2'].min():.0f} 名")
    print(f"机制2最大下降: {df['rank_diff_2'].max():.0f} 名")
    
    print("\n图表生成完成！")

def create_sample_rank_data():
    """创建示例排名数据用于演示"""
    import numpy as np
    np.random.seed(42)
    
    n_samples = 100
    actual_ranks = np.random.randint(1, 11, n_samples)
    
    # 创建有偏的假设排名
    hypothetical_1 = actual_ranks + np.random.normal(0, 1.5, n_samples)
    hypothetical_2 = actual_ranks + np.random.normal(0.5, 2, n_samples)
    
    # 确保排名在合理范围内
    hypothetical_1 = np.clip(np.round(hypothetical_1), 1, 10)
    hypothetical_2 = np.clip(np.round(hypothetical_2), 1, 10)
    
    df = pd.DataFrame({
        'celebrity_name': [f'选手_{i}' for i in range(n_samples)],
        'season': np.random.randint(1, 6, n_samples),
        'actual_rank': actual_ranks,
        'hypothetical_rank_1': hypothetical_1,
        'hypothetical_rank_2': hypothetical_2
    })
    
    return df

def plot_ranking_comparison(file_path):
    # 1. 读取数据
    try:
        df = pd.read_csv(file_path)
        print(f"成功读取数据，共 {len(df)} 行。")
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return

    # 2. 设置绘图风格 (学术论文风格)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 创建 2x2 的画布
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Comparison of Ranking Mechanisms: Actual vs Hypothetical', fontsize=20, fontweight='bold', y=0.98)

    # ---------------------------------------------------------
    # 图表 1: 实际名次 vs 假设名次 (散点图)
    # ---------------------------------------------------------
    sns.scatterplot(
        data=df,
        x='final_placement',
        y='hypothetical_rank',
        hue='system_type',      # 不同颜色代表不同原始赛制
        style='system_type',    # 不同形状区分
        s=100,                  # 点的大小
        alpha=0.7,              # 透明度
        ax=axes[0, 0],
        palette='deep'
    )
    
    # 添加对角线 (名次不变线)
    max_rank = max(df['final_placement'].max(), df['hypothetical_rank'].max())
    axes[0, 0].plot([0, max_rank], [0, max_rank], 'r--', linewidth=2, label='No Change Line')
    
    axes[0, 0].set_title('Actual Placement vs. Hypothetical Rank', fontsize=16)
    axes[0, 0].set_xlabel('Actual Placement (Lower is Better)', fontsize=14)
    axes[0, 0].set_ylabel('Hypothetical Rank (Lower is Better)', fontsize=14)
    axes[0, 0].legend(title='Original System')
    axes[0, 0].text(1, max_rank-2, "Points Below Line = Improved in Hypothetical\nPoints Above Line = Worsened in Hypothetical", 
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    # ---------------------------------------------------------
    # 图表 2: 名次差异分布 (直方图)
    # ---------------------------------------------------------
    # 计算差异: 正值代表假设名次更好 (例如实际10 - 假设5 = +5)
    # 注意：这里我们用 rank_difference 列，假设 csv 里已经是 (实际 - 假设)
    # 如果 CSV 里 rank_difference 是负数代表变差，请根据实际数据调整解释
    
    sns.histplot(
        data=df,
        x='rank_difference',
        hue='system_type',
        kde=True,               # 显示密度曲线
        bins=20,
        ax=axes[0, 1],
        palette='deep',
        edgecolor='white'
    )
    axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Distribution of Rank Impact', fontsize=16)
    axes[0, 1].set_xlabel('Rank Difference (Actual - Hypothetical)\nPositive = Hypothetical System is Better for Contestant', fontsize=14)
    axes[0, 1].set_ylabel('Number of Contestants', fontsize=14)

    # ---------------------------------------------------------
    # 图表 3: 假设赛制下的最大受益者 Top 10 (条形图)
    # ---------------------------------------------------------
    # 筛选出差异最大的正值 (受益最大)
    top_improved = df.nlargest(10, 'rank_difference')
    
    sns.barplot(
        data=top_improved,
        x='rank_difference',
        y='celebrity_name',
        hue='system_type',
        dodge=False,
        ax=axes[1, 0],
        palette='viridis'       # 绿色系代表受益
    )
    axes[1, 0].set_title('Top 10 Beneficiaries of Hypothetical System', fontsize=16)
    axes[1, 0].set_xlabel('Positions Gained (Rank Improvement)', fontsize=14)
    axes[1, 0].set_ylabel('')
    axes[1, 0].legend(title='Original System')

    # ---------------------------------------------------------
    # 图表 4: 假设赛制下的最大受害者 Top 10 (条形图)
    # ---------------------------------------------------------
    # 筛选出差异最小的负值 (受损最大)
    top_worsened = df.nsmallest(10, 'rank_difference')
    # 为了绘图好看，取绝对值显示长度，但保留负号逻辑
    top_worsened['abs_diff'] = top_worsened['rank_difference'].abs()
    
    sns.barplot(
        data=top_worsened,
        x='rank_difference',    # 这里显示负值
        y='celebrity_name',
        hue='system_type',
        dodge=False,
        ax=axes[1, 1],
        palette='magma'         # 红色系代表受损
    )
    axes[1, 1].set_title('Top 10 "Victims" of Hypothetical System', fontsize=16)
    axes[1, 1].set_xlabel('Positions Lost (Rank Decline)', fontsize=14)
    axes[1, 1].set_ylabel('')
    axes[1, 1].axvline(0, color='black', linewidth=1)
    axes[1, 1].legend(title='Original System')

    # 3. 调整布局并保存
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 留出标题空间
    output_filename = config.RESULTS_FIG / 'ranking_comparison_analysis.png'
    plt.savefig(output_filename, dpi=300)
    print(f"图表已生成并保存为: {output_filename}")
    # plt.show() # 如果在本地运行，可以取消注释查看窗口

def correlations():
    # 1. 读取数据
    df_raw = pd.read_csv(config.DATA_RAW / '2026_MCM_Problem_C_Data.csv')
    df_partner = pd.read_csv(config.DATA_PROCESSED / 'q3_partner_characteristics.csv')
    df_ind = pd.read_csv(config.DATA_PROCESSED / 'industry.csv')
    df_reg = pd.read_csv(config.DATA_PROCESSED / 'region.csv')

    # 2. 数据重命名与预处理
    # 为了清晰，将统计表中的 avg_placement 重命名，表明它是该行业/地区的平均表现
    df_partner = df_partner.rename(columns={
        'appearance_count': 'partner_appearance_count',
        'avg_placement': 'partner_avg_placement',
        'avg_judge_score': 'partner_avg_judge_score',
        'avg_elim_week': 'partner_avg_elim_week',
        'champion_count': 'partner_champion_count'
    })
    df_ind = df_ind.rename(columns={'avg_placement': 'industry_avg_placement', 'industry': 'celebrity_industry'})
    df_reg = df_reg.rename(columns={'avg_placement': 'region_avg_placement', 'region': 'celebrity_homestate'})

    # 3. 数据合并 (以"对/Couple"为单位)
    # 将舞伴特征、明星行业特征、明星地区特征合并到原始比赛数据中
    df_merged = pd.merge(df_raw, df_partner, left_on='ballroom_partner', right_on='partner_name', how='left')
    df_merged = pd.merge(df_merged, df_ind[['celebrity_industry', 'industry_avg_placement']], on='celebrity_industry', how='left')
    df_merged = pd.merge(df_merged, df_reg[['celebrity_homestate', 'region_avg_placement']], on='celebrity_homestate', how='left')

    # 4. 选择要计算相关性的变量
    cols_to_corr = [
        'partner_appearance_count',
        'partner_avg_placement',
        'partner_avg_judge_score',
        'partner_avg_elim_week',
        'partner_champion_count',
        'industry_avg_placement', # 代表该明星所属行业的平均水平
        'region_avg_placement',    # 代表该明星所属地区的平均水平
        'celebrity_age_during_season' # New Feature
    ]

    # 5. 计算斯皮尔曼相关系数 (Spearman Correlation)
    # 使用 Spearman 是因为排名等数据属于序数数据，非正态分布
    corr_matrix = df_merged[cols_to_corr].corr(method='spearman')

    # 6. 绘制热度图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.grid(False)
    plt.title('Spearman Correlation Heatmap: Partner & Celebrity Characteristics')
    # 设置横轴标签倾斜45度，避免重叠
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(config.RESULTS_FIG / 'correlation_heatmap.png')
    log("相关性图已生成")

def plot_rank_difference_distribution(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    # 按照排名差异排序，以形成瀑布流/S形曲线效果，美观且易读
    df_sorted = df.sort_values('rank_difference').reset_index(drop=True)
    
    # 创建索引用于X轴绘图
    df_sorted['contestant_index'] = df_sorted.index

    # 设置画布
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")

    # 绘制散点图
    # X轴: 排序后的选手索引
    # Y轴: 排名差异
    sns.scatterplot(
        data=df_sorted,
        x='contestant_index',
        y='rank_difference',
        hue='system_type',
        style='system_type',
        palette='coolwarm', # 冷暖色调适合表现正负差异
        s=60,
        alpha=0.8,
        edgecolor='k' 
    )

    # 添加0基准线
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='No Change')

    # 添加解释性文字
    max_diff = df['rank_difference'].max()
    min_diff = df['rank_difference'].min()
    
    plt.text(10, max_diff - 0.5, "Positive: Better Rank in Hypothetical System", 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), verticalalignment='top')
    plt.text(10, min_diff + 0.5, "Negative: Worse Rank in Hypothetical System", 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), verticalalignment='bottom')

    # 标题和标签
    plt.title('Distribution of Rank Differences for All 421 Contestants', fontsize=16)
    plt.xlabel('Contestants (Sorted by Rank Difference)', fontsize=12)
    plt.ylabel('Rank Difference (Actual - Hypothetical)', fontsize=12)
    plt.legend(title='Original System Type')

    plt.tight_layout()
    plt.savefig(config.RESULTS_FIG / 'rank_difference_waterfall.png')
    print("Plot saved to rank_difference_waterfall.png")
if __name__ == "__main__":
    # task1()
    # plot_voting_results(TASK_1_FILE)
    # plot_task2(TASK2_DATA)
    # plot_ranking_comparison(r'D:\NUAA\2026\MCM\26\results\data\integrated_seasonal_data_with_hypothetical_ranks.csv')
    correlations()
    # plot_rank_difference_distribution(config.RESULTS_DATA / 'integrated_seasonal_data_with_hypothetical_ranks.csv')