#本代码用于比较排名制和百分比制下的表现
import config
from utils import log, save_result
import pandas as pd
import numpy as np
import os
import re
from scipy.stats import rankdata
import scipy.interpolate as interp

# 路径定义 (根据您的要求直接读取 processed 文件)
DATA_ESTIMATION = config.RESULTS_DATA / 'votes_estimation_CI.csv' # 任务1产出的观众投票分布
DATA_WEEKLY = config.DATA_PROCESSED / 'weekly_data_with_popularity.csv' # 包含评委分(weekly_total)
DATA_VIEWERS = config.DATA_RAW / 'viewers.csv' # 收视人数文件
OUTPUT_PATH = config.RESULTS_DATA

# ==========================================
# 1. 辅助功能：生成收视数据
# ==========================================
def generate_viewers_if_missing():
    """如果不存在 viewers.csv，则根据历史数据插值生成"""
    if os.path.exists(DATA_VIEWERS): return
    
    # 关键赛季的收视人数 (单位: 百万)
    anchors = {
        1: 16.8, 2: 17.7, 3: 20.7, 5: 20.0,
        10: 19.0, 15: 14.0, 20: 11.0, 25: 9.0,
        28: 7.0, 30: 6.0, 32: 4.8, 33: 4.9, 34: 7.5
    }
    x = list(anchors.keys())
    y = list(anchors.values())
    
    seasons = np.arange(1, 35)
    # 线性插值
    f = interp.interp1d(x, y, kind='linear', fill_value="extrapolate")
    v_interp = f(seasons)
    
    df = pd.DataFrame({'season': seasons, 'viewers_millions': v_interp})
    df.to_csv(DATA_VIEWERS, index=False)
    print(f"[Info] 已生成收视数据文件: {DATA_VIEWERS}")

# ==========================================
# 2. 核心分析逻辑
# ==========================================
def solve_task_2():
    print("正在启动 任务2 (赛制对比分析)...")
    generate_viewers_if_missing()
    
    # 检查输入文件
    if not os.path.exists(DATA_ESTIMATION) or not os.path.exists(DATA_WEEKLY):
        print(f"错误: 缺少必要文件。\n请确保目录下存在:\n1. {DATA_ESTIMATION} (任务1运行结果)\n2. {DATA_WEEKLY}")
        return

    # 读取数据
    df_votes = pd.read_csv(DATA_ESTIMATION)
    df_weekly = pd.read_csv(DATA_WEEKLY)
    df_viewers = pd.read_csv(DATA_VIEWERS)
    
    # 清理列名空格
    df_weekly.columns = [c.strip() for c in df_weekly.columns]
    
    # 数据合并
    # 我们需要从 df_weekly 中获取评委分 (weekly_total)
    df_full = pd.merge(df_votes, df_weekly[['season', 'week', 'celebrity_name', 'weekly_total']], 
                       on=['season', 'week', 'celebrity_name'], how='inner')
    
    # 合并收视数据 (用于计算估计票数)
    df_full = pd.merge(df_full, df_viewers, on='season', how='left')
    
    # 计算估计票数 (Proxy Vote Count)
    # 假设 vote_mean 是得票百分比，乘以收视人数即为票数
    df_full['estimated_vote_count'] = df_full['vote_mean'] * df_full['viewers'] * 1_000_000

    results = []
    
    # 遍历每一周进行模拟
    for (season, week), group in df_full.groupby(['season', 'week']):
        if len(group) < 2: continue # 至少2人才能比较
        
        # 提取关键向量
        names = group['celebrity_name'].values
        j_scores = group['weekly_total'].values # 评委原始分
        v_means = group['vote_mean'].values # 观众得票率 (百分比)
        
        # ----------------------------------------
        # 模拟 A: 排名制 (Rank System)
        # ----------------------------------------
        # 规则: 评委分越高 -> 排名越好(数值越小 1,2,3)
        r_judge = rankdata(-j_scores, method='min')
        # 规则: 观众票越高 -> 排名越好
        r_fan = rankdata(-v_means, method='min')
        
        # 总排名 = 评委排名 + 观众排名。总和最大者淘汰。
        rank_sum = r_judge + r_fan
        rank_elim_idx = np.argmax(rank_sum)
        rank_elim_name = names[rank_elim_idx]
        
        # 方差分析 (排名制): 直接比较 1-N 排名的方差
        var_rank_judge = np.var(r_judge)
        var_rank_fan = np.var(r_fan)
        
        # ----------------------------------------
        # 模拟 B: 百分比制 (Percentage System)
        # ----------------------------------------
        # 规则: 评委百分比 + 观众百分比。总和最小者淘汰。
        j_sum = np.sum(j_scores)
        p_judge = j_scores / j_sum if j_sum > 0 else np.zeros_like(j_scores)
        p_fan = v_means # 已经是归一化的
        
        pct_sum = p_judge + p_fan
        pct_elim_idx = np.argmin(pct_sum)
        pct_elim_name = names[pct_elim_idx]
        
        # 方差分析 (百分比制): 直接比较 0-1 百分比的方差
        var_pct_judge = np.var(p_judge)
        var_pct_fan = np.var(p_fan)

        # ----------------------------------------
        # 主导权分析 (Dominance Check)
        # ----------------------------------------
        # 谁是观众最想淘汰的人 (得票率最低)?
        fan_target_idx = np.argmin(v_means)
        fan_target_name = names[fan_target_idx]
        # 2. 评委的目标 (分数最低者) -- 之前漏了这个
        judge_target_idx = np.argmin(j_scores)
        judge_target_name = names[judge_target_idx]
        
        # 获取真实淘汰结果 (Ground Truth)
        actual_name = None
        if 'is_eliminated' in group.columns and group['is_eliminated'].sum() > 0:
            actual_name = group[group['is_eliminated']]['celebrity_name'].values[0]
            
        results.append({
            'season': season, 'week': week,
            'actual_elim': actual_name,
            
            # 预测结果
            'rank_pred': rank_elim_name,
            'pct_pred': pct_elim_name,
            
            # 方差指标
            'var_rank_judge': var_rank_judge,
            'var_rank_fan': var_rank_fan,
            'var_pct_judge': var_pct_judge,
            'var_pct_fan': var_pct_fan,
            
            # 是否顺从观众意愿?
            'rank_obeys_fan': (rank_elim_name == fan_target_name),
            'pct_obeys_fan': (pct_elim_name == fan_target_name),
            'rank_obeys_judge': (rank_elim_name == judge_target_name), # 修复 KeyError
            'pct_obeys_judge': (pct_elim_name == judge_target_name),   # 修复 KeyError
            
            # 是否匹配历史?
            'rank_correct': (rank_elim_name == actual_name),
            'pct_correct': (pct_elim_name == actual_name)
        })
        
    df_res = pd.DataFrame(results)
    
    # ==========================================
    # 3. 输出分析报告
    # ==========================================
    print("\n" + "="*60)
    print("任务2: 赛制对比分析报告 (METHOD COMPARISON ANALYSIS)")
    print("="*60)
    
    # A. 方差对比分析
    avg_var_rank_judge = df_res['var_rank_judge'].mean()
    avg_var_rank_fan = df_res['var_rank_fan'].mean()
    avg_var_pct_judge = df_res['var_pct_judge'].mean()
    avg_var_pct_fan = df_res['var_pct_fan'].mean()
    
    print("\n[1] 方差成分分析 (Variance Analysis)")
    print("目的: 比较两种机制下，评委和观众谁的波动(影响力)更大。")
    print("-" * 40)
    print("1. 排名制 (Rank System):")
    print(f"   评委排名方差: {avg_var_rank_judge:.4f}")
    print(f"   观众排名方差: {avg_var_rank_fan:.4f}")
    print(f"   影响力比值 (观众/评委): {avg_var_rank_fan/avg_var_rank_judge:.4f} (接近 1.0 说明势均力敌)")
    
    print("\n2. 百分比制 (Percentage System):")
    print(f"   评委百分比方差: {avg_var_pct_judge:.6f}")
    print(f"   观众百分比方差: {avg_var_pct_fan:.6f}")
    pct_ratio = avg_var_pct_fan/avg_var_pct_judge
    print(f"   影响力比值 (观众/评委): {pct_ratio:.2f} (数值巨大说明观众主导)")
    
    # B. 主导权分析
    fan_dom_rank = df_res['rank_obeys_fan'].mean()
    fan_dom_pct = df_res['pct_obeys_fan'].mean()
    
    print("\n[2] 观众主导率 (Fan Dominance Rate)")
    print("定义: 最终淘汰结果与“观众票数最低者”一致的概率。")
    print("-" * 40)
    print(f"排名制 - 顺从观众概率: {fan_dom_rank:.2%}")
    print(f"百分比制 - 顺从观众概率: {fan_dom_pct:.2%}")
    
    # C. 结论
    print("\n" + "="*60)
    print("最终结论 (CONCLUSION):")
    if pct_ratio > avg_var_rank_fan/avg_var_rank_judge:
        print(">> 【百分比制】显著更偏向于观众投票 (More Biased towards Audience)。")
        print(f"   原因: 在百分比制下，观众投票的方差是评委打分的 {pct_ratio:.1f} 倍。")
        print("   这意味着高人气选手的得票优势可以轻松覆盖评委的分数劣势。")
    else:
        print(">> 【排名制】更偏向于观众投票。")
    print("="*60)
    
    # 保存详细结果
    save_result(df_res,'task2_analysis_results.csv')
    print("\n详细数据已保存至: task2_analysis_results.csv")

if __name__ == "__main__":
    solve_task_2()