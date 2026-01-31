import pandas as pd
import numpy as np
import os
from scipy.stats import rankdata
import config
from utils import log, save_result

# 路径定义
DATA_ESTIMATION = config.RESULTS_DATA / 'votes_estimation_CI.csv'
WEEKLY_FILE = config.DATA_PROCESSED / 'weekly_data_with_popularity.csv'
VIEWERS_FILE = config.DATA_RAW / 'viewers.csv'
OUTPUT_PATH = config.RESULTS_DATA

def load_merged_data():
    """读取并合并所需数据"""
    if not os.path.exists(DATA_ESTIMATION) or not os.path.exists(WEEKLY_FILE):
        print("缺少数据文件")
        return None
        
    df_votes = pd.read_csv(DATA_ESTIMATION)
    df_weekly = pd.read_csv(WEEKLY_FILE)
    df_weekly.columns = [c.strip() for c in df_weekly.columns] # 清理列名
    
    # 读取收视数据(如果有)
    if os.path.exists(VIEWERS_FILE):
        df_viewers = pd.read_csv(VIEWERS_FILE)
        df_merged = pd.merge(df_votes, df_weekly[['season', 'week', 'celebrity_name', 'weekly_total']], 
                             on=['season', 'week', 'celebrity_name'], how='inner')
        df_merged = pd.merge(df_merged, df_viewers, on='season', how='left')
    else:
        df_merged = pd.merge(df_votes, df_weekly[['season', 'week', 'celebrity_name', 'weekly_total']], 
                             on=['season', 'week', 'celebrity_name'], how='inner')
    return df_merged

def counterfactual_simulation():
    """
    【新增核心功能】争议案例的反事实模拟
    目的：回答"如果S27用的是排名制，Bobby Bones会不会被淘汰？"
    """
    log("\n启动反事实模拟 (Counterfactual Simulation)...")
    df = load_merged_data()
    if df is None: return

    counterfactual_results = []
    
    # 遍历每一周
    for (season, week), group in df.groupby(['season', 'week']):
        if len(group) < 2: continue
        
        # 基础数据
        names = group['celebrity_name'].values
        j_scores = group['weekly_total'].values
        v_means = group['vote_mean'].values
        
        # 1. 确定该周的"实际"淘汰者
        actual_elim = None
        if 'is_eliminated' in group.columns and group['is_eliminated'].sum() > 0:
            actual_elim = group[group['is_eliminated']]['celebrity_name'].values[0]
            
        # 2. 找到"评委最讨厌的人" (Judge's Target)
        # 如果这个人没有被淘汰，他就是"争议幸存者"
        judge_min_idx = np.argmin(j_scores)
        judge_target = names[judge_min_idx]
        
        # 3. 计算两种机制下的淘汰者
        # --- 机制 A: 排名制 ---
        r_judge = rankdata(-j_scores, method='min')
        r_fan = rankdata(-v_means, method='min')
        rank_sum = r_judge + r_fan
        # 排名制淘汰者: Rank Sum 最大者 (数值最大=表现最差)
        # 平局处理：观众排名更差的走
        max_val = np.max(rank_sum)
        cands = np.where(rank_sum == max_val)[0]
        if len(cands) > 1:
            tie_break = np.argmax(r_fan[cands])
            rank_elim_idx = cands[tie_break]
        else:
            rank_elim_idx = cands[0]
        rank_elim = names[rank_elim_idx]
        
        # --- 机制 B: 百分比制 ---
        j_sum = np.sum(j_scores)
        p_judge = j_scores / j_sum if j_sum > 0 else 0
        p_fan = v_means
        pct_sum = p_judge + p_fan
        # 百分比制淘汰者: Sum 最小者
        pct_elim_idx = np.argmin(pct_sum)
        pct_elim = names[pct_elim_idx]
        
        # 4. 判断当前赛季的真实赛制
        # S1-S2, S28+ 是排名制；S3-S27 是百分比制
        current_system = "Percent"
        if season <= 2 or season >= 28:
            current_system = "Rank"
            
        # 5. 核心逻辑：检测"反事实翻转" (Counterfactual Flip)
        # 我们只关心那些"评委想杀，但实际活下来了"的人
        if judge_target != actual_elim and actual_elim is not None:
            
            # 如果是 Bobby Bones 这种 (S27, Percent System)
            if current_system == "Percent":
                # 真实情况：百分比制救了他 (Pct Safe)
                # 反事实：如果用排名制，他会死吗？
                if rank_elim == judge_target:
                    counterfactual_results.append({
                        'season': season,
                        'week': week,
                        'contestant': judge_target,
                        'scenario': 'Saved by Percent, Killed by Rank',
                        'judge_score': j_scores[judge_min_idx],
                        'fan_vote_share': f"{v_means[judge_min_idx]:.1%}",
                        'rank_system_result': 'Eliminated',
                        'percent_system_result': 'Safe (Actual)'
                    })
            
            # 如果是 Jerry Rice 这种 (S2, Rank System) -> 其实S2就是Rank制，但他活了
            # 那我们看看如果用百分比制，他会不会死得更透？或者也能活？
            elif current_system == "Rank":
                # 真实情况：排名制救了他？(Rank Safe)
                # 反事实：如果用百分比制...
                if pct_elim == judge_target:
                    counterfactual_results.append({
                        'season': season,
                        'week': week,
                        'contestant': judge_target,
                        'scenario': 'Saved by Rank, Killed by Percent',
                        'judge_score': j_scores[judge_min_idx],
                        'fan_vote_share': f"{v_means[judge_min_idx]:.1%}",
                        'rank_system_result': 'Safe (Actual)',
                        'percent_system_result': 'Eliminated'
                    })

    # === 输出分析结果 ===
    df_cf = pd.DataFrame(counterfactual_results)
    
    if len(df_cf) > 0:
        log("\n" + "="*60)
        log("【重磅发现】赛制对立模拟结果 (Counterfactual Analysis)")
        log("="*60)
        
        # 1. 重点分析 S27 Bobby Bones 现象
        bobby = df_cf[df_cf['contestant'].str.contains("Bobby", case=False)]
        if not bobby.empty:
            log("\n>> 案例分析: Bobby Bones (S27)")
            log("模拟显示：如果在 S27 采用排名制 (Rank System)，Bobby Bones 将在以下周次被淘汰：")
            log(bobby[['season', 'week', 'fan_vote_share', 'rank_system_result']].to_string(index=False))
            log("结论：百分比制确实是 Bobby Bones 夺冠的'帮凶'。")
        
        # 2. 统计哪种翻转更常见
        type_a = df_cf[df_cf['scenario'] == 'Saved by Percent, Killed by Rank']
        type_b = df_cf[df_cf['scenario'] == 'Saved by Rank, Killed by Percent']
        
        log(f"\n>> 统计数据:")
        log(f"类型 A (靠百分比制幸存，但在排名制下会死): 共发现 {len(type_a)} 例")
        log(f"类型 B (靠排名制幸存，但在百分比制下会死): 共发现 {len(type_b)} 例")
        
        if len(type_a) > len(type_b):
            log("\n>> 最终推论:")
            log("数据表明，'类型 A' 更加普遍。")
            log("这意味着【百分比制】更容易产生'低分高人气'的幸存者。")
            log("排名制对低分选手的惩罚更重（Rank Last 是致命的），而百分比制允许用海量票数填补分差。")
            
        save_path =  'counterfactual_analysis.csv'
        save_result(df_cf, save_path)
        log(f"\n详细反事实模拟表已保存至: {save_path}")
    else:
        log("未发现明显的反事实翻转案例（这可能意味着两种赛制在多数情况下结果一致，或者数据不足）。")

if __name__ == "__main__":
    counterfactual_simulation()