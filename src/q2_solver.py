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
DATA_SEASONAL = config.DATA_PROCESSED / 'seasonal_data_with_popularity.csv' # 包含赛季最终名次(final_place)

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
    # 读取赛季最终名次（用于作为真实的当前赛制最终排名）
    if os.path.exists(DATA_SEASONAL):
        df_seasonal = pd.read_csv(DATA_SEASONAL)
        df_seasonal.columns = [c.strip() for c in df_seasonal.columns]
    else:
        df_seasonal = pd.DataFrame(columns=['season', 'celebrity_name', 'final_place'])
    
    # 清理列名空格
    df_weekly.columns = [c.strip() for c in df_weekly.columns]
    
    # 数据合并
    # 我们需要从 df_weekly 中获取评委分 (weekly_total)
    # 包含 weekly_rank 以便使用表中真实名次作为当前赛制的参考
    df_full = pd.merge(df_votes, df_weekly[['season', 'week', 'celebrity_name', 'weekly_total','weekly_rank']], 
                       on=['season', 'week', 'celebrity_name'], how='inner')
    
    # 合并收视数据 (用于计算估计票数)
    df_full = pd.merge(df_full, df_viewers, on='season', how='left')
    
    # 计算估计票数 (Proxy Vote Count)
    # 假设 vote_mean 是得票百分比，乘以收视人数即为票数
    df_full['estimated_vote_count'] = df_full['vote_mean'] * df_full['viewers'] * 1_000_000

    results = []
    # 存放逐人排名的数据
    person_rows = []
    # 赛季层面的聚合，用于计算对立赛制下的赛季最终排名
    # key: (season, celebrity_name) -> {'sum_judge':..., 'sum_est_votes':..., 'weeks':...}
    season_agg = {}
    
    # 遍历每一周进行模拟
    for (season, week), group in df_full.groupby(['season', 'week']):
        if len(group) < 2: continue # 至少2人才能比较
        # 保证索引整齐，便于按位置取值
        group = group.reset_index(drop=True)
        
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
        # ----------------------------------------
        # 逐人排名：计算在两套规则下的最终名次，并标注当前赛季采用的赛制
        # 确定当前赛制: S1-S2 and S28+ 为 Rank System; 其余为 Percentage System
        current_system = 'Percent'
        if season <= 2 or season >= 28:
            current_system = 'Rank'

        # 在排名制下：根据 rank_sum 排序，值越小表示表现越好 -> final_rank_rank: 1 最好
        final_rank_rank = rankdata(rank_sum, method='min')

        # 在百分比制下：pct_sum 值越大表示越好 -> final_rank_pct: 1 最好
        final_rank_pct = rankdata(-pct_sum, method='min')

        # 将逐人结果加入列表
        for i, name in enumerate(names):
            # 优先使用表中真实的 weekly_rank 作为该赛季 "当前赛制" 的名次（如果存在）
            weekly_rank_val = None
            if 'weekly_rank' in group.columns:
                v = group['weekly_rank'].values[i]
                if not pd.isna(v):
                    try:
                        weekly_rank_val = int(v)
                    except Exception:
                        weekly_rank_val = None

            # 对立赛制的排名始终使用我们计算得到的值
            opposing_rank_val = int(final_rank_pct[i]) if current_system == 'Rank' else int(final_rank_rank[i])
            opp_system = 'Percent' if current_system == 'Rank' else 'Rank'

            # 当前赛制名次: 若表中存在真实 weekly_rank 则直接采用，否则回退到计算值
            if weekly_rank_val is not None:
                cur_rank = weekly_rank_val
            else:
                cur_rank = int(final_rank_rank[i]) if current_system == 'Rank' else int(final_rank_pct[i])

            person_rows.append({
                'season': season,
                'week': week,
                'celebrity_name': name,
                'judge_score': float(j_scores[i]),
                'fan_vote_mean': float(v_means[i]),
                'rank_based_rank': int(final_rank_rank[i]),
                'pct_based_rank': int(final_rank_pct[i]),
                'current_system': current_system,
                'current_system_rank': cur_rank,
                'opposing_system': opp_system,
                'opposing_system_rank': opposing_rank_val,
            })
            # 更新赛季聚合
            key = (season, name)
            if key not in season_agg:
                season_agg[key] = {'sum_judge': 0.0, 'sum_est_votes': 0.0, 'weeks': 0}
            season_agg[key]['sum_judge'] += float(j_scores[i])
            # 使用估计票数作为观众票聚合量
            season_agg[key]['sum_est_votes'] += float(group['estimated_vote_count'].values[i])
            season_agg[key]['weeks'] += 1
        
    df_res = pd.DataFrame(results)
    df_person = pd.DataFrame(person_rows)

    # 生成赛季层面的逐人最终排名对比（每赛季每人一条）
    final_rows = []
    # 先把赛季内的选手按season分组
    seasons = set(k[0] for k in season_agg.keys())
    for s in sorted(seasons):
        # 收集该赛季的选手和聚合数据
        keys = [k for k in season_agg.keys() if k[0] == s]
        names_s = [k[1] for k in keys]
        sum_judges = np.array([season_agg[(s, n)]['sum_judge'] for n in names_s], dtype=float)
        sum_fans = np.array([season_agg[(s, n)]['sum_est_votes'] for n in names_s], dtype=float)

        # Rank-system season-level: judge rank (by sum_judges) and fan rank (by sum_fans)
        r_judge_season = rankdata(-sum_judges, method='min')
        r_fan_season = rankdata(-sum_fans, method='min')
        season_rank_sum = r_judge_season + r_fan_season
        final_rank_by_rank_system = rankdata(season_rank_sum, method='min')

        # Percentage-system season-level: compute percentages and sum
        total_j = np.sum(sum_judges)
        total_f = np.sum(sum_fans)
        p_judge_season = sum_judges / total_j if total_j > 0 else np.zeros_like(sum_judges)
        p_fan_season = sum_fans / total_f if total_f > 0 else np.zeros_like(sum_fans)
        pct_sum_season = p_judge_season + p_fan_season
        final_rank_by_pct_system = rankdata(-pct_sum_season, method='min')

        # 当前赛制判断（与逐周相同规则）
        current_system = 'Percent'
        if s <= 2 or s >= 28:
            current_system = 'Rank'

        # 取季赛真实 final_place 作为当前赛制真实名次（若存在）
        seasonal_map = {}
        if not df_seasonal.empty:
            sub = df_seasonal[df_seasonal['season'] == s]
            for _, row in sub.iterrows():
                fp = None
                # 优先使用 final_placement（文件中的列名），兼容 final_place
                if 'final_placement' in row and not pd.isna(row['final_placement']):
                    try:
                        fp = int(row['final_placement'])
                    except Exception:
                        fp = None
                elif 'final_place' in row and not pd.isna(row['final_place']):
                    try:
                        fp = int(row['final_place'])
                    except Exception:
                        fp = None

                seasonal_map[row['celebrity_name']] = fp

        for idx, name in enumerate(names_s):
            # 真实当前赛制名次：优先使用 seasonal.final_place
            real_final = seasonal_map.get(name, None)
            if real_final is None:
                # 回退到我们计算的值
                real_final = int(final_rank_by_rank_system[idx]) if current_system == 'Rank' else int(final_rank_by_pct_system[idx])

            # 对立赛制名次：使用我们计算的赛季级排名
            opposing_final = int(final_rank_by_pct_system[idx]) if current_system == 'Rank' else int(final_rank_by_rank_system[idx])

            final_rows.append({
                'season': s,
                'celebrity_name': name,
                'sum_judge_score': float(sum_judges[idx]),
                'sum_estimated_votes': float(sum_fans[idx]),
                'weeks_count': season_agg[(s, name)]['weeks'],
                'current_system': current_system,
                'current_system_final_place': real_final,
                'opposing_system_final_place': opposing_final,
                'final_rank_if_rank_system': int(final_rank_by_rank_system[idx]),
                'final_rank_if_pct_system': int(final_rank_by_pct_system[idx]),
            })

    df_person_final = pd.DataFrame(final_rows)
    
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
    # 保存逐周逐人排名信息（历史周级记录）
    save_result(df_person, 'task2_person_weekly_rankings.csv')
    print("逐周逐人数据已保存至: task2_person_weekly_rankings.csv")
    # 保存赛季层面的逐人最终排名对比（每赛季每人一条）
    save_result(df_person_final, 'task2_person_rankings.csv')
    print("赛季层面逐人最终排名已保存至: task2_person_rankings.csv")


if __name__ == "__main__":
    solve_task_2()