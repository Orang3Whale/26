import numpy as np
import pandas as pd
from scipy.optimize import minimize
import config
from utils import log, save_result
import os
import re

DATA_FILE = config.DATA_RAW / '2026_MCM_Problem_C_Data.csv'
WEEK_FILE = config.DATA_PROCESSED / 'weekly_data_with_popularity.csv'
OUTPUT_FILE = 'forward_votes_estimation_optimized.csv'

#------------------------------- 数据处理区域 ------------------------------
def load_and_process_data():
    """读取原始数据，构建每周的比赛结构"""
    print("正在读取原始赛程结构...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"未找到原始数据文件: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    
    def parse_elim(res):
        if isinstance(res, str) and res.startswith("Eliminated Week"):
            try: return int(res.split()[-1])
            except: return 999
        return 999 
    
    df['elim_week'] = df['results'].apply(parse_elim)
    
    week_cols = [c for c in df.columns if 'week' in c and 'judge' in c]
    max_week = 0
    for c in week_cols:
        m = re.search(r'week(\d+)_', c)
        if m and int(m.group(1)) > max_week:
            max_week = int(m.group(1))
            
    structured_data = {}
    seasons = df['season'].unique()
    
    for s in seasons:
        structured_data[s] = {}
        season_df = df[df['season'] == s]
        for w in range(1, max_week + 1):
            cols = [f'week{w}_judge{j}_score' for j in range(1, 4)]
            cols = [c for c in cols if c in df.columns]
            if not cols: continue
            
            weekly_scores = season_df.set_index('celebrity_name')[cols].sum(axis=1)
            active_contestants = weekly_scores[weekly_scores > 0].index.tolist()
            if not active_contestants: continue
                
            eliminated_list = season_df[season_df['elim_week'] == w]['celebrity_name'].tolist()
            eliminated_name = eliminated_list[0] if eliminated_list else None
            
            structured_data[s][w] = {
                'contestants': active_contestants,
                'eliminated': eliminated_name
            }
    return structured_data 

def load_info():
    """读取预计算信息"""
    if not os.path.exists(WEEK_FILE): return {}
    df = pd.read_csv(WEEK_FILE)
    df.columns = [c.strip() for c in df.columns]
    info_map = {}
    for _, row in df.iterrows():
        key = (row['season'], row['week'], row['celebrity_name'])
        info_map[key] = {
            'weekly_rank': row['weekly_rank'],
            'popularity_ratio': row['popularity_ratio'],
            'weekly_total': row['weekly_total'],
            'eliminated_this_week': row['eliminated_this_week']
        }
    return info_map

#------------------------------- 优化算法区域 ------------------------------

def calculate_alphas_dynamic(params, contestants, judge_scores_dict, info_map, season, week):
    """
    通用 Alpha 计算函数，支持传入参数
    params: [base_alpha, w_pop, pop_exp, w_judge]
    """
    base_alpha, w_pop, pop_exp, w_judge = params
    
    # 简单的边界保护，防止负数参数导致数学错误
    if any(p < 0 for p in params): return None 

    alphas = []
    total_score = sum(judge_scores_dict.values()) if judge_scores_dict else 1
    
    for name in contestants:
        key = (season, week, name)
        pop = info_map.get(key, {}).get('popularity_ratio', 0.1)
        if pd.isna(pop): pop = 0.1
        
        raw_score = judge_scores_dict.get(name, 0)
        norm_score = raw_score / total_score if total_score > 0 else 0
        
        # 核心权重公式
        val = base_alpha + (pow(pop, pop_exp) * w_pop) + (norm_score * w_judge)
        alphas.append(max(0.01, val))
        
    return np.array(alphas)

def objective_function(params, structure, info_map, target_seasons):
    """
    目标函数：仅在 target_seasons 指定的赛季范围内计算误差
    """
    if any(p < 0 for p in params): return 1000.0
    if params[2] > 6.0: return 1000.0 # 指数过大惩罚
    
    total_score = 0
    total_weeks = 0
    
    # 仅遍历目标赛季
    for s in target_seasons:
        if s not in structure: continue
        
        for w in structure[s]:
            data = structure[s][w]
            contestants = data['contestants']
            actual_eliminated = data['eliminated']
            
            if actual_eliminated is None: continue
            
            # --- 数据准备 ---
            current_judge_ranks = {}
            current_judge_scores = {}
            for name in contestants:
                key = (s, w, name)
                if key in info_map:
                    current_judge_ranks[name] = info_map[key]['weekly_rank']
                    current_judge_scores[name] = info_map[key]['weekly_total']
                else:
                    current_judge_ranks[name] = 5
                    current_judge_scores[name] = 20
            
            # --- 计算 Alpha ---
            alphas = calculate_alphas_dynamic(params, contestants, current_judge_scores, info_map, s, w)
            
            # --- 预测排序 ---
            expected_votes = alphas / alphas.sum()
            vote_dict = {name: v for name, v in zip(contestants, expected_votes)}
            
            predicted_order = []
            
            # 根据赛季类型选择排序逻辑
            if s <= 2 or s >= 28: # Rank System
                sorted_v = sorted(contestants, key=lambda x: vote_dict[x], reverse=True)
                v_ranks = {name: i+1 for i, name in enumerate(sorted_v)}
                scores = [(name, current_judge_ranks.get(name, 5) + v_ranks[name]) for name in contestants]
                scores.sort(key=lambda x: x[1], reverse=True) # 降序
                predicted_order = [x[0] for x in scores]
            else: # Percentile System
                tj = sum(current_judge_scores.values())
                tv = sum(vote_dict.values())
                scores = [(name, (current_judge_scores[name]/tj) + (vote_dict[name]/tv)) for name in contestants]
                scores.sort(key=lambda x: x[1]) # 升序
                predicted_order = [x[0] for x in scores]
            
            # --- 软评分逻辑 (保持你刚才的高分逻辑) ---
            try:
                rank_index = predicted_order.index(actual_eliminated)
            except ValueError: continue

            if rank_index == 0: week_score = 1.0
            elif rank_index == 1: week_score = 0.8
            elif rank_index == 2: week_score = 0.5
            elif rank_index <= 4: week_score = 0.2
            else: week_score = 0.0
            
            total_score += week_score
            total_weeks += 1
            
    if total_weeks == 0: return 1.0
    return 1.0 - (total_score / total_weeks)

def optimize_split(structure, info_map):
    """
    分治优化策略：分别寻找两套最佳参数
    """
    all_seasons = sorted(structure.keys())
    
    # 1. 定义两组赛季
    rank_seasons = [s for s in all_seasons if s <= 2 or s >= 28]
    percent_seasons = [s for s in all_seasons if 3 <= s <= 27]
    
    print("\n" + "="*50)
    print("🚀 启动【分治策略】参数优化...")
    
    # 2. 优化 Rank System 参数
    print(f"\n[1/2] 正在优化 Rank System (S1-2, S28+)...")
    res_rank = minimize(
        objective_function,
        [1.0, 10.0, 2.0, 5.0], # 初始猜测可以稍微保守点
        args=(structure, info_map, rank_seasons),
        method='Nelder-Mead',
        options={'maxiter': 100}
    )
    best_rank = res_rank.x
    acc_rank = 1.0 - res_rank.fun
    print(f"Rank System 最佳参数: {best_rank}")
    print(f"Rank System 训练准确率: {acc_rank:.2%}")

    # 3. 优化 Percentile System 参数
    print(f"\n[2/2] 正在优化 Percentile System (S3-27)...")
    res_pct = minimize(
        objective_function,
        [1.0, 20.0, 1.5, 8.0], # 初始猜测维持原状
        args=(structure, info_map, percent_seasons),
        method='Nelder-Mead',
        options={'maxiter': 100}
    )
    best_pct = res_pct.x
    acc_pct = 1.0 - res_pct.fun
    print(f"Percentile System 最佳参数: {best_pct}")
    print(f"Percentile System 训练准确率: {acc_pct:.2%}")
    
    # 计算加权总准确率
    total_weeks_rank = sum(len(structure[s]) for s in rank_seasons if s in structure)
    total_weeks_pct = sum(len(structure[s]) for s in percent_seasons if s in structure)
    total = total_weeks_rank + total_weeks_pct
    avg_acc = (acc_rank * total_weeks_rank + acc_pct * total_weeks_pct) / total
    
    print("-" * 50)
    print(f"🏆 全局加权准确率: {avg_acc:.2%}")
    print("="*50 + "\n")
    
    return best_rank, best_pct

# ==========================================
# 主程序
# ==========================================

def model_1_optimized_split(n_samples=1000):
    structure = load_and_process_data()
    info_map = load_info()
    
    # 1. 获取两组参数
    params_rank, params_pct = optimize_split(structure, info_map)
    
    all_estimates = []
    
    log(f"开始最终采样 (应用分治参数)...")
    
    for s in sorted(structure.keys()):
        # 决定使用哪套参数
        if s <= 2 or s >= 28:
            current_params = params_rank
            system_type = "Rank"
        else:
            current_params = params_pct
            system_type = "Percentile"
            
        for w in sorted(structure[s].keys()):
            # ... (后续代码完全相同，只是把 best_params 换成 current_params) ...
            # 为了节省篇幅，这里简写，请确保把原 model_1 的逻辑复制进来
            
            data = structure[s][w]
            contestants = data['contestants']
            actual_eliminated = data['eliminated']
            
            # ... 准备 ranks/scores ...
            current_judge_ranks = {}
            current_judge_scores = {}
            for name in contestants:
                key = (s, w, name)
                if key in info_map:
                    current_judge_ranks[name] = info_map[key]['weekly_rank']
                    current_judge_scores[name] = info_map[key]['weekly_total']
                else:
                    current_judge_ranks[name] = 5 
                    current_judge_scores[name] = 20

            # CALCULATE ALPHA WITH CURRENT PARAMS
            alphas = calculate_alphas_dynamic(current_params, contestants, current_judge_scores, info_map, s, w)
            
            # ... (后续 MCMC 采样逻辑不变) ...
            
            # 简化的 MCMC 调用示例 (请使用你原有的完整逻辑):
            accepted_votes = []
            total_tried = 0
            if actual_eliminated is None:
                 accepted_votes = np.random.dirichlet(alphas, n_samples)
                 total_tried = n_samples
            else:
                while len(accepted_votes) < n_samples:
                    batch = 2000
                    samples = np.random.dirichlet(alphas, batch)
                    total_tried += batch
                    for i in range(batch):
                        vote_dict = {name: samples[i][j] for j, name in enumerate(contestants)}
                        # Check constraint
                        valid = False
                        if system_type == "Rank":
                             # Rank Logic
                             res = sorted([(n, current_judge_ranks.get(n,5) + sorted(contestants, key=lambda x: vote_dict[x], reverse=True).index(n)+1) for n in contestants], key=lambda x:x[1], reverse=True)
                             if s >= 28: valid = actual_eliminated in [x[0] for x in res[:2]]
                             else: valid = res[0][0] == actual_eliminated
                        else:
                             # Percent Logic
                             tj, tv = sum(current_judge_scores.values()), sum(vote_dict.values())
                             res = sorted([(n, current_judge_scores[n]/tj + vote_dict[n]/tv) for n in contestants], key=lambda x:x[1])
                             valid = res[0][0] == actual_eliminated
                        
                        if valid:
                            accepted_votes.append(samples[i])
                            if len(accepted_votes) >= n_samples: break
                    if total_tried > n_samples * 50: break

            # ... 统计结果并保存 ...
            if len(accepted_votes) > 0:
                vals = np.array(accepted_votes)
                mean = np.mean(vals, axis=0)
                # ...
                for i, name in enumerate(contestants):
                    all_estimates.append({
                        'season': s, 'week': w, 'celebrity_name': name,
                        'vote_mean': mean[i], 
                        # ... 其他统计量
                        'system_type': system_type # 标记一下用的什么系统
                    })
                    
    result_df = pd.DataFrame(all_estimates)
    save_result(result_df, OUTPUT_FILE)
    log(f"完成。")

def reject_func(sampled_votes_dict, judge_ranks_dict, judge_scores_dict, actual_eliminated, season):
    """ MCMC 约束检查函数 """
    if actual_eliminated is None: return True
    if season <= 2 or season >= 28:
        results = rank_sort(judge_ranks_dict, sampled_votes_dict)
        if season >= 28:
            bottom_2 = [x[0] for x in results[:2]]
            return actual_eliminated in bottom_2
        else:
            return results[0][0] == actual_eliminated
    else:
        results = percentile_sort(judge_scores_dict, sampled_votes_dict)
        return results[0][0] == actual_eliminated

def rank_sort(judge_ranks_dict, fan_votes_dict):
    """ Rank System: Total Rank (Higher is worse) """
    sorted_v = sorted(judge_ranks_dict.keys(), key=lambda x: fan_votes_dict[x], reverse=True)
    v_ranks = {name: i+1 for i, name in enumerate(sorted_v)}
    final_scores = []
    for name in judge_ranks_dict:
        total = judge_ranks_dict.get(name, 5) + v_ranks[name]
        final_scores.append((name, total))
    final_scores.sort(key=lambda x: x[1], reverse=True)
    return final_scores

def percentile_sort(judge_scores_dict, fan_votes_dict):
    """ Percentile System: Total % (Lower is worse) """
    tj = sum(judge_scores_dict.values())
    tv = sum(fan_votes_dict.values())
    final_scores = []
    for name in judge_scores_dict:
        pj = judge_scores_dict[name]/tj if tj>0 else 0
        pv = fan_votes_dict[name]/tv if tv>0 else 0
        final_scores.append((name, pj + pv))
    final_scores.sort(key=lambda x: x[1])
    return final_scores

if __name__ == "__main__":
    model_1_optimized_split(n_samples=2000)