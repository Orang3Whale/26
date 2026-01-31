import numpy as np
import pandas as pd
from scipy.optimize import minimize
import config
from utils import log, save_result
import os
import re

DATA_FILE = config.DATA_RAW / '2026_MCM_Problem_C_Data.csv'
WEEK_FILE = config.DATA_PROCESSED / 'weekly_data_with_popularity.csv'
OUTPUT_FILE = 'votes_estimation_final.csv'

# ==========================================
# 1. 数据加载模块
# ==========================================
def load_and_process_data():
    """读取原始数据，构建赛程结构"""
    if not os.path.exists(DATA_FILE): return {}
    df = pd.read_csv(DATA_FILE)
    
    def parse_elim(res):
        if isinstance(res, str) and res.startswith("Eliminated Week"):
            try: return int(res.split()[-1])
            except: return 999
        return 999 
    
    df['elim_week'] = df['results'].apply(parse_elim)
    
    # 识别最大周数
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
    """读取预计算的每周辅助信息（人气、排名等）"""
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

# ==========================================
# 2. 核心数学逻辑
# ==========================================

def calculate_alphas(params, contestants, judge_scores_dict, info_map, season, week):
    """
    计算 Dirichlet 分布的参数 Alpha (投票潜力)
    Alpha = Base + (Pop^Exp * W_Pop) + (Norm_Score * W_Judge)
    """
    base_alpha, w_pop, pop_exp, w_judge = params
    
    # === 新增：温度系数 (Temperature) ===
    # T > 1.0: 增加不确定性 (方差变大)，接受率会提高
    # T < 1.0: 减少不确定性 (方差变小)，接受率会降低
    temperature = 2.0  # 建议尝试 1.5 到 3.0
    
    alphas = []
    total_score = sum(judge_scores_dict.values()) if judge_scores_dict else 1
    
    for name in contestants:
        key = (season, week, name)
        pop = info_map.get(key, {}).get('popularity_ratio', 0.1)
        if pd.isna(pop): pop = 0.1
        raw_score = judge_scores_dict.get(name, 0)
        norm_score = raw_score / total_score if total_score > 0 else 0
        
        # 原始计算
        val = base_alpha + (pow(pop, pop_exp) * w_pop) + (norm_score * w_judge)
        
        # 应用温度缩放：Alpha 越小，方差越大，分布越平坦
        val = val / temperature 
        
        alphas.append(max(0.01, val))
        
    return np.array(alphas)

def rank_sort(judge_ranks_dict, fan_votes_dict):
    """ Rank System: Total Rank (Higher value = Worse) """
    sorted_v = sorted(judge_ranks_dict.keys(), key=lambda x: fan_votes_dict[x], reverse=True)
    v_ranks = {name: i+1 for i, name in enumerate(sorted_v)}
    final_scores = []
    for name in judge_ranks_dict:
        # Judge Rank + Fan Rank
        total = judge_ranks_dict.get(name, 5) + v_ranks[name]
        final_scores.append((name, total))
    # 降序排列：分高者(表现差)在前
    final_scores.sort(key=lambda x: x[1], reverse=True)
    return final_scores

def percentile_sort(judge_scores_dict, fan_votes_dict):
    """ Percentile System: Total % (Lower value = Worse) """
    tj = sum(judge_scores_dict.values())
    tv = sum(fan_votes_dict.values())
    final_scores = []
    for name in judge_scores_dict:
        pj = judge_scores_dict[name]/tj if tj>0 else 0
        pv = fan_votes_dict[name]/tv if tv>0 else 0
        final_scores.append((name, pj + pv))
    # 升序排列：分低者(表现差)在前
    final_scores.sort(key=lambda x: x[1])
    return final_scores

# ==========================================
# 3. 优化模块 (Soft Rank Score)
# ==========================================

def objective_function(params, structure, info_map):
    """
    目标函数：最小化“软排名损失”
    不只看Top-1是否命中，还看预测的淘汰者离倒数第一有多近。
    """
    if any(p < 0 for p in params): return 1000.0
    if params[2] > 6.0: return 1000.0 # 限制指数不过大
    
    total_score = 0
    total_weeks = 0
    
    for s in structure:
        for w in structure[s]:
            data = structure[s][w]
            contestants = data['contestants']
            actual_eliminated = data['eliminated']
            
            if actual_eliminated is None: continue
            
            # 准备数据
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
            
            # 计算 Alpha
            alphas = calculate_alphas(params, contestants, current_judge_scores, info_map, s, w)
            if alphas is None: return 1000.0
            
            # 使用期望值 (Expectation) 进行确定性预测
            expected_votes = alphas / alphas.sum()
            vote_dict = {name: v for name, v in zip(contestants, expected_votes)}
            
            predicted_order = []
            
            # 生成预测排名 (头部为最可能淘汰者)
            if s <= 2 or s >= 28: # Rank System
                res = rank_sort(current_judge_ranks, vote_dict)
                predicted_order = [x[0] for x in res]
            else: # Percentile System
                res = percentile_sort(current_judge_scores, vote_dict)
                predicted_order = [x[0] for x in res]
            
            # --- 软评分逻辑 (Soft Rank Scoring) ---
            try:
                rank_index = predicted_order.index(actual_eliminated) # 0 means Correct (Bottom 1)
            except ValueError: continue

            # 评分卡
            if rank_index == 0:
                week_score = 1.0   # 完美预测
            elif rank_index == 1:
                week_score = 0.8   # 预测在倒数第二 (S28+评委拯救区，或极接近)
            elif rank_index == 2:
                week_score = 0.5   # 预测在倒数第三 (危险区)
            elif rank_index <= 4:
                week_score = 0.2   # 勉强相关
            else:
                week_score = 0.0   # 预测完全偏离
            
            total_score += week_score
            total_weeks += 1
            
    if total_weeks == 0: return 1.0
    # 目标是最小化 (1 - 平均得分)
    return 1.0 - (total_score / total_weeks)

def optimize_weights(structure, info_map):
    """运行优化算法"""
    print("\n" + "="*50)
    print("🚀 启动参数优化 (Soft Rank Score Mechanism)...")
    
    # 初始猜测 [Base, W_Pop, Pop_Exp, W_Judge]
    initial_guess = [1.0, 10.0, 1.5, 5.0]
    
    result = minimize(
        objective_function,
        initial_guess,
        args=(structure, info_map),
        method='Nelder-Mead',
        options={'maxiter': 80, 'disp': True}
    )
    
    best_params = result.x
    final_score = 1.0 - result.fun
    print(f"\n✅ 优化完成! 最佳软评分准确率: {final_score:.2%}")
    print(f"最佳参数: Base={best_params[0]:.2f}, Pop_W={best_params[1]:.2f}, "
          f"Pop_Exp={best_params[2]:.2f}, Judge_W={best_params[3]:.2f}")
    print("="*50 + "\n")
    return best_params

# ==========================================
# 4. 主程序 (Sampling & Metrics)
# ==========================================

def model_main(n_samples=1000, target_season=None):
    structure = load_and_process_data()
    info_map = load_info()
    
    if not structure:
        print("无数据，终止。")
        return

    # 1. 获取最佳权重 (使用之前的优化函数)
    best_params = optimize_weights(structure, info_map)
    
    all_estimates = []
    all_seasons = sorted(structure.keys())
    if target_season:
        all_seasons = [target_season] if target_season in all_seasons else []

    log(f"开始 MCMC 采样与一致性深度评估 (Samples={n_samples})...")
    
    for s in all_seasons:
        for w in sorted(structure[s].keys()):
            print(f"Processing S{s} W{w}...", end='\r')
            
            data = structure[s][w]
            contestants = data['contestants']
            actual_eliminated = data['eliminated']
            
            # 准备数据
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

            # --- A. 计算先验分布 (Prior) ---
            alphas = calculate_alphas(best_params, contestants, current_judge_scores, info_map, s, w)
            prior_means = alphas / alphas.sum() # 先验期望投票率
            
            # --- B. MCMC 采样 (Posterior) ---
            accepted_votes = []
            total_tried = 0
            
            if actual_eliminated is None:
                accepted_votes = np.random.dirichlet(alphas, n_samples)
                total_tried = n_samples
            else:
                while len(accepted_votes) < n_samples:
                    batch = (n_samples - len(accepted_votes)) * 5
                    batch = max(batch, 1000)
                    samples = np.random.dirichlet(alphas, batch)
                    total_tried += batch
                    
                    for i in range(batch):
                        vote_dict = {name: samples[i][j] for j, name in enumerate(contestants)}
                        
                        # 约束检查
                        valid = False
                        if s <= 2 or s >= 28: # Rank
                            res = rank_sort(current_judge_ranks, vote_dict)
                            if s >= 28: valid = actual_eliminated in [x[0] for x in res[:2]]
                            else: valid = res[0][0] == actual_eliminated
                        else: # Percentile
                            res = percentile_sort(current_judge_scores, vote_dict)
                            valid = res[0][0] == actual_eliminated
                        
                        if valid:
                            accepted_votes.append(samples[i])
                            if len(accepted_votes) >= n_samples: break
                    
                    if total_tried > n_samples * 100 and len(accepted_votes) < n_samples * 0.05:
                        break 
            
            # --- C. 计算统计量与一致性指标 ---
            if len(accepted_votes) > 0:
                vals = np.array(accepted_votes)
                posterior_means = np.mean(vals, axis=0) # 后验期望
                
                # 1. MSE (Distortion): 后验与先验的均方误差
                # 衡量：为了解释结果，投票率偏离了常理多少？
                mse_distortion = np.mean((posterior_means - prior_means) ** 2)
                
                # 2. KL Divergence (Approximation): 
                # 衡量：信息增益/惊奇度 (避免log0，加一个小epsilon)
                epsilon = 1e-9
                kl_divergence = np.sum(posterior_means * np.log((posterior_means + epsilon) / (prior_means + epsilon)))
                
                # 3. Acceptance Rate
                consistency_score = len(accepted_votes) / total_tried
                
                # 确定性指标
                std_votes = np.std(vals, axis=0)
                ci_lower = np.percentile(vals, 2.5, axis=0)
                ci_upper = np.percentile(vals, 97.5, axis=0)
                
            else:
                posterior_means = prior_means
                std_votes = np.zeros_like(prior_means)
                ci_lower, ci_upper = prior_means, prior_means
                consistency_score = 0.0
                mse_distortion = 0.0 # 无法计算，或者设为最大值
                kl_divergence = 0.0

            # 存入列表
            for i, name in enumerate(contestants):
                all_estimates.append({
                    'season': s, 
                    'week': w, 
                    'celebrity_name': name,
                    
                    # 核心数据
                    'vote_mean': posterior_means[i],
                    'prior_mean': prior_means[i], # 保存先验以便后续分析
                    
                    # 确定性
                    'vote_std': std_votes[i],
                    'vote_CI_lower': ci_lower[i], 
                    'vote_CI_upper': ci_upper[i],
                    
                    # 一致性/争议度指标 (整周共享)
                    'consistency_acceptance': consistency_score,
                    'consistency_mse': mse_distortion, # 新增
                    'consistency_kl': kl_divergence,   # 新增
                    
                    'is_eliminated': (name == actual_eliminated)
                })

    log(f"\n所有计算完成。")
    result_df = pd.DataFrame(all_estimates)
    save_result(result_df, OUTPUT_FILE)
    log(f"结果已保存至 {OUTPUT_FILE}")
if __name__ == "__main__":
    # 示例运行 (n_samples建议设为10000以获得平滑的CI)
    model_main(n_samples=2000)