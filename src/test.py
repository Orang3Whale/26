import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
import config
from utils import log, save_result
import os
import re

DATA_FILE = config.DATA_RAW / '2026_MCM_Problem_C_Data.csv'
WEEK_FILE = config.DATA_PROCESSED / 'weekly_data_with_popularity.csv'
SEASON_FILE = config.DATA_PROCESSED / 'seasonal_data_with_popularity.csv'
OUTPUT_FILE = 'votes_estimation_CI.csv'
VALIDATION_FILE = 'forward_prediction_validation.csv'

#------------------------------- 数据加载区域 ------------------------------
def load_and_process_data():
    """
    读取原始数据，构建每周的比赛结构（谁参赛、谁被淘汰）。
    """
    log("正在读取原始赛程结构...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"未找到原始数据文件: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    
    def parse_elim(res):
        if isinstance(res, str) and res.startswith("Eliminated Week"):
            return int(res.split()[-1])
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
                
            eliminated = season_df[season_df['elim_week'] == w]['celebrity_name'].tolist()
            eliminated_name = eliminated[0] if eliminated else None
            
            structured_data[s][w] = {
                'contestants': active_contestants,
                'eliminated': eliminated_name
            }
            
    return structured_data 

def load_info():
    """
    读取预计算好的 weekly_data 文件。
    """
    log(f"正在读取预计算信息: {WEEK_FILE} ...")
    if not os.path.exists(WEEK_FILE):
        log(f"警告: 未找到 {WEEK_FILE}，将无法获取精确排名和人气数据。")
        return {}
        
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

#------------------------------- Task 1 求解区域 ------------------------------

def model_forward(target_season=None):
    """
    【新增函数】正向预测模型
    利用 人气先验 + 评委分 -> 直接计算理论得票率 -> 预测排名 -> 对比真实淘汰
    """
    structure = load_and_process_data()
    info_map = load_info()
    
    results = []
    
    all_seasons = sorted(structure.keys())
    if target_season is not None:
        if target_season in all_seasons:
            all_seasons = [target_season]
            log(f"正向预测模式：仅运行第 {target_season} 季...")
        else:
            log(f"错误：Season {target_season} 未找到。")
            return

    log("开始执行正向预测验证...")

    for s in all_seasons:
        for w in sorted(structure[s].keys()):
            data = structure[s][w]
            contestants = data['contestants']
            actual_elim = data['eliminated']
            
            # 1. 准备数据
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
            
            # 2. 计算先验 Alpha (期望人气分布)
            alphas = dirichlet_alpha(s, contestants, current_judge_scores, info_map, w)
            
            # 3. 计算理论得票率 (Expected Vote Share)
            # Dirichlet分布的期望 E[x_i] = alpha_i / sum(alpha)
            total_alpha = np.sum(alphas)
            expected_votes = alphas / total_alpha
            vote_dict = {name: v for name, v in zip(contestants, expected_votes)}
            
            # 4. 模拟赛制结算 (正向推演)
            if s <= 2 or s >= 28:
                # 排名制
                sorted_results = rank_sort(current_judge_ranks, vote_dict)
                # sorted_results[0] 是 Rank Sum 最大的人 (即最差的人)
            else:
                # 百分比制
                sorted_results = percentile_sort(current_judge_scores, vote_dict)
                # sorted_results[0] 是 Total Score 最小的人 (即最差的人)
            
            # 5. 分析预测结果
            predicted_elim = sorted_results[0][0] # 模型认为该走的人
            
            # 真实淘汰者在预测名单里的排位 (1表示模型认为他就是倒数第一，准确)
            actual_elim_rank = -1
            elimination_order = [x[0] for x in sorted_results] # 0是最危险, -1是最安全
            
            if actual_elim:
                if actual_elim in elimination_order:
                    # 获取真实淘汰者的索引 + 1 (即第几倒霉)
                    actual_elim_rank = elimination_order.index(actual_elim) + 1
            
            # 判断是否精确命中
            is_correct = (predicted_elim == actual_elim)
            
            # S28+ 评委拯救区命中 (只要在前两名都算预测成功)
            is_saved_zone = False
            if s >= 28 and actual_elim in elimination_order[:2]:
                is_saved_zone = True
                
            results.append({
                'season': s,
                'week': w,
                'actual_eliminated': actual_elim,
                'predicted_eliminated': predicted_elim,
                'is_correct_exact': is_correct,
                'is_correct_save_zone': is_saved_zone, # 对S28+有意义
                'rank_of_actual_eliminated': actual_elim_rank, # 越小越好，1代表完美预测
                'num_contestants': len(contestants)
            })
            
    df_res = pd.DataFrame(results)
    save_result(df_res, VALIDATION_FILE)
    log(f"正向验证完成，结果已保存至 {VALIDATION_FILE}")
    
    # 打印简单的准确率统计
    total_weeks = len(df_res[df_res['actual_eliminated'].notna()])
    correct_weeks = len(df_res[df_res['is_correct_exact'] == True])
    log(f"总体精确预测准确率: {correct_weeks}/{total_weeks} ({correct_weeks/total_weeks:.2%})")
    return df_res

def model_1(n_samples=1000, target_season=1):
    """
    粉丝投票预测模型 (逆向工程：MCMC采样)
    """
    structure = load_and_process_data()
    info_map = load_info()
    
    all_estimates = []
    
    all_seasons = sorted(structure.keys())
    if target_season is not None:
        if target_season in all_seasons:
            all_seasons = [target_season]
            log(f"测试模式：仅运行第 {target_season} 季的数据...")
        else:
            log(f"错误：数据中未找到第 {target_season} 季。")
            return
    else:
        log(f"全量模式：运行所有赛季。")

    log(f"开始计算 MCMC，目标采样数: {n_samples}/周")
    
    for s in all_seasons:
        sorted_weeks = sorted(structure[s].keys())
        for w in sorted_weeks:
            print(f"计算中: Season {s} Week {w}", end='\r')
            
            data = structure[s][w]
            contestants = data['contestants']
            actual_eliminated = data['eliminated']
            
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
            
            alphas = dirichlet_alpha(s, contestants, current_judge_scores, info_map, w)
            
            accepted_votes = []
            batch_size = n_samples * 20 
            total_tried = 0  
            
            if actual_eliminated is None:
                accepted_votes = np.random.dirichlet(alphas, n_samples)
                total_tried = n_samples
            else:
                while len(accepted_votes) < n_samples:
                    batch_size = max((n_samples - len(accepted_votes)) * 20, 1000)
                    samples_matrix = np.random.dirichlet(alphas, batch_size)
                    total_tried += batch_size 
                    
                    for i in range(batch_size):
                        vote_sample = samples_matrix[i]
                        vote_dict = {name: v for name, v in zip(contestants, vote_sample)}
                        
                        if reject_func(vote_dict, current_judge_ranks, current_judge_scores, actual_eliminated, s):
                            accepted_votes.append(vote_sample)
                            if len(accepted_votes) >= n_samples:
                                break
                    
                    if len(accepted_votes) < n_samples and batch_size > 50000:
                        break # 收敛困难
                    if len(accepted_votes) < n_samples:
                        batch_size *= 2 
            
            if len(accepted_votes) > 0:
                votes_array = np.array(accepted_votes)
                mean_votes = np.mean(votes_array, axis=0)
                std_votes = np.std(votes_array, axis=0)
                ci_lower = np.percentile(votes_array, 2.5, axis=0)
                ci_upper = np.percentile(votes_array, 97.5, axis=0)
                acceptance_rate = len(accepted_votes) / total_tried
            else:
                mean_votes = alphas / alphas.sum()
                std_votes = np.zeros_like(mean_votes)
                ci_lower = mean_votes
                ci_upper = mean_votes
                acceptance_rate = 0

            for i, name in enumerate(contestants):
                all_estimates.append({
                    'season': s,
                    'week': w,
                    'celebrity_name': name,
                    'vote_mean': mean_votes[i],
                    'vote_std': std_votes[i],
                    'vote_CI_lower': ci_lower[i],
                    'vote_CI_upper': ci_upper[i],
                    'judge_rank': current_judge_ranks[name],
                    'judge_score': current_judge_scores[name],
                    'is_eliminated': (name == actual_eliminated),
                    'consistency': acceptance_rate
                })

    log("\n所有赛季计算完成！")
    result_df = pd.DataFrame(all_estimates)
    save_result(result_df, OUTPUT_FILE)
    log(f"结果已保存至 {OUTPUT_FILE}")

def dirichlet_alpha(season, contestants, judge_scores_dict, info_map, week):
    """
    计算dirichlet函数的浓度参数 Alpha
    """
    alphas = []
    base_alpha = 2.0 
    w_pop = 5.0   
    w_judge = 0.1 
    
    for name in contestants:
        key = (season, week, name)
        if key in info_map:
            pop = info_map[key].get('popularity_ratio', 0.5)
            if pd.isna(pop): pop = 0.5
        else:
            pop = 0.5 
        
        score = judge_scores_dict.get(name, 0)
        val = base_alpha + w_pop * pop + w_judge * score
        alphas.append(max(0.1, val)) 
        
    return np.array(alphas)

def reject_func(sampled_votes_dict, judge_ranks_dict, judge_scores_dict, actual_eliminated, season):
    """
    拒绝采样逻辑
    """
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
    """
    排名制: 返回按“最该淘汰”到“最安全”排序的列表
    """
    contestants = list(judge_ranks_dict.keys())
    sorted_v = sorted(contestants, key=lambda x: fan_votes_dict[x], reverse=True)
    v_ranks = {name: i+1 for i, name in enumerate(sorted_v)}
    
    final_scores = []
    for name in contestants:
        j_rank = judge_ranks_dict.get(name, 99)
        v_rank = v_ranks[name]
        total = j_rank + v_rank
        final_scores.append((name, total))
    
    # 降序排列，RankSum越大越糟糕（Index 0 = Eliminated）
    final_scores.sort(key=lambda x: x[1], reverse=True)
    return final_scores

def percentile_sort(judge_scores_dict, fan_votes_dict):
    """
    百分比制: 返回按“最该淘汰”到“最安全”排序的列表
    """
    contestants = list(judge_scores_dict.keys())
    total_j = sum(judge_scores_dict.values())
    total_v = sum(fan_votes_dict.values()) 
    
    final_scores = []
    for name in contestants:
        p_j = judge_scores_dict[name] / total_j if total_j > 0 else 0
        p_v = fan_votes_dict[name] / total_v if total_v > 0 else 0
        score = p_j + p_v
        final_scores.append((name, score))
    
    # 升序排列，TotalScore越小越糟糕（Index 0 = Eliminated）
    final_scores.sort(key=lambda x: x[1])
    return final_scores

#------------------------------ 主程序 --------------------------------

if __name__ == "__main__":
    # 1. 运行正向预测验证 (Forward Prediction)
    model_forward(target_season=1) 
    
    # 2. 运行逆向 MCMC 求解 (Task 1 主要任务)
    # model_1(n_samples=10000, target_season=None)