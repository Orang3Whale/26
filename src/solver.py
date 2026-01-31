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
#-------------------------------task 1 求解区域------------------------------
def load_and_process_data():
    """
    读取原始数据，构建每周的比赛结构（谁参赛、谁被淘汰）。
    """
    print("正在读取原始赛程结构...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"未找到原始数据文件: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    
    # 解析淘汰周
    def parse_elim(res):
        if isinstance(res, str) and res.startswith("Eliminated Week"):
            return int(res.split()[-1])
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
            # 获取本周分数列
            cols = [f'week{w}_judge{j}_score' for j in range(1, 4)]
            cols = [c for c in cols if c in df.columns]
            
            if not cols: continue
            
            # 计算总分以判断是否参赛 (Sum > 0)
            weekly_scores = season_df.set_index('celebrity_name')[cols].sum(axis=1)
            active_contestants = weekly_scores[weekly_scores > 0].index.tolist()
            
            if not active_contestants:
                continue
                
            # 确定本周淘汰者
            eliminated = season_df[season_df['elim_week'] == w]['celebrity_name'].tolist()
            eliminated_name = eliminated[0] if eliminated else None
            
            structured_data[s][w] = {
                'contestants': active_contestants,
                'eliminated': eliminated_name
            }
            
    return structured_data 
def model_1(n_samples=1000,target_season=1):
    """
    粉丝投票预测模型
    dirichlet分布+MCMC采样
    :params n_samples:采样数
    :params target_season:目标季，用于测试
    """
    structure = load_and_process_data()
    info_map = load_info()
    
    all_estimates = []
    
    all_seasons = sorted(structure.keys())
    # --- 修改点：过滤赛季 ---
    if target_season is not None:
        if target_season in all_seasons:
            sorted_seasons = [target_season]
            print(f"测试模式：仅运行第 {target_season} 季的数据...")
        else:
            print(f"错误：数据中未找到第 {target_season} 季。")
            return
    else:
        sorted_seasons = all_seasons
    log(f"开始计算，目标采样数: {n_samples}/周")
    
    for s in sorted_seasons:
        sorted_weeks = sorted(structure[s].keys())
        
        for w in sorted_weeks:
            print(f"计算中: Season {s} Week {w}", end='\r')
            
            data = structure[s][w]
            contestants = data['contestants']
            actual_eliminated = data['eliminated']
            
            # 从 info_map 中准备当周所需的 Ranks 和 Scores
            current_judge_ranks = {}
            current_judge_scores = {}
            
            for name in contestants:
                key = (s, w, name)
                if key in info_map:
                    current_judge_ranks[name] = info_map[key]['weekly_rank']
                    current_judge_scores[name] = info_map[key]['weekly_total']
                else:
                    # 如果缺失数据，给默认值防止报错
                    current_judge_ranks[name] = 5 
                    current_judge_scores[name] = 20
            
            # 3. 构造先验 Alpha (传入 info_map 以获取 popularity)
            alphas = dirichlet_alpha(s, contestants, current_judge_scores, info_map, w)
            
            # 4. 采样循环
            accepted_votes = []
            batch_size = n_samples * 20 # 加大 Batch Size 提高效率
            total_tried = 0  # <--- 新增计数器
            
            # 安全检查：如果没有人淘汰，就不需要复杂的拒绝采样，直接采一次即可
            if actual_eliminated is None:
                accepted_votes = np.random.dirichlet(alphas, n_samples)
                total_tried = n_samples
            else:
                while len(accepted_votes) < n_samples:
                    # 动态调整 batch_size
                    batch_size = (n_samples - len(accepted_votes)) * 20
                    batch_size = max(batch_size, 1000) # 至少采1000
                    
                    samples_matrix = np.random.dirichlet(alphas, batch_size)
                    total_tried += batch_size # <--- 累加尝试次数
                    for i in range(batch_size):
                        vote_sample = samples_matrix[i]
                        vote_dict = {name: v for name, v in zip(contestants, vote_sample)}
                        
                        # 验证样本 (Rank制传Rank字典，百分比制传Score字典)
                        if reject_func(vote_dict, current_judge_ranks, current_judge_scores, actual_eliminated, s):
                            accepted_votes.append(vote_sample)
                            if len(accepted_votes) >= n_samples:
                                break
                    
                    # 防止死循环：如果尝试太多次仍未找到解，说明先验偏差太大或无解，强制退出
                    if len(accepted_votes) < n_samples and batch_size > 50000:
                        print(f"\nWarning: S{s} W{w} 收敛困难，仅收集到 {len(accepted_votes)} 样本。")
                        break
                    
                    # 如果第一轮没采够，下一次尝试更多
                    if len(accepted_votes) < n_samples:
                        batch_size *= 2 
            # 计算一致性
            # consistency_rate = len(accepted_votes) / total_tried if total_tried > 0 else 0
            # 6. 聚合结果 (修改后：计算分布特征)
            if len(accepted_votes) > 0:
                # 转换为 numpy 数组以便计算
                votes_array = np.array(accepted_votes)
                
                # A. 点估计 (Point Estimate)
                mean_votes = np.mean(votes_array, axis=0)
                
                # B. 确定性度量 (Certainty): 标准差
                std_votes = np.std(votes_array, axis=0)
                
                # C. 区间估计 (Interval): 95% 置信区间 (2.5% - 97.5%)
                ci_lower = np.percentile(votes_array, 2.5, axis=0)
                ci_upper = np.percentile(votes_array, 97.5, axis=0)
                
                # D. 一致性度量 (Consistency): 接受率
                # 尝试采样的总次数 = batch_size * 循环次数 (这里简单估算)
                acceptance_rate = len(accepted_votes) / total_tried
                # 注意：需要在上面的循环里记录 total_tried 变量
                
            else:
                # 兜底逻辑
                mean_votes = alphas / alphas.sum()
                std_votes = np.zeros_like(mean_votes) # 无法衡量不确定性
                ci_lower = mean_votes
                ci_upper = mean_votes

            # 存入结果列表
            for i, name in enumerate(contestants):
                all_estimates.append({
                    'season': s,
                    'week': w,
                    'celebrity_name': name,
                    
                    # --- 核心预测值 ---
                    'vote_mean': mean_votes[i],   # 你的点估计
                    
                    # --- 确定性/波动性 ---
                    'vote_std': std_votes[i],     # 标准差
                    'vote_CI_lower': ci_lower[i], # 95% CI 下界
                    'vote_CI_upper': ci_upper[i], # 95% CI 上界
                    
                    # --- 辅助信息 ---
                    'judge_rank': current_judge_ranks[name],
                    'judge_score': current_judge_scores[name],
                    'is_eliminated': (name == actual_eliminated),
                    'consistency':acceptance_rate
                })

    log("\n所有赛季计算完成！")
    result_df = pd.DataFrame(all_estimates)
    save_result(result_df,OUTPUT_FILE)
    log(f"结果已保存至 {OUTPUT_FILE}")

def load_info():
    """
    读取预计算好的 weekly_data 文件。
    返回一个字典，Key为 (season, week, celebrity_name)，
    Value 为包含 rank, popularity, score 等信息的字典。
    """
    print(f"正在读取预计算信息: {WEEK_FILE} ...")
    if not os.path.exists(WEEK_FILE):
        print(f"警告: 未找到 {WEEK_FILE}，将无法获取精确排名和人气数据。")
        return {}
        
    df = pd.read_csv(WEEK_FILE)
    
    # 确保列名没有空格
    df.columns = [c.strip() for c in df.columns]
    
    info_map = {}
    # 遍历每一行，存入字典
    for _, row in df.iterrows():
        key = (row['season'], row['week'], row['celebrity_name'])
        info_map[key] = {
            'weekly_rank': row['weekly_rank'],           # 评委排名
            'popularity_ratio': row['popularity_ratio'], # 人气比率
            'weekly_total': row['weekly_total'],         # 评委总分
            'eliminated_this_week': row['eliminated_this_week']
        }
    
    return info_map

def dirichlet():
    """
    该函数为迪利克雷分布
    """

def dirichlet_alpha(season, contestants, judge_scores_dict, info_map, week):
    """
    计算dirichlet函数的浓度参数
    ：人气＋β*裁判分数
    最后需要归一化
    Alpha_i = Base + w1 * Popularity_i + w2 * Judge_Score_i
    """
    alphas = []
    base_alpha = 2.0 
    w_pop = 5.0   # 人气权重
    w_judge = 0.1 # 评委分权重
    
    for name in contestants:
        # 从 info_map 获取该选手当周的人气 ratio
        key = (season, week, name)
        if key in info_map:
            pop = info_map[key].get('popularity_ratio', 0.5)
            # 处理可能的空值
            if pd.isna(pop): pop = 0.5
        else:
            pop = 0.5 # 默认值
        
        # 获取评委分
        score = judge_scores_dict.get(name, 0)
        
        # 计算 alpha
        val = base_alpha + w_pop * pop + w_judge * score
        alphas.append(max(0.1, val)) # 保证非负
        
    return np.array(alphas)

def reject_func(sampled_votes_dict, judge_ranks_dict, judge_scores_dict, actual_eliminated, season):
    """
    该函数用于MCMC的拒绝算法，确保后验分布结果正确
    检查生成的 votes 是否会导致 actual_eliminated 被淘汰。
    返回: True (接受样本), False (拒绝样本)
    """
    if actual_eliminated is None:
        return True 
    
    # S1-2, S28-34: Rank System
    if season <= 2 or season >= 28:
        # 注意这里传入的是 Ranks Dict
        results = rank_sort(judge_ranks_dict, sampled_votes_dict)
        
        if season >= 28:
            # S28+ 评委拯救环节: 只要在倒数两名内，就有可能被淘汰
            # 如果历史上的淘汰者进入了模拟结果的 Bottom 2，则该样本可行
            bottom_2 = [x[0] for x in results[:2]]
            return actual_eliminated in bottom_2
        else:
            # S1-2: 必须是最后一名
            return results[0][0] == actual_eliminated
            
    # S3-27: Percentile System
    else:
        # 注意这里传入的是 Scores Dict
        results = percentile_sort(judge_scores_dict, sampled_votes_dict)
        return results[0][0] == actual_eliminated

def rank_sort(judge_ranks_dict, fan_votes_dict):
    """
    本函数用于定义排名制
    规则：总分 = 评委排名 + 观众排名。总分最高者（数值最大）淘汰。
    适用范围：1-2 28-34
    """
    contestants = list(judge_ranks_dict.keys())
    
    # 1. 获取预计算的评委排名
    # 直接使用传入的字典，无需重算
    
    # 2. 计算观众排名 (票数越高，排名数值越小 1,2,3...)
    # 票数高的排前面 (Rank 1)
    sorted_v = sorted(contestants, key=lambda x: fan_votes_dict[x], reverse=True)
    v_ranks = {name: i+1 for i, name in enumerate(sorted_v)}
    
    # 3. 计算总排位分
    final_scores = []
    for name in contestants:
        j_rank = judge_ranks_dict.get(name, 99) # 默认防错
        v_rank = v_ranks[name]
        total_rank_score = j_rank + v_rank
        final_scores.append((name, total_rank_score))
    
    # 排序：按总 rank score 降序排列 (数值大 = 表现差 = 淘汰)
    # 例如：Judge Rank 10 + Fan Rank 10 = 20 (Worst)
    final_scores.sort(key=lambda x: x[1], reverse=True)
    return final_scores

def percentile_sort(judge_scores_dict, fan_votes_dict):
    """
    本函数用于实现百分比制
    适用范围：3-27
    规则：总分 = (评委分/评委总分) + (观众票/观众总票)。总分最低者淘汰。
    """
    contestants = list(judge_scores_dict.keys())
    
    total_j = sum(judge_scores_dict.values())
    total_v = sum(fan_votes_dict.values()) # Should be 1.0 if normalized
    
    final_scores = []
    for name in contestants:
        p_j = judge_scores_dict[name] / total_j if total_j > 0 else 0
        p_v = fan_votes_dict[name] / total_v if total_v > 0 else 0
        score = p_j + p_v
        final_scores.append((name, score))
    
    # 排序：按总分升序排列 (分数低=淘汰)
    final_scores.sort(key=lambda x: x[1])
    return final_scores
#------------------------------task 1 求解结束--------------------------------


if __name__ == "__main__":
    model_1(n_samples=100000,target_season=None)