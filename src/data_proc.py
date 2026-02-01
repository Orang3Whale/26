#本代码进行数据处理
from utils import log
import config
import pandas as pd
import numpy as np
import re

RAW_file = config.DATA_RAW / "2026_MCM_Problem_C_Data.csv"


def stat_unique(filename,column):
    """本函数用于读取raw并查看该列下的独立的属性名"""
    file = pd.read_csv(filename)
    unique_values = file[column].unique()
    print(f"列 '{column}' 的独立值:")
    for i, value in enumerate(unique_values, 1):
        print(f"{i}: {value}")
    print(f"\n总共有 {len(unique_values)} 个独立值")

def stat_clean(filename):
    """本函数用于数据处理,提取数据特征"""
    #首先，提取每个人的淘汰周次
    df = pd.read_csv(filename)
    df['weeks_survived'] = df['results']

def weekly_stat_process():
    """本函数用于生成周维度下的数据"""
        # 1. 加载原始数据
    df = pd.read_csv(RAW_file)

    # 2. 辅助函数：解析淘汰周
    def parse_elimination(res):
        if isinstance(res, str):
            if res.startswith("Eliminated Week"):
                return int(res.split()[-1])
            elif "Place" in res:
                return 100 # 代表进入决赛，从未被淘汰
            elif res == "Withdrew":
                return -1 # 特殊处理
        return 100

    df['elim_week_parsed'] = df['results'].apply(parse_elimination)

    # 3. 转换数据 (Wide to Long)
    long_data = []

    # 检测最大周数 (根据列名)
    week_cols = [c for c in df.columns if c.startswith('week') and 'judge' in c]
    max_week = 0
    for c in week_cols:
        match = re.search(r'week(\d+)_', c)
        if match:
            w = int(match.group(1))
            if w > max_week: max_week = w

    for idx, row in df.iterrows():
        # 处理 "Withdrew" 情况：找到最后一个有分数的周
        last_active_week = 0
        for w in range(1, max_week + 1):
            s_cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
            scores = [row[c] for c in s_cols if c in df.columns and pd.notna(row[c])]
            if sum(scores) > 0:
                last_active_week = w
        
        elim_week = row['elim_week_parsed']
        if elim_week == -1: elim_week = last_active_week

        # 遍历每一周提取数据
        for w in range(1, max_week + 1):
            score_cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
            valid_cols = [c for c in score_cols if c in df.columns]
            
            if not valid_cols: continue
            
            scores = row[valid_cols]
            current_total = scores.sum()
            
            # 过滤掉得分为0或全空的周（代表未参赛或已淘汰）
            if current_total == 0 or scores.isna().all():
                continue
                
            valid_scores = scores[scores > 0]
            judges_count = valid_scores.count()
            weekly_avg = valid_scores.mean() if judges_count > 0 else 0
            
            age = row['celebrity_age_during_season']
            if age < 30: age_group = '<30'
            elif age < 40: age_group = '30-39'
            elif age < 50: age_group = '40-49'
            else: age_group = '50+'
            
            long_data.append({
                'celebrity_name': row['celebrity_name'],
                'season': row['season'],
                'week': w,
                'weekly_total': current_total,
                'weekly_avg': weekly_avg,
                'judges_count': judges_count,
                'alive': True, # 只要有分就是alive
                'eliminated_this_week': (w == elim_week),
                'age_group': age_group,
                'us_born': (row['celebrity_homecountry/region'] == 'United States')
            })

    df_weekly = pd.DataFrame(long_data)

    # 4. 计算组内统计量 (排名、百分位、赛季进度)
    # 周排名
    df_weekly['weekly_rank'] = df_weekly.groupby(['season', 'week'])['weekly_total'].rank(ascending=False, method='min')
    # 分数百分位
    df_weekly['score_percentile'] = df_weekly.groupby(['season', 'week'])['weekly_total'].rank(pct=True) * 100
    # 赛季进度
    season_max_weeks = df_weekly.groupby('season')['week'].max().rename('total_weeks')
    df_weekly = df_weekly.merge(season_max_weeks, on='season')
    df_weekly['week_progress'] = df_weekly['week'] / df_weekly['total_weeks']

    # 5. 输出结果
    final_cols = [
        'celebrity_name', 'season', 'week', 'weekly_total', 'weekly_avg', 
        'judges_count', 'alive', 'eliminated_this_week', 'weekly_rank', 
        'score_percentile', 'age_group', 'us_born', 'week_progress'
    ]
    df_final = df_weekly[final_cols]
    output_file = config.DATA_PROCESSED / 'processed_weekly_data.csv'
    df_final.to_csv(output_file, index=False)

def seasonal_process():
    """
    将原始DWTS数据处理为赛季/选手级别的聚合数据集。
    """
    # 1. 加载原始数据
    df = pd.read_csv(RAW_file)
    
    processed_rows = []
    
    # 2. 遍历每一行（每个选手一个赛季的数据）
    for idx, row in df.iterrows():
        # --- 基础信息 ---
        celebrity_name = row['celebrity_name']
        season = row['season']
        age = row['celebrity_age_during_season']
        industry = row['celebrity_industry']
        
        # --- 是否美国出生 ---
        # 逻辑：判断 celebrity_homecountry/region 是否为 "United States"
        us_born = (row['celebrity_homecountry/region'] == 'United States')
        
        # --- 分数统计 (Avg & Std) ---
        # 提取该选手所有周、所有评委的有效分数 (非NaN, 非0)
        # 原始列名格式示例: week1_judge1_score
        c_scores = []
        weeks_active = set()
        
        # 遍历所有可能的周和评委列
        for col in df.columns:
            if 'judge' in col and 'score' in col and 'week' in col:
                val = row[col]
                # 只有大于0的分数才计入统计（0通常代表已淘汰或未参赛）
                if pd.notna(val) and val > 0:
                    c_scores.append(val)
                    # 提取周号用于计算存活周数
                    match = re.search(r'week(\d+)_', col)
                    if match:
                        weeks_active.add(int(match.group(1)))
        
        if c_scores:
            avg_score = np.mean(c_scores)
            score_std = np.std(c_scores)
        else:
            avg_score = 0.0
            score_std = 0.0
            
        weeks_survived = len(weeks_active)
        
        # --- 最终名次 & 前三名 ---
        # 优先使用 'placement' 列，如果没有则解析 'results'
        final_placement = np.nan
        if 'placement' in df.columns and pd.notna(row['placement']):
            final_placement = int(row['placement'])
        else:
            # 备用解析逻辑
            res = str(row['results'])
            if 'Place' in res: # e.g., "1st Place"
                match = re.search(r'(\d+)', res)
                if match: final_placement = int(match.group(1))
        
        # 标记是否前三
        top_3 = (final_placement <= 3) if pd.notna(final_placement) else False
        
        # --- 封装数据 ---
        processed_rows.append({
            'celebrity_name': celebrity_name,
            'season': season,
            'avg_score': avg_score,
            'score_std': score_std,
            'weeks_survived': weeks_survived,
            'final_placement': final_placement,
            'top_3': top_3,
            'age': age,
            'industry': industry,
            'us_born': us_born
        })
        
    # 3. 生成DataFrame
    result_df = pd.DataFrame(processed_rows)
    result_df.to_csv(config.DATA_PROCESSED / 'processed_seasonal_data.csv',index=False)

def process_select(str):
    """通过字典选择操作"""
    if str == "week":
        weekly_stat_process()
    elif str == "season":
        seasonal_process()
    elif str == "unique":
        stat_unique()

def popularity_merge():
    popularity_file = config.DATA_PROCESSED / "popularity_prior_results.csv"
    week_file = config.DATA_PROCESSED / "processed_weekly_data.csv"
    season_file = config.DATA_PROCESSED / "processed_seasonal_data.csv"
    df_popu = pd.read_csv(popularity_file)
    df_week = pd.read_csv(week_file)
    df_season = pd.read_csv(season_file)
    # 获取非零的最小值（第二小的值）
    non_zero_ratios = df_popu['popularity_ratio'][df_popu['popularity_ratio'] > 0]
    second_min_ratio = np.min(non_zero_ratios) if len(non_zero_ratios) > 0 else 0.001
    
    # 将popularity_ratio为0的值替换为第二小的值
    df_popu['popularity_ratio'] = df_popu['popularity_ratio'].replace(0.0, second_min_ratio)
    
    
    # 1. 合并到周数据
    # 选择需要的列：celebrity_name, season, popularity_ratio
    df_popu_merge = df_popu[['celebrity_name', 'season', 'popularity_ratio']]
    
    # 使用左连接将popularity_ratio合并到周数据
    df_week_merged = df_week.merge(df_popu_merge, 
                                   on=['celebrity_name', 'season'], 
                                   how='left')
    
    # 2. 合并到赛季数据
    df_season_merged = df_season.merge(df_popu_merge, 
                                       on=['celebrity_name', 'season'], 
                                       how='left')
    
    # 3. 保存合并后的数据
    df_week_merged.to_csv(config.DATA_PROCESSED / "weekly_data_with_popularity.csv", index=False)
    df_season_merged.to_csv(config.DATA_PROCESSED / "seasonal_data_with_popularity.csv", index=False)
    
    print("数据合并完成！")
    print(f"周数据合并后形状: {df_week_merged.shape}")
    print(f"赛季数据合并后形状: {df_season_merged.shape}")
    print(f"周数据中popularity_ratio缺失值数量: {df_week_merged['popularity_ratio'].isna().sum()}")
    print(f"赛季数据中popularity_ratio缺失值数量: {df_season_merged['popularity_ratio'].isna().sum()}")
    
    return df_week_merged, df_season_merged

if __name__=="__main__":
    # weekly_stat_process()
    stat_unique(RAW_file,"celebrity_homecountry/region")
    # seasonal_process()
    # process_select("week")
    # stat_unique(RAW_file,"ballroom_partner")
    # popularity_merge()