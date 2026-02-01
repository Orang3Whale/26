#本代码用来计算相关度矩阵
import pandas as pd
import numpy as np
from scipy import stats

RAW_DATA = 'data/raw/2026_MCM_Problem_C_Data.csv'#含舞伴类别的原始数据

def partner_character():
    """
    舞伴特征的生成
    统计舞伴的参赛次数，所带平均成绩（排名），是否带过冠军
    
    返回:
        DataFrame: 包含舞伴特征的表格
    """
    # 1. 读取原始数据
    df = pd.read_csv(RAW_DATA)
    
    # 2. 数据预处理
    # 解析淘汰周数，将文本结果转换为数值
    def parse_elimination(res):
        if isinstance(res, str):
            if res.startswith("Eliminated Week"):
                return int(res.split()[-1])
            elif "Place" in res:
                return 100  # 代表进入决赛，从未被淘汰
            elif res == "Withdrew":
                return -1   # 特殊处理
        return 100
    
    df['elim_week'] = df['results'].apply(parse_elimination)
    
    # 3. 按舞伴分组统计特征
    partner_stats = []
    
    for partner_name in df['ballroom_partner'].unique():
        if pd.isna(partner_name):
            continue
            
        # 获取该舞伴的所有参赛记录
        partner_data = df[df['ballroom_partner'] == partner_name]
        
        # 基本统计
        appearance_count = len(partner_data)  # 参赛次数
        
        # 平均排名（placement越小越好）
        avg_placement = partner_data['placement'].mean()
        
        # 是否带过冠军（placement == 1）
        has_champion = 1 if (partner_data['placement'] == 1).any() else 0
        
        # 冠军数量统计（placement == 1的次数）
        champion_count = (partner_data['placement'] == 1).sum()
        
        

        # 平均淘汰周数（越大表示成绩越好）
        avg_elim_week = partner_data['elim_week'].mean()
        
        # 最佳成绩（最小placement）
        best_placement = partner_data['placement'].min()
        
        # 最差成绩（最大placement）
        worst_placement = partner_data['placement'].max()
        
        # 稳定性（placement的标准差，越小越稳定）
        placement_std = partner_data['placement'].std()
        
        # 参赛赛季数
        unique_seasons = partner_data['season'].nunique()
        
        # 计算平均每周评委分数（需要处理N/A值）
        judge_columns = [col for col in df.columns if 'judge' in col and 'score' in col]
        avg_scores = []
        
        for _, row in partner_data.iterrows():
            week_scores = []
            for col in judge_columns:
                score = row[col]
                if pd.notna(score) and score != 'N/A' and score != 0:
                    try:
                        week_scores.append(float(score))
                    except:
                        pass
            if week_scores:
                avg_scores.append(np.mean(week_scores))
        
        avg_judge_score = np.mean(avg_scores) if avg_scores else 0
        
        # 收集统计结果
        partner_stats.append({
            'partner_name': partner_name,
            'appearance_count': appearance_count,      # 参赛次数
            'avg_placement': avg_placement,            # 平均排名
            'has_champion': has_champion,              # 是否带过冠军
            'champion_count': champion_count,          # 冠军数量
            'avg_elim_week': avg_elim_week,            # 平均淘汰周数
            'best_placement': best_placement,          # 最佳排名
            'worst_placement': worst_placement,        # 最差排名
            'placement_std': placement_std,            # 排名稳定性
            'unique_seasons': unique_seasons,          # 参赛赛季数
            'avg_judge_score': avg_judge_score         # 平均评委分数
        })
    
    # 4. 转换为DataFrame并排序
    partner_df = pd.DataFrame(partner_stats)
    partner_df = partner_df.sort_values('appearance_count', ascending=False)
    
    # 5. 输出结果
    print("舞伴特征统计完成！")
    print(f"共统计了 {len(partner_df)} 位舞伴")
    print("\n参赛次数最多的前10位舞伴：")
    print(partner_df.head(10)[['partner_name', 'appearance_count', 'avg_placement', 'has_champion']])
    
    # 6. 计算斯皮尔曼相关系数
    print("\n" + "="*50)
    print("斯皮尔曼相关系数分析")
    print("="*50)
    
    # 计算参赛次数与平均排名的斯皮尔曼相关系数
    if len(partner_df) >= 2:
        # 参赛次数 vs 平均排名
        spearman_corr_placement, p_value_placement = stats.spearmanr(
            partner_df['appearance_count'], 
            partner_df['avg_placement']
        )
        
        # 参赛次数 vs 平均评委分数
        spearman_corr_score, p_value_score = stats.spearmanr(
            partner_df['appearance_count'], 
            partner_df['avg_judge_score']
        )
        
        # 参赛次数 vs 冠军数量
        spearman_corr_champion, p_value_champion = stats.spearmanr(
            partner_df['appearance_count'], 
            partner_df['champion_count']
        )
        
        # 参赛次数 vs 平均淘汰周数
        spearman_corr_elim, p_value_elim = stats.spearmanr(
            partner_df['appearance_count'], 
            partner_df['avg_elim_week']
        )
        
        print(f"参赛次数 vs 平均排名: 相关系数 = {spearman_corr_placement:.4f}, p值 = {p_value_placement:.4f}")
        print(f"参赛次数 vs 平均评委分数: 相关系数 = {spearman_corr_score:.4f}, p值 = {p_value_score:.4f}")
        print(f"参赛次数 vs 冠军数量: 相关系数 = {spearman_corr_champion:.4f}, p值 = {p_value_champion:.4f}")
        print(f"参赛次数 vs 平均淘汰周数: 相关系数 = {spearman_corr_elim:.4f}, p值 = {p_value_elim:.4f}")
        
        # 解释相关系数的含义
        print("\n相关系数解释:")
        print("- 正值: 两个变量正相关（一个增加，另一个也增加）")
        print("- 负值: 两个变量负相关（一个增加，另一个减少）")
        print("- 接近0: 两个变量没有线性关系")
        print("- p值 < 0.05: 相关性统计显著")
        
        # 保存相关系数结果
        corr_results = {
            'correlation_type': ['appearance_count', '参赛次数_评委分数', '参赛次数_冠军数量', '参赛次数_淘汰周数'],
            'spearman_correlation': [spearman_corr_placement, spearman_corr_score, 
                                   spearman_corr_champion, spearman_corr_elim],
            'p_value': [p_value_placement, p_value_score, p_value_champion, p_value_elim],
            'significance': ['显著' if p < 0.05 else '不显著' for p in 
                           [p_value_placement, p_value_score, p_value_champion, p_value_elim]]
        }
        
        corr_df = pd.DataFrame(corr_results)
        corr_output_file = 'data/processed/q3_partner_correlations.csv'
        corr_df.to_csv(corr_output_file, index=False)
        print(f"\n相关系数结果已保存至: {corr_output_file}")
    else:
        print("数据量不足，无法计算相关系数")
    
    # 7. 保存结果
    output_file = 'data/processed/q3_partner_characteristics.csv'
    partner_df.to_csv(output_file, index=False)
    print(f"\n舞伴特征数据已保存至: {output_file}")
    
    return partner_df

if __name__ == "__main__":
    partner_character()