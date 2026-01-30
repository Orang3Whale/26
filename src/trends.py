import pandas as pd
import time
import random
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import config
from utils import log

# ================= 配置区域 =================
# 1. 设置锚点关键词（用于归一化不同选手的热度）
ANCHOR_KEYWORD = "Dancing with the Stars"

# 2. 读取你的数据文件
# 假设CSV里有 'celebrity_name' 和 'season' 列
input_file = config.DATA_RAW / "2026_MCM_Problem_C_Data.csv"
output_file = config.DATA_PROCESSED / 'popularity_prior_results.csv'

# 3. 必须手动补充每个赛季的起止时间（因为CSV里没有）
# 格式: 'Season': ('YYYY-MM-DD', 'YYYY-MM-DD')
# 建议去Wikipedia查每个赛季的首播和决赛日期，这里仅作示例
SEASON_DATES = {
    1: ('2005-06-01', '2005-07-06'),
    2: ('2006-01-05', '2006-02-24'),
    3: ('2006-09-12', '2006-11-15'),
    4: ('2007-03-19', '2007-05-22'),
    5: ('2007-09-24', '2007-11-27'),
    6: ('2008-03-17', '2008-05-20'),
    7: ('2008-09-22', '2008-11-25'),
    8: ('2009-03-09', '2009-05-19'),
    9: ('2009-09-21', '2009-11-24'),
    10: ('2010-03-22', '2010-05-25'),
    11: ('2010-09-20', '2010-11-23'),
    12: ('2011-03-21', '2011-05-24'),
    13: ('2011-09-19', '2011-11-22'),
    14: ('2012-03-19', '2012-05-22'),
    15: ('2012-09-24', '2012-11-27'),
    16: ('2013-03-18', '2013-05-21'),
    17: ('2013-09-16', '2013-11-26'),
    18: ('2014-03-17', '2014-05-20'),
    19: ('2014-09-15', '2014-11-25'),
    20: ('2015-03-16', '2015-05-19'),
    21: ('2015-09-14', '2015-11-24'),
    22: ('2016-03-21', '2016-05-24'),
    23: ('2016-09-12', '2016-11-22'),
    24: ('2017-03-20', '2017-05-23'),
    25: ('2017-09-18', '2017-11-21'),
    26: ('2018-04-30', '2018-05-21'),
    27: ('2018-09-24', '2018-11-19'),
    28: ('2019-09-16', '2019-11-25'),
    29: ('2020-09-14', '2020-11-23'),
    30: ('2021-09-20', '2021-11-22'),
    31: ('2022-09-19', '2022-11-21'),
    32: ('2023-09-26', '2023-12-05'),
    33: ('2024-09-17', '2024-11-26'),
    34: ('2025-09-16', '2025-12-02')
}

# ===========================================

def get_google_trends_data():
    # 初始化 pytrends
    # hl='en-US' 确保语言环境，tz=360 是时区偏移
    pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25))
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return

    results = []
    
    # 去重：同一个选手在一个赛季只搜一次
    unique_entries = df[['season', 'celebrity_name']].drop_duplicates()
    
    print(f"开始爬取 {len(unique_entries)} 名选手的数据...")
    
    for index, row in unique_entries.iterrows():
        season = row['season']
        name = row['celebrity_name']
        
        # 检查是否有该赛季的日期数据
        if season not in SEASON_DATES:
            print(f"跳过 {name} (Season {season}): 缺少日期配置")
            continue
            
        start_date, end_date = SEASON_DATES[season]
        timeframe = f"{start_date} {end_date}"
        
        # 构建关键词列表：[选手名, 锚点名]
        # 注意：有些名字可能有特殊字符，建议简单清洗
        clean_name = name.replace("'", "") 
        kw_list = [clean_name, ANCHOR_KEYWORD]
        
        try:
            # 发送请求
            pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo='US', gprop='')
            
            # 获取随时间变化的热度
            interest_over_time = pytrends.interest_over_time()
            
            if not interest_over_time.empty:
                # === 核心逻辑：计算相对热度 ===
                # 取该时间段内的平均值
                mean_scores = interest_over_time.mean()
                celeb_score = mean_scores[clean_name]
                anchor_score = mean_scores[ANCHOR_KEYWORD]
                
                # 计算比率：相对于节目的热度
                # 如果 anchor_score 为 0 (极少见), 则直接用 celeb_score
                if anchor_score > 0:
                    relative_ratio = celeb_score / anchor_score
                else:
                    relative_ratio = celeb_score # 降级处理
                
                print(f"[{index}/{len(unique_entries)}] {name} (S{season}): Ratio = {relative_ratio:.4f}")
                
                results.append({
                    'season': season,
                    'celebrity_name': name,
                    'google_trend_raw': celeb_score,
                    'anchor_score': anchor_score,
                    'popularity_ratio': relative_ratio # <--- 这是你要进模型的参数
                })
            else:
                print(f"[{index}] {name}: 无数据返回")
                results.append({'season': season, 'celebrity_name': name, 'popularity_ratio': 0})

            # === 关键：随机延时防止封IP ===
            # 建议设置在 5-10 秒之间，跑完几百个数据需要一小时，但安全
            sleep_time = random.uniform(5, 10)
            time.sleep(sleep_time)

        except Exception as e:
            print(f"Error crawling {name}: {e}")
            # 如果遇到 429 Too Many Requests，休息更久
            if "429" in str(e):
                print("触发限流，暂停 60 秒...")
                time.sleep(60)
            continue

    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"完成！数据已保存至 {output_file}")

if __name__ == "__main__":
    get_google_trends_data()