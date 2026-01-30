import pandas as pd
import time
import random
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import config
from utils import log
import os
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
    pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25))
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return

    # ================= 核心修改 1: 读取已有进度 =================
    results = []
    scraped_set = set() # 用于存放 (season, name) 的指纹，用于快速查找

    if os.path.exists(output_file):
        try:
            print(f"检测到已有文件 {output_file}，正在读取进度...")
            existing_df = pd.read_csv(output_file)
            
            # 将已有的数据加载到 results 列表，防止覆盖旧数据
            results = existing_df.to_dict('records')
            
            # 创建去重指纹集合
            # 注意：这里确保 season 是整数类型，和后面遍历时保持一致
            scraped_set = set(zip(existing_df['season'], existing_df['celebrity_name']))
            print(f"已成功加载 {len(scraped_set)} 条历史数据，将跳过这些选手。")
        except Exception as e:
            print(f"读取历史文件出错: {e}，将重新开始爬取。")
            results = []
            scraped_set = set()
    else:
        print("未检测到历史文件，将开始全新爬取。")
    # ==========================================================

    # 去重：同一个选手在一个赛季只搜一次
    unique_entries = df[['season', 'celebrity_name']].drop_duplicates()
    
    print(f"总任务量: {len(unique_entries)} 名选手...")
    
    for index, row in unique_entries.iterrows():
        season = row['season']
        name = row['celebrity_name']
        
        # ================= 核心修改 2: 跳过已存在的 =================
        if (season, name) in scraped_set:
            # 可以在这里打印日志，也可以为了清屏选择不打印
            # print(f"跳过已存在: {name} (S{season})") 
            continue
        # ==========================================================
        
        # 检查是否有该赛季的日期数据
        if season not in SEASON_DATES:
            print(f"跳过 {name} (Season {season}): 缺少日期配置")
            continue
            
        start_date, end_date = SEASON_DATES[season]
        timeframe = f"{start_date} {end_date}"
        
        # 构建关键词列表
        clean_name = name.replace("'", "") 
        kw_list = [clean_name, ANCHOR_KEYWORD]
        
        print(f"正在爬取 [{index}/{len(unique_entries)}]: {name} (S{season})...")

        try:
            # 发送请求
            pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo='US', gprop='')
            
            # 获取随时间变化的热度
            interest_over_time = pytrends.interest_over_time()
            
            current_result = {}

            if not interest_over_time.empty:
                # === 计算相对热度 ===
                mean_scores = interest_over_time.mean()
                
                # 安全获取数据，防止有些时候返回的列名不一致
                if clean_name in mean_scores and ANCHOR_KEYWORD in mean_scores:
                    celeb_score = mean_scores[clean_name]
                    anchor_score = mean_scores[ANCHOR_KEYWORD]
                    
                    if anchor_score > 0:
                        relative_ratio = celeb_score / anchor_score
                    else:
                        relative_ratio = celeb_score 
                    
                    print(f"  -> 成功: Ratio = {relative_ratio:.4f}")
                    
                    current_result = {
                        'season': season,
                        'celebrity_name': name,
                        'google_trend_raw': celeb_score,
                        'anchor_score': anchor_score,
                        'popularity_ratio': relative_ratio
                    }
                else:
                    print(f"  -> 数据异常: 返回数据中缺少列")
                    current_result = {'season': season, 'celebrity_name': name, 'popularity_ratio': 0}

            else:
                print(f"  -> 无数据返回 (可能由于搜索量过低)")
                current_result = {'season': season, 'celebrity_name': name, 'popularity_ratio': 0}

            # 添加到总结果中
            results.append(current_result)

            # ================= 核心修改 3: 实时保存 =================
            # 每爬取成功一个，就保存一次文件。
            # 这样即使程序崩溃，也只损失当前这一条，不用重头再来。
            pd.DataFrame(results).to_csv(output_file, index=False)
            # ======================================================

            # 随机延时
            sleep_time = random.uniform(5, 10)
            time.sleep(sleep_time)

        except Exception as e:
            print(f"Error crawling {name}: {e}")
            if "429" in str(e):
                print("触发限流 (429)，暂停 60 秒...")
                time.sleep(60)
            continue

    print(f"全部完成！最终数据已保存至 {output_file}")

if __name__ == "__main__":
    get_google_trends_data()