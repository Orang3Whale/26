#本代码用于爬取收视人数
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import config
def scrape_dwts_ratings():
    # 1. 设置目标URL
    url = "https://en.wikipedia.org/wiki/Dancing_with_the_Stars_(American_TV_series)"
    
    # 2. 发送请求
    print(f"正在获取页面: {url} ...")
    try:
        response = requests.get(url)
        response.raise_for_status() # 检查请求是否成功
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return

    # 3. 解析HTML
    soup = BeautifulSoup(response.content, 'html.parser')

    # 4. 定位 "U.S. Nielsen ratings" 章节
    # 维基百科的章节标题通常在 span 标签中，且 id 为 "U.S._Nielsen_ratings"
    section_header = soup.find(id="U.S._Nielsen_ratings")
    
    if not section_header:
        # 尝试备用的 ID 写法（防止维基百科微调格式）
        section_header = soup.find(id="U.S._Nielsen_ratings_2")
        
    if not section_header:
        print("错误：未找到 'U.S. Nielsen ratings' 章节，请检查网页结构是否变化。")
        return

    # 5. 查找该标题后的第一个表格
    # find_next 会在HTML流中寻找下一个符合条件的元素
    rating_table = section_header.find_next("table", class_="wikitable")

    if not rating_table:
        print("错误：在指定章节下未找到收视率表格。")
        return

    print("已找到收视率表格，正在解析数据...")

    # 6. 使用 Pandas 快速读取表格
    # pd.read_html 返回一个 DataFrame 列表，我们取第一个 [0]
    # str(rating_table) 将 BeautifulSoup 对象转为字符串传给 pandas
    try:
        df = pd.read_html(str(rating_table))[0]
    except ValueError as e:
        print(f"表格解析失败: {e}")
        return

    # 7. 数据清洗
    # 定义一个函数去除维基百科的引用标签 (例如: [12], [a])
    def clean_text(text):
        if isinstance(text, str):
            # 去除方括号及其内容
            text = re.sub(r'\[.*?\]', '', text)
            # 去除不可见字符
            text = text.strip()
        return text

    # 将清洗函数应用到整个 DataFrame
    df = df.applymap(clean_text)

    # 8. 打印预览并保存
    print("\n--- 数据预览 (前 5 行) ---")
    print(df.head())

    output_file = config.DATA_PROCESSED / "dwts_ratings.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n成功！数据已保存至: {output_file}")

if __name__ == "__main__":
    scrape_dwts_ratings()