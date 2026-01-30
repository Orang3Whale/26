#本代码进行数据处理
from utils import log
import config
import pandas as pd
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
    

if __name__=="__main__":

    stat_unique(RAW_file,"results")