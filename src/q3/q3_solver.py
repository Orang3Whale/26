import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.miscmodels.ordinal_model import OrderedModel
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载所有数据
# 名人表现数据（含特征）
seasonal_df = pd.read_csv('/mnt/seasonal_data_with_popularity.csv')
# 舞伴数据（新上传）
partner_df = pd.read_csv('/mnt/2026_MCM_Problem_C_Data.csv')

# 2. 探索舞伴数据结构
print("=== 舞伴数据基本信息 ===")
print(f"数据形状：{partner_df.shape}")
print(f"列名：{list(partner_df.columns)}")
print(f"\n缺失值统计：")
print(partner_df.isnull().sum()[partner_df.isnull().sum() > 0])
print(f"\n前5行数据：")
print(partner_df.head())

# 3. 数据关联（关键：确保赛季+名人能匹配）
# 假设两表共有关联键：season（赛季）、celebrity_name（名人姓名）
# 先统一关联键格式（去重、统一大小写）
seasonal_df['celebrity_name'] = seasonal_df['celebrity_name'].str.strip().str.lower()
partner_df['celebrity_name'] = partner_df['celebrity_name'].str.strip().str.lower()
seasonal_df['season'] = seasonal_df['season'].astype(int)
partner_df['season'] = partner_df['season'].astype(int)

# 合并数据（inner join：只保留双方都有的记录）
merged_df = pd.merge(
    seasonal_df, 
    partner_df, 
    on=['season', 'celebrity_name'], 
    how='inner',
    suffixes=('_celebrity', '_partner')  # 区分同名列（如age可能同时存在于名人/舞伴数据）
)

print(f"\n=== 合并后数据信息 ===")
print(f"合并后数据形状：{merged_df.shape}")
print(f"保留赛季数：{merged_df['season'].nunique()}")
print(f"保留名人数量：{merged_df['celebrity_name'].nunique()}")
print(f"舞伴相关字段：{[col for col in merged_df.columns if '_partner' in col or 'partner' in col.lower()]}")

# 1. 提取并清洗舞伴核心字段（根据实际数据调整字段名，此处以常见字段为例）
# 假设舞伴数据包含：partner_name（舞伴姓名）、partner_age（舞伴年龄）、partner_seasons（舞伴累计合作季数）
# partner_champions（舞伴带教过的冠军数）、partner_avg_score（舞伴过往带教平均评分）

# 定义舞伴特征列表（根据实际数据调整）
partner_feature_cols = [
    'partner_age', 'partner_seasons', 'partner_champions', 'partner_avg_score',
    'partner_gender'  # 舞伴性别（若有）
]

# 2. 处理缺失值（根据字段类型选择填充方式）
# 数值型字段：用中位数填充（避免极值影响）
for col in ['partner_age', 'partner_seasons', 'partner_champions', 'partner_avg_score']:
    if col in merged_df.columns:
        merged_df[col].fillna(merged_df[col].median(), inplace=True)

# 分类字段（如partner_gender）：用众数填充
if 'partner_gender' in merged_df.columns:
    merged_df['partner_gender'].fillna(merged_df['partner_gender'].mode()[0], inplace=True)

# 3. 构建舞伴衍生特征
# 3.1 舞伴经验等级：根据合作季数分档（0-2季=新手，3-5季=中级，≥6季=资深）
merged_df['partner_experience_level'] = pd.cut(
    merged_df['partner_seasons'],
    bins=[0, 2, 5, np.inf],
    labels=['Novice', 'Intermediate', 'Senior']
)

# 3.2 舞伴冠军经验：是否带教过冠军（0=无，1=有）
merged_df['partner_has_champion'] = (merged_df['partner_champions'] > 0).astype(int)

# 3.3 名人-舞伴年龄差：绝对值（可能影响配合默契度）
if 'age_celebrity' in merged_df.columns and 'partner_age' in merged_df.columns:
    merged_df['age_diff_celebrity_partner'] = abs(merged_df['age_celebrity'] - merged_df['partner_age'])

# 3.4 舞伴历史表现：标准化（消除量纲）
scaler_partner = StandardScaler()
partner_numeric_cols = ['partner_age', 'partner_seasons', 'partner_champions', 'partner_avg_score', 'age_diff_celebrity_partner']
partner_numeric_cols = [col for col in partner_numeric_cols if col in merged_df.columns]
merged_df[[f'{col}_std' for col in partner_numeric_cols]] = scaler_partner.fit_transform(
    merged_df[partner_numeric_cols]
)

# 4. 分类变量编码（独热编码）
# 舞伴性别、经验等级
encoder = OneHotEncoder(sparse_output=False, drop='first')
# 编码舞伴性别（若有）
if 'partner_gender' in merged_df.columns:
    gender_encoded = encoder.fit_transform(merged_df[['partner_gender']])
    gender_df = pd.DataFrame(gender_encoded, columns=[f'partner_gender_{cat}' for cat in encoder.categories_[0][1:]])
else:
    gender_df = pd.DataFrame()

# 编码舞伴经验等级
exp_encoded = encoder.fit_transform(merged_df[['partner_experience_level']])
exp_df = pd.DataFrame(exp_encoded, columns=[f'partner_exp_{cat}' for cat in encoder.categories_[0][1:]])

# 合并舞伴编码特征
partner_encoded_features = pd.concat([gender_df, exp_df], axis=1)

print(f"\n=== 舞伴特征工程完成 ===")
print(f"舞伴数值特征（标准化后）：{[f'{col}_std' for col in partner_numeric_cols]}")
print(f"舞伴编码特征：{list(partner_encoded_features.columns)}")

# 1. 加载之前的名人特征（已处理完成的数值+编码特征）
# 名人数值特征（标准化后）：年龄、人气比、赛季
celebrity_numeric_std_cols = [col for col in merged_df.columns if col in ['age_std', 'popularity_std', 'season_std']]
# 名人编码特征：职业、是否美国出生、是否进前三
celebrity_encoded_cols = [col for col in merged_df.columns if col.startswith('industry_') or col in ['us_born_bin', 'top_3_bin']]

# 2. 整合所有特征
# 名人特征
celebrity_features = merged_df[celebrity_numeric_std_cols + celebrity_encoded_cols].copy()
# 舞伴特征（标准化数值+编码特征）
partner_features = merged_df[[f'{col}_std' for col in partner_numeric_cols]].copy()
partner_features = pd.concat([partner_features, partner_encoded_features], axis=1)

# 3. 最终特征矩阵（名人+舞伴）
X = pd.concat([celebrity_features, partner_features], axis=1)

# 4. 目标变量（与之前一致）
y_score = merged_df['avg_score']  # 平均评委评分
y_rank = merged_df['final_placement']  # 最终排名（数值越小越好）
y_weeks = merged_df['weeks_survived']  # 存活周数

# 5. 划分训练集/测试集（8:2，固定随机种子确保可复现）
X_train, X_test, y_score_train, y_score_test = train_test_split(X, y_score, test_size=0.2, random_state=42)
X_train_rank, X_test_rank, y_rank_train, y_rank_test = train_test_split(X, y_rank, test_size=0.2, random_state=42)
X_train_weeks, X_test_weeks, y_weeks_train, y_weeks_test = train_test_split(X, y_weeks, test_size=0.2, random_state=42)

print(f"\n=== 最终特征矩阵信息 ===")
print(f"特征矩阵形状（训练集）：{X_train.shape}")
print(f"名人特征数量：{celebrity_features.shape[1]}")
print(f"舞伴特征数量：{partner_features.shape[1]}")
print(f"目标变量样本量：评分={len(y_score)}, 排名={len(y_rank)}, 存活周数={len(y_weeks)}")