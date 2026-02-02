#one-hot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import config
from utils import log,save_result
# 1. Load Data
try:
    df_raw = pd.read_csv(config.DATA_RAW / '2026_MCM_Problem_C_Data.csv')
    df_partner = pd.read_csv(config.DATA_PROCESSED / 'q3_partner_characteristics.csv')
    df_integrated = pd.read_csv(config.RESULTS_DATA / 'integrated_seasonal_data_with_hypothetical_ranks.csv')
    df_rank = pd.read_csv(config.DATA_PROCESSED / 'rank_data.csv')
except FileNotFoundError:
    print("Error loading files.")

# 2. Data Preparation & Merging

# A. Partner Features
df_partner_features = df_partner[['partner_name', 'avg_judge_score', 'champion_count', 'appearance_count']].copy()
df_partner_features.rename(columns={
    'avg_judge_score': 'partner_hist_score', 
    'champion_count': 'partner_hist_champs',
    'appearance_count': 'partner_experience'
}, inplace=True)

# B. Celebrity Basic Features (Age, Pop, Placement) from Integrated
# We treat 'final_placement' as Target
df_celebs = df_integrated[['celebrity_name', 'season', 'age', 'popularity_ratio', 'final_placement', 'us_born']].copy()
df_celebs['us_born'] = df_celebs['us_born'].astype(int)

# C. Region/Industry Ranks from Rank Data
# Note: rank_data doesn't have season, merging on name. Assuming unique names or consistent attributes.
df_rank_subset = df_rank[['celebrity_name', 'region_rank', 'industry_rank']].drop_duplicates()
df_celebs = pd.merge(df_celebs, df_rank_subset, on='celebrity_name', how='left')

# D. Link Celebrity to Partner
df_link = df_raw[['celebrity_name', 'season', 'ballroom_partner']].drop_duplicates(subset=['celebrity_name', 'season'])
df_level1 = pd.merge(df_celebs, df_link, on=['celebrity_name', 'season'], how='left')

# E. Merge Partner Features
df_final = pd.merge(df_level1, df_partner_features, left_on='ballroom_partner', right_on='partner_name', how='left')

# Drop NA (missing partner stats or ranks)
df_model = df_final.dropna().copy()

# 3. Advanced Feature Engineering

# Calculate Season Context (Size, Means, Stds)
season_stats = df_model.groupby('season').agg({
    'celebrity_name': 'count',
    'age': ['mean', 'std'],
    'popularity_ratio': ['mean', 'std'],
    'partner_hist_score': ['mean', 'std']
}).reset_index()

season_stats.columns = ['season', 'season_size', 'age_mean', 'age_std', 'pop_mean', 'pop_std', 'p_score_mean', 'p_score_std']

# Merge back
df_model = pd.merge(df_model, season_stats, on='season', how='left')

# Create Relative Features (Z-scores)
eps = 1e-6
df_model['z_age'] = (df_model['age'] - df_model['age_mean']) / (df_model['age_std'] + eps)
df_model['z_pop'] = (df_model['popularity_ratio'] - df_model['pop_mean']) / (df_model['pop_std'] + eps)
df_model['z_partner_score'] = (df_model['partner_hist_score'] - df_model['p_score_mean']) / (df_model['p_score_std'] + eps)

# Interactions
df_model['age_x_partner'] = df_model['age'] * df_model['partner_hist_score']
df_model['pop_x_partner'] = df_model['popularity_ratio'] * df_model['partner_hist_score']
df_model['region_x_industry'] = df_model['region_rank'] * df_model['industry_rank']

# Define Features
features = [
    'age', 'popularity_ratio', 'us_born',
    'region_rank', 'industry_rank',
    'partner_hist_score', 'partner_experience', 'partner_hist_champs',
    'season_size', # Important context
    'z_age', 'z_pop', 'z_partner_score',
    'age_x_partner', 'pop_x_partner'
]

X = df_model[features]
y = df_model['final_placement']

# 4. Modeling: Gradient Boosting with Grid Search
# Gradient Boosting often beats Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0]
}
log('start grid_search')
gb = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=5, scoring='r2', n_jobs=1)
grid_search.fit(X, y)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X)
log('grid_search successful')
# Metrics
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print("="*50)
print("  Optimized Gradient Boosting Model Results")
print("="*50)
print(f"Best Params: {grid_search.best_params_}")
print(f"R-squared: {r2:.4f}")
print(f"MSE: {mse:.4f}")

# Feature Importance
importances = pd.DataFrame({
    'Feature': features,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(importances)
result = pd.DataFrame(importances)
save_result(result,'q3_fit_result_importance.csv')
# 5. Visualization (No Jitter)
plt.figure(figsize=(10, 6))

# Color points by Season Size to see if larger seasons are harder to predict
sns.scatterplot(x=y, y=y_pred, hue=df_model['partner_hist_score'], palette='magma', alpha=0.8, s=70)

# Perfect fit line
min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit', linewidth=2)

plt.title('Final Optimized Model: Actual vs Predicted Placement\n(Gradient Boosting + Full Feature Set)')
plt.xlabel('Actual Final Placement')
plt.ylabel('Predicted Final Placement')
plt.legend(title='Partner Score')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(config.RESULTS_FIG / 'optimized_model_evaluation.png')
log("\nPlot saved to optimized_model_evaluation.png")