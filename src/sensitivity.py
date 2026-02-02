import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import config
from utils import log,save_result

# 1. Load Data
# We need Raw Data for structure, Integ Data for Pop/Age, Partner Data for Partner Score
dfs = {}
for f in [config.DATA_RAW / '2026_MCM_Problem_C_Data.csv', config.RESULTS_DATA / 'integrated_seasonal_data_with_hypothetical_ranks.csv', config.DATA_PROCESSED / 'q3_partner_characteristics.csv']:
    if os.path.exists(f):
        try:
            dfs[f] = pd.read_csv(f)
        except: pass

df_raw = dfs.get(config.DATA_RAW / '2026_MCM_Problem_C_Data.csv')
df_integ = dfs.get(config.RESULTS_DATA / 'integrated_seasonal_data_with_hypothetical_ranks.csv')
df_partner = dfs.get(config.DATA_PROCESSED / 'q3_partner_characteristics.csv')

if df_raw is not None and df_integ is not None:
    # 2. Build Maps
    pop_map = df_integ.set_index(['celebrity_name', 'season'])['popularity_ratio'].to_dict()
    partner_score_map = {}
    if df_partner is not None:
        partner_score_map = df_partner.set_index('partner_name')['avg_judge_score'].to_dict()

    # Elim Parser
    def parse_elim(res):
        if isinstance(res, str) and res.startswith("Eliminated Week"):
            try: return int(res.split()[-1])
            except: return 999
        return 999 
    df_raw['elim_week'] = df_raw['results'].apply(parse_elim)

    # Max Week
    week_cols = [c for c in df_raw.columns if 'week' in c and 'judge' in c]
    max_week = 0
    for c in week_cols:
        m = re.search(r'week(\d+)_', c)
        if m and int(m.group(1)) > max_week: max_week = int(m.group(1))
    
    # 3. Define Evaluation Function
    def evaluate_sensitivity(w_pop, w_judge, w_partner):
        results = []
        seasons = df_raw['season'].unique()
        
        for s in seasons:
            season_df = df_raw[df_raw['season'] == s]
            for w in range(1, max_week + 1):
                cols = [f'week{w}_judge{j}_score' for j in range(1, 4)]
                cols = [c for c in cols if c in df_raw.columns]
                if not cols: continue
                
                # Weekly Data
                weekly_scores = season_df.set_index('celebrity_name')[cols].sum(axis=1).to_dict()
                if not weekly_scores: continue
                
                elim_list = season_df[season_df['elim_week'] == w]['celebrity_name'].tolist()
                eliminated_name = elim_list[0] if elim_list else None
                if eliminated_name is None: continue

                active_contestants = [name for name, sc in weekly_scores.items() if sc > 0]
                if not active_contestants: continue
                
                total_judge_score = sum(weekly_scores.values())
                
                # Calculate Scores
                model_scores = {}
                for name in active_contestants:
                    try: partner = season_df[season_df['celebrity_name']==name]['ballroom_partner'].values[0]
                    except: partner = "Unknown"
                    
                    pop = pop_map.get((name, s), 0.1)
                    norm_judge = weekly_scores[name] / total_judge_score if total_judge_score > 0 else 0
                    p_score = partner_score_map.get(partner, 7.0)
                    
                    # Formula
                    # Base (0.5) is constant, Pop_Exp (1.5) is constant for this test
                    score = 0.5 + (w_pop * pow(pop, 1.5)) + (w_judge * norm_judge) + (w_partner * p_score)
                    model_scores[name] = score
                
                # Check Rank
                # Sort Low to High (Lowest Score = Eliminated)
                sorted_names = sorted(model_scores, key=model_scores.get)
                try: 
                    rank = sorted_names.index(eliminated_name) # 0 is bottom
                    is_correct_top3 = 1 if rank < 3 else 0
                except: 
                    is_correct_top3 = 0
                
                results.append(is_correct_top3)
        
        if not results: return 0
        return sum(results) / len(results)

    # 4. Run Sensitivity Analysis (One-at-a-Time)
    sensitivity_data = []
    
    # Defaults
    def_pop = 15.0
    def_judge = 5.0
    def_partner = 0.5
    
    # A. Vary Popularity Weight
    pop_range = np.linspace(0, 30, 7)
    for val in pop_range:
        acc = evaluate_sensitivity(val, def_judge, def_partner)
        sensitivity_data.append({'Parameter': 'Popularity Weight', 'Value': val, 'Accuracy': acc})
        
    # B. Vary Judge Weight
    judge_range = np.linspace(0, 20, 9) # 0 to 20
    for val in judge_range:
        acc = evaluate_sensitivity(def_pop, val, def_partner)
        sensitivity_data.append({'Parameter': 'Judge Weight', 'Value': val, 'Accuracy': acc})
        
    # C. Vary Partner Weight
    partner_range = np.linspace(0, 2.0, 9) # 0 to 2.0
    for val in partner_range:
        acc = evaluate_sensitivity(def_pop, def_judge, val)
        sensitivity_data.append({'Parameter': 'Partner Weight', 'Value': val, 'Accuracy': acc})
        
    df_sens = pd.DataFrame(sensitivity_data)
    
    # 5. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    # Plot Pop
    data_pop = df_sens[df_sens['Parameter'] == 'Popularity Weight']
    sns.lineplot(data=data_pop, x='Value', y='Accuracy', ax=axes[0], marker='o', color='purple', linewidth=2)
    axes[0].set_title('Sensitivity: Popularity Weight')
    axes[0].axvline(def_pop, color='gray', linestyle='--', label='Baseline (15.0)')
    axes[0].grid(True)
    
    # Plot Judge
    data_judge = df_sens[df_sens['Parameter'] == 'Judge Weight']
    sns.lineplot(data=data_judge, x='Value', y='Accuracy', ax=axes[1], marker='s', color='blue', linewidth=2)
    axes[1].set_title('Sensitivity: Judge Weight')
    axes[1].axvline(def_judge, color='gray', linestyle='--', label='Baseline (5.0)')
    axes[1].grid(True)
    
    # Plot Partner
    data_partner = df_sens[df_sens['Parameter'] == 'Partner Weight']
    sns.lineplot(data=data_partner, x='Value', y='Accuracy', ax=axes[2], marker='^', color='green', linewidth=2)
    axes[2].set_title('Sensitivity: Partner Weight')
    axes[2].axvline(def_partner, color='gray', linestyle='--', label='Baseline (0.5)')
    axes[2].grid(True)
    
    axes[0].set_ylabel('Top-3 Prediction Accuracy')
    plt.suptitle('Sensitivity Analysis of V4 Model Parameters', fontsize=16)
    plt.tight_layout()
    plt.savefig(config.RESULTS_FIG / 'sensitivity_analysis.png')
    
    print("Sensitivity Analysis Completed.")
    print("Results head:")
    print(df_sens.head())
    
    # Find optimal values (just for info)
    best_row = df_sens.loc[df_sens['Accuracy'].idxmax()]
    print(f"\nBest observed configuration in single sweep: {best_row['Parameter']} = {best_row['Value']} (Acc: {best_row['Accuracy']:.2%})")

else:
    print("Missing data for sensitivity analysis.")