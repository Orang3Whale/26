import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import config

# 设置中文显示（如果需要）
plt.rcParams['font.family'] = 'Times New Roman'  # 论文标准字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ============================
# 1. 加载数据集 - 提取第27季数据
# ============================
# 文件路径
file_path = r'd:\NUAA\2026\MCM\26\data\processed\processed_weekly_data.csv'

# 读取数据
print("正在加载数据...")
df_raw = pd.read_csv(file_path)
print(f"原始数据形状: {df_raw.shape}")

# 提取第27季的数据
season_num = 27
df_season = df_raw[df_raw['season'] == season_num].copy()
print(f"第{season_num}季数据形状: {df_season.shape}")

# 查看数据列
print(f"可用列: {list(df_season.columns)}")

# 检查是否有第27季数据
if df_season.empty:
    print(f"警告: 没有第{season_num}季的数据!")
    print(f"可用的赛季: {sorted(df_raw['season'].unique())}")
    exit()

# 获取参赛选手列表
celebrities = sorted(df_season['celebrity_name'].unique())
print(f"第{season_num}季参赛选手: {celebrities}")

# 转换数据为宽格式（选手×周数）
df = df_season.pivot(index='celebrity_name', columns='week', values='weekly_avg')
print(f"宽格式数据形状: {df.shape}")

# 重命名列
column_mapping = {col: f'week{col}' for col in df.columns}
df = df.rename(columns=column_mapping)

print("\n=== 第27季数据预览 ===")
print(df.head())
print(f"\n=== 数据形状: ==", df.shape)

# ============================
# 2. 创建多子图可视化
# ============================
fig = plt.figure(figsize=(16, 10))
fig.suptitle(f'Dancing with the Stars - Season {season_num} Performance Analysis', fontsize=16, fontweight='bold')

# --------------------------------------------------
# 子图1: 点阵图（最直观显示参与情况）
# --------------------------------------------------
ax1 = plt.subplot(2, 2, 1)

# 为每个名人绘制得分点
for i, celeb in enumerate(celebrities):
    scores = df.loc[celeb].dropna()
    weeks = scores.index
    week_nums = [int(w.replace('week', '')) for w in weeks]

    # 绘制得分点（大小和颜色表示分数）
    scatter = ax1.scatter(week_nums, [i] * len(scores),
                          s=scores.values * 40,  # 点的大小正比于分数
                          c=scores.values,  # 颜色表示分数
                          cmap='RdYlGn',  # 红-黄-绿色彩映射
                          vmin=0, vmax=10,  # 颜色范围 0-10
                          edgecolor='black',  # 黑色边框
                          linewidth=0.5,
                          alpha=0.8)

    # 绘制淘汰线
    if len(scores) < len(df.columns):
        last_week = week_nums[-1]
        ax1.plot([last_week, last_week + 0.5], [i, i],
                 'r--', linewidth=1, alpha=0.5)

# 美化
ax1.set_yticks(range(len(celebrities)))
ax1.set_yticklabels(celebrities, fontsize=9)
ax1.set_xlabel('Week Number', fontweight='bold')
ax1.set_ylabel('Celebrity', fontweight='bold')
ax1.set_title('Score Distribution by Week', fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')

# 添加颜色条
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Score Value', fontweight='bold')

# --------------------------------------------------
# 子图2: 折线图（显示趋势）
# --------------------------------------------------
ax2 = plt.subplot(2, 2, 2)

# 为每位名人绘制折线
for i, celeb in enumerate(celebrities):
    scores = df.loc[celeb].dropna()
    weeks = scores.index
    week_nums = [int(w.replace('week', '')) for w in weeks]

    # 绘制折线
    line = ax2.plot(week_nums, scores.values,
                    marker='o',
                    linewidth=2,
                    markersize=6,
                    label=celeb,
                    alpha=0.7)

    # 标记淘汰点
    if len(scores) < len(df.columns):
        ax2.plot(week_nums[-1], scores.iloc[-1],
                 'rx', markersize=10, markeredgewidth=2)

ax2.set_xlabel('Week Number', fontweight='bold')
ax2.set_ylabel('Score', fontweight='bold')
ax2.set_title('Score Trends Over Time', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax2.set_ylim(0, 11)

# --------------------------------------------------
# 子图3: 热力图（改进版）
# --------------------------------------------------
ax3 = plt.subplot(2, 2, 3)

# 创建热力图数据
heatmap_data = df.copy()

# 绘制热力图
im = ax3.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto',
                vmin=0, vmax=10, alpha=0.8)

# 添加数值标签
for i in range(len(celebrities)):
    for j in range(len(df.columns)):
        value = heatmap_data.iloc[i, j]
        if not np.isnan(value):
            ax3.text(j, i, f'{value:.1f}',
                     ha='center', va='center',
                     fontsize=8, fontweight='bold',
                     color='black' if value > 5 else 'white')

# 设置坐标轴
ax3.set_xticks(range(len(df.columns)))
ax3.set_xticklabels([f'Week {i + 1}' for i in range(len(df.columns))])
ax3.set_yticks(range(len(celebrities)))
ax3.set_yticklabels(celebrities, fontsize=9)
ax3.set_title('Performance Heatmap', fontweight='bold')
ax3.set_xlabel('Week', fontweight='bold')

# 旋转x轴标签
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

# --------------------------------------------------
# 子图4: 条形图（平均分和参与周数）
# --------------------------------------------------
ax4 = plt.subplot(2, 2, 4)

# 计算统计量
avg_scores = df.mean(axis=1)
weeks_participated = df.count(axis=1)

# 创建分组条形图
x = np.arange(len(celebrities))
width = 0.35

bars1 = ax4.barh(x - width / 2, avg_scores.values, width,
                 label='Average Score', color='skyblue', edgecolor='black')
bars2 = ax4.barh(x + width / 2, weeks_participated.values, width,
                 label='Weeks Participated', color='lightcoral', edgecolor='black')

# 添加数值标签
for i, (avg, weeks) in enumerate(zip(avg_scores, weeks_participated)):
    ax4.text(avg + 0.1, i - width / 2, f'{avg:.1f}',
             va='center', fontsize=8, fontweight='bold')
    ax4.text(weeks + 0.1, i + width / 2, f'{weeks:.0f}',
             va='center', fontsize=8, fontweight='bold')

ax4.set_yticks(x)
ax4.set_yticklabels(celebrities, fontsize=9)
ax4.set_xlabel('Value', fontweight='bold')
ax4.set_title('Average Score vs Participation Weeks', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='x')

# --------------------------------------------------
# 图例说明
# --------------------------------------------------
# 创建自定义图例
legend_elements = [
    Patch(facecolor='red', edgecolor='black', alpha=0.6, label='Low Score (0-3)'),
    Patch(facecolor='yellow', edgecolor='black', alpha=0.6, label='Medium Score (4-7)'),
    Patch(facecolor='green', edgecolor='black', alpha=0.6, label='High Score (8-10)'),
    plt.Line2D([0], [0], marker='x', color='red', label='Elimination Point',
               markersize=10, linewidth=0)
]

fig.legend(handles=legend_elements, loc='lower center',
           bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=9)

# ============================
# 3. 调整布局并保存
# ============================
plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.1)
plt.savefig(f'season_{season_num}_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================
# 4. 额外：淘汰顺序可视化
# ============================
print("\n=== 淘汰顺序 ===")
elimination_order = []
for week in range(1, len(df.columns) + 1):
    week_col = f'week{week}'
    for celeb in celebrities:
        if celeb not in elimination_order:
            if pd.isna(df.loc[celeb, week_col]):
                elimination_order.append(celeb)
                print(f"Week {week}淘汰: {celeb}")

# 添加最后剩下的选手到淘汰顺序
remaining = [c for c in celebrities if c not in elimination_order]
elimination_order.extend(remaining)
print(f"最终排名 (冠军→亚军): {remaining}")

# 创建淘汰时间线图
fig2, ax = plt.subplots(figsize=(10, 6))

for i, celeb in enumerate(elimination_order[::-1]):  # 反转，让最后淘汰的在上面
    scores = df.loc[celeb].dropna()
    weeks = [int(w.replace('week', '')) for w in scores.index]

    # 绘制参赛时间线
    if len(weeks) > 0:
        start = weeks[0]
        end = weeks[-1]
        ax.plot([start, end], [i, i], linewidth=3,
                color=plt.cm.tab10(i % 10), marker='o', markersize=8)

        # 添加姓名和淘汰周
        ax.text(end + 0.2, i, f"{celeb} (Week {end})",
                va='center', fontsize=9, fontweight='bold')

ax.set_yticks([])
ax.set_xlabel('Week Number', fontweight='bold')
ax.set_title(f'Season {season_num} Elimination Timeline (Latest → Earliest)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, len(df.columns) + 1.5)

plt.tight_layout()
plt.savefig(f'season_{season_num}_elimination_timeline.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================
# 5. 数据统计摘要
# ============================
print("\n" + "=" * 50)
print(f"SEASON {season_num} DATA SUMMARY")
print("=" * 50)

print(f"\n1. 第{season_num}季整体统计:")
print(f"- 参赛选手数量: {len(celebrities)}")
print(f"- 比赛周数: {len(df.columns)}")
print(f"- 平均得分: {df.mean().mean():.2f}")
print(f"- 最高得分: {df.max().max():.1f}")
print(f"- 最低得分: {df[df > 0].min().min():.1f}")

print("\n2. 每位选手的统计信息:")
summary_df = pd.DataFrame({
    'Average Score': df.mean(axis=1).round(2),
    'Highest Score': df.max(axis=1),
    'Weeks Participated': df.count(axis=1),
    'Eliminated at Week': [max([int(w.replace('week', '')) for w in df.loc[c].dropna().index]) if df.loc[c].notna().any() else 0 for c in celebrities]
})
print(summary_df.sort_values('Weeks Participated', ascending=False))

print("\n3. 每周的统计信息:")
weekly_stats = pd.DataFrame({
    'Average Score': df.mean().round(2),
    'Participants': df.count(),
    'High Score': df.max(),
    'Low Score': df.min()
})
print(weekly_stats)

print("\n4. 赛季冠军分析:")
winner = avg_scores.idxmax()
print(f"🏆 平均分最高: {winner} ({avg_scores.max():.2f}分)")
longest = weeks_participated.idxmax()
print(f"⏱️  坚持最久: {longest} ({weeks_participated.max()}周)")
print(f"🎯 最终排名: {', '.join(remaining[::-1])}")

print("\n分析完成！图表已保存。")