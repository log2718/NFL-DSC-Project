import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Old Data/merged_nfl_data.csv")

# Win Percentage Distribution Graph

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='W - L %', bins=30, kde=True)
plt.axvline(x=0.500, color='red', linestyle='--', label='Break-even (8-9 wins)')
plt.axvline(x=0.563, color='green', linestyle='--', label='Playoff Threshold (~10 wins)')
plt.title('Distribution of Team Win Percentages (2000-2024)')
plt.xlabel('Win Percentage')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("EDA Graphs/win_percentage_distribution.png", dpi=300)
plt.close()


#Features that result in winning games
correlations = df.corr(numeric_only=True)['W - L %'].sort_values(ascending=False)[1:16]

plt.figure(figsize=(10, 8))
sns.barplot(x=correlations.values, y=correlations.index, palette='coolwarm')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.title('Top 15 Features Correlated with Win Percentage')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig("EDA Graphs/top_15_correlations.png", dpi=300)
plt.close()

#Teams that are elite vs poor
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    df['Points Scored/Game'],
    df['Points Allowed/Game'],
    c=df['W - L %'],
    s=100,
    cmap='OrRd',
    alpha=0.75,
    edgecolors='black'
)
plt.colorbar(scatter, label='Win Percentage')
plt.xlabel('Points Scored Per Game (Offense)')
plt.ylabel('Points Allowed Per Game (Defense)')
plt.title('Offensive vs Defensive Performance')
plt.axhline(y=df['Points Allowed/Game'].median(), color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=df['Points Scored/Game'].median(), color='gray', linestyle='--', alpha=0.5)
plt.text(30, 15, 'Elite Teams', fontsize=12, weight='bold')
plt.text(15, 30, 'Poor Teams', fontsize=12, weight='bold')
plt.savefig("EDA Graphs/offense_vs_defense_scatter.png", dpi=300)
plt.close()

