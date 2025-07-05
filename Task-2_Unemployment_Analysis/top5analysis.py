import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("C:/Users/aryan/Desktop/OIB-SIP/Task-2_Unemployment_Analysis/Unemployment_Rate_upto_11_2020.csv")

df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Calculate average unemployment rate per region
top5_regions = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False).head(5)

# Filter original data for only top 5 regions
top5_names = top5_regions.index.tolist()
df_top5 = df[df['Region'].isin(top5_names)]

# Plot unemployment trends for top 5 regions
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_top5, x='Date', y='Estimated Unemployment Rate (%)', hue='Region', linewidth=2.2)
plt.title('Top 5 Regions by Average Unemployment Rate (2020)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Save the figure (optional)
plt.savefig("top5_unemployment_trend.png")

# Show the plot
plt.show()
