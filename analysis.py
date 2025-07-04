import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Set plot style
sns.set(style="whitegrid")

# Create a pivot table for cleaner plotting
pivot_df = df.pivot_table(index='Date', columns='Region', values='Estimated Unemployment Rate (%)')

# Plot
plt.figure(figsize=(14, 7))
pivot_df.plot(figsize=(14, 7), linewidth=2.0)
plt.title('📈 Unemployment Rate Over Time by Region (India)', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend(loc='upper right', title='Region', bbox_to_anchor=(1.15, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

plt.savefig("unemployment_rate_plot.png")
