import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\aryan\Desktop\OIB-SIP\Task-3_CarPricePrediction\car data.csv")

print("First 5 rows:\n", df.head())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Convert categorical data to numeric using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\nR² Score:", r2_score(y_test, y_pred))

# Plot actual vs predicted
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")  # Optional
plt.show()
