import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("C:/Users/aryan/Desktop/OIB-SIP/Task-1_Iris_Flower_Classification/Iris.csv")


# Drop 'Id' column if it exists
if 'Id' in df.columns:
    df.drop(columns=['Id'], inplace=True)

# Basic info
print("First 5 rows:\n", df.head())
print("\nClass distribution:\n", df['Species'].value_counts())

# Pairplot
sns.pairplot(df, hue="Species")
plt.suptitle("Iris Data Pairplot", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.drop(columns='Species').corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Splitting dataset
X = df.drop('Species', axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
