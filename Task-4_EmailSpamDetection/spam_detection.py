# Email Spam Detection with Machine Learning

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load Dataset
df = pd.read_csv("Task-4_EmailSpamDetection/spam.csv", encoding='latin-1')


# Step 2: Clean and Prepare Data
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df = df[['label', 'message']]  # Drop extra unnamed columns
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Encode labels

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Step 4: Vectorize Text Data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Predict and Evaluate
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("ConfusionMatrix.png")  # save the image
plt.show()
