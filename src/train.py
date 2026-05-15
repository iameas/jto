import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/student_performance.csv")

# Drop unnecessary column
df = df.drop("student_id", axis=1)

# Features and target
X = df.drop("grade", axis=1)
y = df["grade"]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# Initialize model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
# This measures how many preductions were correct

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

# Classification report
# This shows precision, recall and F1-score

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
# This shows where the model predicts corectly, and where it confuses grades

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# After the training, the expected result is: high accuracy!
# This is because: total_score strongly influences grade