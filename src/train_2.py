import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Added Random Forest
from sklearn.ensemble import RandomForestClassifier 

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
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

# Classification report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# This section is called feature importance

# It is important because:

# It adds interpretability,
# Supports the analysis section,
# And, improve the presentation

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

# Sort values
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

# === Random Forest === #
# This improves accuracy, handles nonlinearity better, and reduces overfitting

# In this train_2.py, Random Forest creates:

# Multiple decision treesc
# Combines their predictions,
# Improve robustness

# Outcome:

# After test, the model accuracy is about 99.81%
# We have an even highier accuracy, 
# Better handling of D/F grades
# Improved F1-score