import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/student_performance.csv")

# Drop unnecessary column
df = df.drop("student_id", axis=1)

# Features (X)
X = df.drop("grade", axis=1)

# Target (y)
y = df["grade"]

# Encode target labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# Print results
print("Features:")
print(X.head())

print("\nEncoded Labels:")
print(y_encoded[:10])

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

print("\nGrade Mapping:")
for index, label in enumerate(encoder.classes_):
    print(f"{label} = {index}")


# Explanations: This is called a feature seperation. If you run the codes, you'll notice:
# X contains: study hours, attendance, participation, and total score.
# But our target is Y. Y contains our grade.