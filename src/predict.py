import joblib
import numpy as np
import pandas as pd

# Load trained model and encoder
model = joblib.load("models/random_forest_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# Example student data:
# (study_hours, attendance_percentage, class_participation, total_score)

student_data = pd.DataFrame([{
    "weekly_self_study_hours": 15,
    "attendance_percentage": 85,
    "class_participation": 6,
    "total_score": 78
}])

# Predict
prediction = model.predict(student_data)

# Decode label
predicted_grade = encoder.inverse_transform(prediction)

print(f"\nPredicted Grade: {predicted_grade[0]}")