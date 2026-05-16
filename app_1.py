import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("models/random_forest_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# App title
st.title("Student Academic Performance Prediction System")

st.write(
    "Enter student information below to predict academic grade."
)

# User inputs
study_hours = st.slider(
    "Weekly Self Study Hours",
    0.0,
    40.0,
    10.0
)

attendance = st.slider(
    "Attendance Percentage",
    0.0,
    100.0,
    75.0
)

participation = st.slider(
    "Class Participation",
    0.0,
    10.0,
    5.0
)

total_score = st.slider(
    "Total Score",
    0.0,
    100.0,
    50.0
)

# Predict button
if st.button("Predict Grade"):

    # Create dataframe
    student_data = pd.DataFrame([{
        "weekly_self_study_hours": study_hours,
        "attendance_percentage": attendance,
        "class_participation": participation,
        "total_score": total_score
    }])

    # Predict
    prediction = model.predict(student_data)

    # Decode prediction
    predicted_grade = encoder.inverse_transform(prediction)

    # Display result
    st.success(
        f"Predicted Grade: {predicted_grade[0]}"
    )

# Note:
# Contains a rough sketch dashboard with datas that can be used for prediction