import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and encoder
model = joblib.load("models/random_forest_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# Page config
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="centered"
)

# Title
st.title("🎓 Student Academic Performance Prediction System")

st.markdown(
    """
    This system predicts a student's academic grade
    using Machine Learning.
    """
)

# Sidebar
st.sidebar.header("Student Inputs")

study_hours = st.sidebar.slider(
    "Weekly Self Study Hours",
    0.0,
    40.0,
    10.0
)

attendance = st.sidebar.slider(
    "Attendance Percentage",
    0.0,
    100.0,
    75.0
)

participation = st.sidebar.slider(
    "Class Participation",
    0.0,
    10.0,
    5.0
)

total_score = st.sidebar.slider(
    "Total Score",
    0.0,
    100.0,
    50.0
)

# Prediction button
if st.button("Predict Grade"):

    # Create dataframe
    student_data = pd.DataFrame([{
        "weekly_self_study_hours": study_hours,
        "attendance_percentage": attendance,
        "class_participation": participation,
        "total_score": total_score
    }])

    # Predict class
    prediction = model.predict(student_data)

    # Predict probabilities
    probabilities = model.predict_proba(student_data)

    # Decode label
    predicted_grade = encoder.inverse_transform(prediction)

    # Confidence score
    confidence = probabilities.max() * 100

    # Display prediction
    st.success(
        f"Predicted Grade: {predicted_grade[0]}"
    )

    st.info(
        f"Prediction Confidence: {confidence:.2f}%"
    )

    # Probability chart
    st.subheader("Grade Probabilities")

    grades = encoder.classes_

    fig, ax = plt.subplots()

    ax.bar(grades, probabilities[0])

    ax.set_xlabel("Grades")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probability Distribution")

    st.pyplot(fig)

# Feature Importance Section
st.subheader("Feature Importance")

importance = model.feature_importances_

features = [
    "Study Hours",
    "Attendance",
    "Participation",
    "Total Score"
]

fig2, ax2 = plt.subplots()

ax2.bar(features, importance)

ax2.set_ylabel("Importance")
ax2.set_title("Model Feature Importance")

st.pyplot(fig2)

# Note:
# I addded confidence score e.g. "Prediction Confidence is 98.72%"
# Addded Probability Distribution Chart that shows probability of A, probability of B, etc.
# Lastly, i added Feature Importance Chart that shows which feature most influences predictions.
# Finally, a better UI structure with inputs on sidebar