import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

# Load trained model and label encoder
model = joblib.load("models/random_forest_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction System",
    layout="centered"
)

# App title
st.title("Student Academic Performance Prediction System")

st.markdown("""
This system predicts a student's academic grade using Machine Learning.
It also explains why the prediction was made using Explainable AI (SHAP).
""")

# Sidebar inputs
st.sidebar.header("Enter Student Information")

study_hours = st.sidebar.slider(
    "Weekly Self Study Hours",
    min_value=0.0,
    max_value=40.0,
    value=10.0
)

attendance = st.sidebar.slider(
    "Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

participation = st.sidebar.slider(
    "Class Participation",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

total_score = st.sidebar.slider(
    "Total Score",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

# Prediction button
if st.button("Predict Grade"):

    # Create dataframe for prediction
    student_data = pd.DataFrame([{
        "weekly_self_study_hours": study_hours,
        "attendance_percentage": attendance,
        "class_participation": participation,
        "total_score": total_score
    }])

    # Make prediction
    prediction = model.predict(student_data)

    # Prediction probabilities
    probabilities = model.predict_proba(student_data)

    # Decode predicted label
    predicted_grade = encoder.inverse_transform(prediction)

    # Confidence score
    confidence = probabilities.max() * 100

    # Display prediction result
    st.success(f"Predicted Grade: {predicted_grade[0]}")

    st.info(f"Prediction Confidence: {confidence:.2f}%")

    # Display probability chart
    st.subheader("Grade Prediction Probabilities")

    grades = encoder.classes_

    fig, ax = plt.subplots()

    ax.bar(grades, probabilities[0])

    ax.set_xlabel("Grades")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probability Distribution")

    st.pyplot(fig)

    # SHAP Explainability
    st.subheader("SHAP Prediction Explanation")

    shap_values = explainer.shap_values(student_data)

    shap_fig, shap_ax = plt.subplots()

    shap.summary_plot(
        shap_values,
        student_data,
        plot_type="bar",
        show=False
    )

    st.pyplot(shap_fig)

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