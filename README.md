# Student Academic Performance Model Using Machine Learning

*Summary: This project is to predict final exam score or CGPA*

## Note: Library used in this development


| Library| Purpose|
|---------|--------|
| pandas | Dataset handling |
| numpy | Numerical operations|
|scikit-learn| ML models|
|matplotlib | Charts|
|xgboost | Advanced model|
streamlit | Dashboard|
| shap | Explainable AI|
|joblib | Save model|

## Note

- *src/train_1.py contains the first training codes for the dataset*
- *For a more advanced/stronger model, run src/train_2.py which uses random forest*
- *Please read through train_1.py to understand train_2.py*
- *Created a prediction script to test our trained model. Check "src/predict.py" for script*

## Guide

- Run inspect_dataset.py on the terminal "python src/inspect_dataset.py" to inspect dataset
- Run "python src/preprocess.py" for feature seperation
- To run the dashboard, please run ``streamlit run app.py`` on the terminal
- app_1.py contains the first dashboard created with plain UI
- app_2.py contains the second dashboard created with more advanced UI including Prediction Confidence, Probability Distribution Chart, Feature Importance Chart
- app.py (ie. the main dashboard app) contains everything in app_2.py but uses SHAP Explainer ie. Explainable AI (XAI)
- SHAP is necessary because, instead of it just predicting model e.g. "Predicted Grade=B", it'll explain why the system will explain
- Example: High attendance increased prediction
- Low participationo reduced score