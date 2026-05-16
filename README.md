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
- *Also check for train_3.py, it contains training codes using XGBoost, which gives you model comparison within Random Forest, and also optionally use the better model*
- *Created a prediction script to test our trained model. Check "src/predict.py" for script*
- *Models trained can be found on the models/ directory*
- *Check for model differentiation, the first models and encoder ``label_encoder.pkl`` and ``random_forest_model.pkl`` was used for app_1.py, app_2.py and app_3.py, hereby trained using train_1.py and train_2.py which doesn't include XGBoost*
- *XGBoost was used in train_3 to make model accuracy better*
- *The new trained model data and encoder is labeled ``best_model.pkl`` and ``best_label_encoder.pkl`` in the model/ directory*
- *These new trained models are used to create the new app.py*

## Guide

- Run inspect_dataset.py on the terminal "python src/inspect_dataset.py" to inspect dataset
- Run "python src/preprocess.py" for feature seperation
- To run the dashboard, please run ``streamlit run app.py`` on the terminal
- app_1.py contains the first dashboard created with plain UI
- app_2.py contains the second dashboard created with more advanced UI including Prediction Confidence, Probability Distribution Chart, Feature Importance Chart
- app_3.py contains everything in app_2.py but uses SHAP Explainer ie. Explainable AI (XAI)
- app.py contains everything in app_3.py, but with XGBoost and a Risk Level System included
- SHAP is necessary because, instead of it just predicting model e.g. "Predicted Grade=B", it'll explain why the system will explain
- Example: High attendance increased prediction
- Low participationo reduced score