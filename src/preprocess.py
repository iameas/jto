import pandas as pd

# Load dataset
df = pd.read_csv("data/student_performance.csv") 

# Show first rows
print(df.head())

# Shows dataset info. This enables us inspect columns, identify target variable, clean the dataset properly before training
print(df.info())

# Check missing values
print(df.isnull().sum())