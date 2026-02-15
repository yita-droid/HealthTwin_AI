import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

# 1. Generate Synthetic Data with History Influence
np.random.seed(42)
n_samples = 1500 

data = {
    'Age': np.random.randint(10, 90, n_samples),
    'Systolic_BP': np.random.randint(90, 200, n_samples),
    'Diastolic_BP': np.random.randint(60, 120, n_samples),
    'Heart_Rate': np.random.randint(50, 130, n_samples),
    'Temperature': np.random.uniform(97.0, 104.0, n_samples),
    'Oxygen_Sat': np.random.randint(85, 100, n_samples),
    'Chest_Pain': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    'Shortness_of_Breath': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'Dizziness': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    'Vomiting': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    'History_Score': np.random.randint(0, 5, n_samples) 
}

df = pd.DataFrame(data)

# 2. Define Logic: History Score now directly shifts risk
def assign_risk(row):
    score = 0
    if row['Systolic_BP'] > 160 or row['Oxygen_Sat'] < 92: score += 3
    if row['Chest_Pain'] == 1: score += 2
    if row['History_Score'] >= 2: score += 2 
    
    if score >= 5: return 2 # High
    if score >= 3: return 1 # Medium
    return 0 # Low

def assign_dept(row):
    if row['Chest_Pain'] == 1 or row['History_Score'] >= 3: return 1 # Cardiology
    elif row['Dizziness'] == 1: return 2 # Neurology
    elif row['Oxygen_Sat'] < 90: return 3 # Emergency
    return 0 # General Medicine

df['Risk_Label'] = df.apply(assign_risk, axis=1)
df['Dept_Label'] = df.apply(assign_dept, axis=1)

X = df.drop(['Risk_Label', 'Dept_Label'], axis=1)

# 3. Train and Save
print("Training models...")
risk_model = xgb.XGBClassifier().fit(X, df['Risk_Label'])
dept_model = RandomForestClassifier().fit(X, df['Dept_Label'])

joblib.dump(risk_model, 'risk_model.pkl')
joblib.dump(dept_model, 'dept_model.pkl')
print("✅ SUCCESS: Models saved with 11 features (including History_Score).")