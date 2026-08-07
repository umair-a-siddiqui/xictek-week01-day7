import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

os.makedirs('Model', exist_ok=True)

# Load dataset
df = pd.read_csv('Database/student_pass_fail_data.csv')

features = ["Attendance", "Assignment_Score", "Midterm_Score", "Final_Score"]
X = df[features]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model (using Day 6's best settings: C=100, solver='lbfgs')
model = LogisticRegression(C=100, solver='lbfgs', max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save model and scaler to files
joblib.dump(model, 'Model/student_model.pkl')
joblib.dump(scaler, 'Model/scaler.pkl')

print("Model and scaler saved successfully in the 'Model' folder.")