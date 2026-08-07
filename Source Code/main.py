from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="Student Performance Prediction API")

# Load the saved model and scaler once, when the API starts
model = joblib.load("Model/student_model.pkl")
scaler = joblib.load("Model/scaler.pkl")

# This defines what data the API expects, and automatically validates it
class StudentData(BaseModel):
    Attendance: float = Field(..., ge=0, le=100)
    Assignment_Score: float = Field(..., ge=0, le=100)
    Midterm_Score: float = Field(..., ge=0, le=100)
    Final_Score: float = Field(..., ge=0, le=100)

@app.get("/")
def home():
    return {"message": "Student Performance Prediction API is running."}

@app.post("/predict")
def predict(data: StudentData):
    input_data = np.array([[
        data.Attendance,
        data.Assignment_Score,
        data.Midterm_Score,
        data.Final_Score
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    result = "Pass" if prediction == 1 else "Fail"

    return {
        "prediction": result,
        "confidence": round(float(probability), 4)
    }