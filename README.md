# Day 7 - Deploy ML Model as an API

HisabDo AI/ML Internship
Intern: Umair Ahmed Siddiqui

## Objective
Deploy the trained Student Performance model as a working API using FastAPI.

## Install Dependencies
pip install -r requirements.txt

## Run the API
uvicorn "Source Code.main:app" --reload

Runs at: http://127.0.0.1:8000

## Endpoint
POST /predict

## Request Format
```json
{
  "Attendance": 85,
  "Assignment_Score": 78,
  "Midterm_Score": 70,
  "Final_Score": 75
}
```

## Response Format
```json
{
  "prediction": "Pass",
  "confidence": 0.9994
}
```

Invalid input (e.g. score above 100) returns a 422 error instead of crashing.

Day 7 Focus: ML Model, FastAPI, Prediction API, Postman