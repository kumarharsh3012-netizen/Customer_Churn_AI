import joblib
import pandas as pd
import numpy as np
# Load the saved churn prediction pipeline

def load_churn_pipeline(model_path="models/churn_pipeline.pkl"):

    pipeline = joblib.load(model_path)

    return pipeline
# Predict customer churn

def predict_customer_churn(
    pipeline,
    customer_data,
    threshold=0.30
):
    probabilities = pipeline.predict_proba(customer_data)[:, 1]

    predictions = np.where(
        probabilities >= threshold,
        "Yes",
        "No"
    )

    results = customer_data.copy()

    results["Churn Probability"] = probabilities
    results["Churn Prediction"] = predictions

    return results