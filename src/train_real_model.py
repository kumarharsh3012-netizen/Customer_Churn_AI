import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from prediction import (
    load_churn_pipeline,
    predict_customer_churn
)
from recommendations import add_recommendations
from data_preprocessing import build_preprocessing_pipeline
# Load dataset

df = pd.read_csv("data/telco_churn.csv")

print("Original Dataset Shape:", df.shape)


# Clean TotalCharges column

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)


# Remove customer ID

df = df.drop(columns=["customerID"])


# Separate features and target

X = df.drop(columns=["Churn"])
y = df["Churn"]


print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAINING DATA SHAPE:", X_train.shape)
print("TESTING DATA SHAPE:", X_test.shape)

print("\nTraining Churn Distribution:")
print(y_train.value_counts())

print("\nTesting Churn Distribution:")
print(y_test.value_counts())
# Build complete machine learning pipeline

preprocessor = build_preprocessing_pipeline(X_train)

churn_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

print("\nComplete ML pipeline created successfully!")
# Train the complete pipeline

churn_pipeline.fit(X_train, y_train)

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")
# Evaluate the complete pipeline

predictions = churn_pipeline.predict(X_test)

probabilities = churn_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, pos_label="Yes"
)

recall = recall_score(
    y_test, predictions, pos_label="Yes"
)

f1 = f1_score(
    y_test, predictions, pos_label="Yes"
)

roc_auc = roc_auc_score(
    y_test == "Yes",
    probabilities
)

print("\nCONFUSION MATRIX")
print(confusion_matrix(
    y_test, predictions, labels=["No", "Yes"]
))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predictions))

print("\nMODEL PERFORMANCE")
print("-----------------")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")
print(f"ROC-AUC:   {roc_auc:.2f}")
# Save complete churn prediction pipeline

import os

os.makedirs("models", exist_ok=True)

joblib.dump(
    churn_pipeline,
    "models/churn_pipeline.pkl"
)

print("\nCOMPLETE PIPELINE SAVED SUCCESSFULLY!")
# Load saved pipeline

loaded_pipeline = joblib.load(
    "models/churn_pipeline.pkl"
)

print("\nSAVED PIPELINE LOADED SUCCESSFULLY!")


# Test predictions on raw customer data

sample_data = X_test.head(5)

sample_predictions = loaded_pipeline.predict(sample_data)

sample_probabilities = loaded_pipeline.predict_proba(sample_data)[:, 1]

print("\nSAMPLE CUSTOMER PREDICTIONS")

for i in range(5):
    print(
        f"Customer {i + 1}: "
        f"Prediction = {sample_predictions[i]}, "
        f"Churn Probability = {sample_probabilities[i]:.2%}"
    )
# Test the prediction module

pipeline = load_churn_pipeline()

sample_customers = X_test.head(5)

prediction_results = predict_customer_churn(
    pipeline,
    sample_customers
)

print("\nPREDICTION MODULE TEST")
print(prediction_results[
    ["Churn Probability", "Churn Prediction"]
])
# Test the retention recommendation engine

final_results = add_recommendations(
    prediction_results
)

print("\nCUSTOMER CHURN AND RETENTION REPORT")

for index, customer in final_results.iterrows():

    print(f"\nCustomer Index: {index}")
    print(f"Churn Probability: {customer['Churn Probability']:.2%}")
    print(f"Churn Prediction: {customer['Churn Prediction']}")

    print("Retention Recommendations:")

    for recommendation in customer["Retention Recommendations"]:
        print("-", recommendation)