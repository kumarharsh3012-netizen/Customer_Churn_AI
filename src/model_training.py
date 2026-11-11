
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

import joblib
import os
import numpy as np

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model
def predict_churn(model, X_test, threshold=0.5):
    yes_index = list(model.classes_).index("Yes")

    probabilities = model.predict_proba(X_test)[:, yes_index]

    predictions = np.where(
        probabilities >= threshold,
        "Yes",
        "No"
    )
    return predictions, probabilities
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, pos_label="Yes", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, pos_label="Yes", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, pos_label="Yes", zero_division=0
    )
    yes_index = list(model.classes_).index("Yes")
    probabilities = model.predict_proba(X_test)[:, yes_index]

    roc_auc = roc_auc_score(
        np.array(y_test) == "Yes",
        probabilities
    )
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["No", "Yes"]
    )

    print("\nCONFUSION MATRIX")
    print(cm)

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["No", "Yes"],
            zero_division=0
        )
    )

    print("\nMODEL EVALUATION RESULTS")
    print("-----------------------")
    print(f"Accuracy:  {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")
    print(f"ROC-AUC:   {roc_auc:.2f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    joblib.dump(model, file_path)

    print(f"Model saved successfully at: {file_path}")
def load_model(file_path):
    model = joblib.load(file_path)

    print(f"Model loaded successfully from: {file_path}")

    return model
# Default churn classification threshold

DEFAULT_THRESHOLD = 0.30