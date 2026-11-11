import pandas as pd
from data_preprocessing import build_preprocessing_pipeline
from model_training import train_logistic_regression, evaluate_model,save_model,predict_churn
# Load the real customer churn dataset
df = pd.read_csv("data/telco_churn.csv")

# Display basic information
print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nCHURN DISTRIBUTION")
print(df["Churn"].value_counts())

# Investigate TotalCharges

print("\nTOTAL CHARGES SAMPLE")
print(df["TotalCharges"].head(20).tolist())

print("\nBLANK TOTAL CHARGES")
print((df["TotalCharges"].str.strip() == "").sum())

# Convert TotalCharges to numeric

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nTOTAL CHARGES DATA TYPE")
print(df["TotalCharges"].dtype)

print("\nMISSING VALUES AFTER CONVERSION")
print(df["TotalCharges"].isnull().sum())

# Display customers with missing TotalCharges

missing_total_charges = df[df["TotalCharges"].isnull()]

print("\nCUSTOMERS WITH MISSING TOTAL CHARGES")
print(
    missing_total_charges[
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ]
)

print("\nTENURE DISTRIBUTION OF THESE CUSTOMERS")
print(missing_total_charges["tenure"].value_counts())

# Fill missing TotalCharges with 0

df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("\nMISSING VALUES AFTER CLEANING")
print(df["TotalCharges"].isnull().sum())

print("\nUPDATED TOTAL CHARGES DATA TYPE")
print(df["TotalCharges"].dtype)

# Final missing-value check

print("\nFINAL DATASET MISSING VALUES")
print(df.isnull().sum())

print("\nTOTAL MISSING VALUES")
print(df.isnull().sum().sum())

# Identify numerical and categorical columns

numerical_columns = df.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNUMERICAL COLUMNS")
print(numerical_columns)

print("\nCATEGORICAL COLUMNS")
print(categorical_columns)

print("\nNUMBER OF NUMERICAL COLUMNS:", len(numerical_columns))
print("NUMBER OF CATEGORICAL COLUMNS:", len(categorical_columns))

# Create a clean dataset for machine learning

# Remove customerID
df = df.drop(columns=["customerID"])

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

print("\nFEATURE DATASET SHAPE")
print(X.shape)

print("\nTARGET DATASET SHAPE")
print(y.shape)

print("\nFEATURE COLUMNS")
print(X.columns.tolist())

print("\nTARGET DISTRIBUTION")
print(y.value_counts())

from sklearn.model_selection import train_test_split

# Split the real dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAINING FEATURES SHAPE:", X_train.shape)
print("TESTING FEATURES SHAPE:", X_test.shape)

print("\nTRAINING TARGET DISTRIBUTION")
print(y_train.value_counts())

print("\nTESTING TARGET DISTRIBUTION")
print(y_test.value_counts())

# Build and apply preprocessing pipeline

preprocessor = build_preprocessing_pipeline(X_train)

X_train_processed = preprocessor.fit_transform(X_train)
# Save the fitted preprocessing pipeline

import joblib

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

print("\nPreprocessing pipeline saved successfully!")
X_test_processed = preprocessor.transform(X_test)

print("\nPROCESSED TRAINING SHAPE:")
print(X_train_processed.shape)

print("\nPROCESSED TESTING SHAPE:")
print(X_test_processed.shape)
# Train Logistic Regression Model

model = train_logistic_regression(
    X_train_processed,
    y_train
)
# Save the trained model

save_model(
    model,
    "models/churn_model.pkl"
)

print("\nMODEL TRAINING COMPLETED!")

# Evaluate Model

results = evaluate_model(
    model,
    X_test_processed,
    y_test
)
# Verify saved files

import joblib

loaded_model = joblib.load("models/churn_model.pkl")
loaded_preprocessor = joblib.load("models/preprocessor.pkl")

print("\nMODEL LOADED SUCCESSFULLY!")
print("PREPROCESSOR LOADED SUCCESSFULLY!")
# Compare different classification thresholds

from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = [0.3, 0.4, 0.5, 0.6]

print("\nTHRESHOLD COMPARISON")
print("--------------------")

for threshold in thresholds:

    predictions, probabilities = predict_churn(
        model,
        X_test_processed,
        threshold=threshold
    )

    precision = precision_score(
        y_test, predictions, pos_label="Yes", zero_division=0
    )

    recall = recall_score(
        y_test, predictions, pos_label="Yes", zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, pos_label="Yes", zero_division=0
    )

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")