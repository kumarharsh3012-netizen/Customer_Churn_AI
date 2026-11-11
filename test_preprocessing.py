
import pandas as pd
from src.model_training import train_logistic_regression,predict_churn,evaluate_model,save_model,load_model
from src.data_preprocessing import (
    load_dataset,
    inspect_dataset,
    remove_duplicates,
    handle_missing_values,
    encode_categorical_columns,
    separate_features_target,split_dataset,
    scale_features,build_preprocessing_pipeline
)
# Step 1: Create sample customer data
data = {
    "CustomerID": [101, 102, 103, 104, 105],
    "Age": [25, 32, 45, 28, 50],
    "MonthlyCharges": [50, 70, 90, 60, 100],
    "Churn": ["Yes", "No", "No", "Yes", "No"]
}

df = pd.DataFrame(data)

# Step 2: Save data as a CSV file
df.to_csv("data/sample_customers.csv", index=False)

# Step 3: Load CSV using our function
loaded_df = load_dataset("data/sample_customers.csv")

# Step 4: Display the results
print("Dataset loaded successfully!")

print("\nFirst five records:")
print(loaded_df.head())

print("\nDataset shape:", loaded_df.shape)

print("\nColumn names:", loaded_df.columns.tolist())
# Step 5: Inspect dataset quality
report = inspect_dataset(loaded_df)

print("\nDATA QUALITY REPORT")
print("-" * 30)

print("Total rows:", report["total_rows"])
print("Total columns:", report["total_columns"])

print("\nMissing values in each column:")
print(report["missing_values"])

print("Total missing values:", report["total_missing_values"])
print("Duplicate rows:", report["duplicate_rows"])

# Step 6: Test duplicate removal

# Create duplicate records deliberately
duplicate_df = pd.concat([loaded_df, loaded_df.iloc[[0]]], ignore_index=True)

print("\nBEFORE DUPLICATE REMOVAL")
print("Total rows:", len(duplicate_df))

# Remove duplicate records
cleaned_df, removed_rows = remove_duplicates(duplicate_df)

print("\nAFTER DUPLICATE REMOVAL")
print("Total rows:", len(cleaned_df))
print("Removed duplicate rows:", removed_rows)

# Step 7: Test missing-value handling

# Create a copy of our dataset
missing_df = loaded_df.copy()

# Introduce missing values deliberately
missing_df.loc[0, "Age"] = float("nan")
missing_df.loc[1, "MonthlyCharges"] = float("nan")
missing_df.loc[2, "Churn"] = None

print("\nBEFORE HANDLING MISSING VALUES")
print(missing_df.isnull().sum())

# Handle missing values
cleaned_missing_df = handle_missing_values(missing_df)

print("\nAFTER HANDLING MISSING VALUES")
print(cleaned_missing_df.isnull().sum())

print("\nCleaned dataset:")
print(cleaned_missing_df)

# Step 8: Test categorical encoding

print("\nBEFORE ENCODING")
print(loaded_df.head())

# Encode categorical columns
encoded_df = encode_categorical_columns(loaded_df)

print("\nAFTER ENCODING")
print(encoded_df.head())

print("\nEncoded column names:")
print(encoded_df.columns.tolist())

print("\nData types:")
print(encoded_df.dtypes)

# Step 9: Test feature-target separation

X, y = separate_features_target(loaded_df, "Churn")

print("\nFEATURES (X)")
print(X.head())

print("\nTARGET (y)")
print(y.head())

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Step 10: Test train-test splitting

# Use a larger dataset because stratification needs enough samples
test_df = pd.concat([loaded_df] * 20, ignore_index=True)

X_test_data, y_test_data = separate_features_target(
    test_df, "Churn"
)

X_train, X_test, y_train, y_test = split_dataset(
    X_test_data, y_test_data
)

print("\nTRAIN-TEST SPLIT RESULTS")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

print("\nTotal records:", len(X_train) + len(X_test))

# Step 11: Test feature scaling

X_train_scaled, X_test_scaled, scaler = scale_features(
    X_train, X_test
)

print("\nFEATURE SCALING RESULTS")

print("Original training shape:", X_train.shape)
print("Scaled training shape:", X_train_scaled.shape)

print("\nOriginal testing shape:", X_test.shape)
print("Scaled testing shape:", X_test_scaled.shape)

print("\nMean of scaled training features:")
print(X_train_scaled.mean(axis=0))

print("\nStandard deviation of scaled training features:")
print(X_train_scaled.std(axis=0))

# Step 12: Test automated preprocessing pipeline

# Prepare a dataset with numerical and categorical features
pipeline_df = test_df.drop(
    columns=["CustomerID"]
).copy()

# Add a categorical feature
pipeline_df["Contract"] = (
    ["Monthly", "Yearly", "Two-Year", "Monthly", "Yearly"] * 20
)

# Introduce missing values
pipeline_df.loc[0, "Age"] = float("nan")
pipeline_df.loc[1, "Contract"] = None

# Separate features and target
X_pipeline, y_pipeline = separate_features_target(
    pipeline_df, "Churn"
)

# Split the dataset
X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_dataset(
    X_pipeline, y_pipeline
)

# Build preprocessing pipeline
preprocessor = build_preprocessing_pipeline(X_train_raw)

# Fit on training data and transform
X_train_processed = preprocessor.fit_transform(X_train_raw)

# Transform testing data
X_test_processed = preprocessor.transform(X_test_raw)

print("\nAUTOMATED PIPELINE RESULTS")

print("Original training shape:", X_train_raw.shape)
print("Processed training shape:", X_train_processed.shape)

print("Original testing shape:", X_test_raw.shape)
print("Processed testing shape:", X_test_processed.shape)

print("\nPipeline executed successfully!")
# Test Logistic Regression model

X_train = [[1, 2], [2, 3], [3, 4], [4, 5],
           [5, 6], [6, 7], [7, 8], [8, 9]]

y_train = ["Yes", "No", "No", "Yes",
           "No", "Yes", "No", "Yes"]

model = train_logistic_regression(X_train, y_train)

print("\nModel training test passed!")
print("Model classes:", model.classes_)
# Test churn predictions

predictions, probabilities = predict_churn(model, X_train)

print("\nPredictions:", predictions)
print("Churn probabilities:", probabilities)
# Test model evaluation

evaluation_results = evaluate_model(
    model,
    X_train,
    y_train
)

print("\nEvaluation dictionary:", evaluation_results)
# Test model saving

save_model(model, "models/churn_model.pkl")
# Test model loading

loaded_model = load_model("models/churn_model.pkl")

print("Loaded model classes:", loaded_model.classes_)




