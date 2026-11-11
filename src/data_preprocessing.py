
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_dataset(file):
    """
    Load a customer dataset from a CSV file.
    """

    try:
        df = pd.read_csv(file)

        if df.empty:
            raise ValueError("The uploaded dataset is empty.")

        return df

    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

def inspect_dataset(df):
    """
    Inspect a dataset for missing values and duplicate records.
    """

    missing_values = df.isnull().sum()
    duplicate_count = df.duplicated().sum()

    report = {
        "total_rows": df.shape[0],
        "total_columns": df.shape[1],
        "missing_values": missing_values,
        "total_missing_values": missing_values.sum(),
        "duplicate_rows": duplicate_count
    }

    return report

def remove_duplicates(df):
    """
    Remove duplicate records from the dataset.
    """

    original_rows = len(df)

    df = df.drop_duplicates()

    removed_rows = original_rows - len(df)

    return df, removed_rows

def handle_missing_values(df):
    """
    Handle missing values in numerical and categorical columns.
    """

    df = df.copy()

    numerical_columns = df.select_dtypes(include=["number"]).columns
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns

    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].median())

    for column in categorical_columns:
        if df[column].isnull().any():
            df[column] = df[column].fillna(df[column].mode()[0])

    return df

def encode_categorical_columns(df):
    """
    Convert categorical columns into numerical format.
    """

    df = df.copy()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        dtype=int
    )

    return df

def separate_features_target(df, target_column):
    """
    Separate input features (X) and target (y).
    """

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y

from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

from sklearn.preprocessing import StandardScaler


def scale_features(X_train, X_test):
    """
    Scale numerical features using StandardScaler.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

def build_preprocessing_pipeline(X):
    """
    Build an automated preprocessing pipeline
    for numerical and categorical features.
    """

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    # Numerical preprocessing
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns)
        ],
        remainder="drop"
    )

    return preprocessor


