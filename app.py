import streamlit as st
import pandas as pd

from src.prediction import (
    load_churn_pipeline,
    predict_customer_churn
)

from src.recommendations import add_recommendations
# Configure the Streamlit page
st.set_page_config(
    page_title="AI Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Main heading
st.title("📊 AI-Powered Customer Churn Prediction")

st.write(
    """
    This application uses Machine Learning to identify existing
    customers who may be likely to leave a service.

    Upload a compatible customer dataset to:
    - Predict customer churn probability.
    - Identify high-risk customers.
    - Generate retention recommendations.
    """
)

st.divider()
# CSV file uploader
st.subheader("📁 Upload Customer Dataset")

uploaded_file = st.file_uploader(
    "Choose a customer CSV file",
    type=["csv"],
    help="Upload a CSV file containing customer information."
)
if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        st.write("Dataset Shape:", df.shape)
                # Required columns expected by the trained model
        required_columns = [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges"
        ]

        # Identify missing columns
        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            st.error("Your dataset is missing required columns:")
            st.write(missing_columns)

        else:
            st.success("All required columns are available!")
                        # Prepare customer data for prediction
            customer_data = df[required_columns].copy()

            # Convert TotalCharges to numeric values
            customer_data["TotalCharges"] = pd.to_numeric(
                customer_data["TotalCharges"],
                errors="coerce"
            )

            # Load the trained ML pipeline
            pipeline = load_churn_pipeline()
                        # Churn probability threshold
            threshold = st.slider(
                "Churn Risk Threshold",
                min_value=0.10,
                max_value=0.90,
                value=0.30,
                step=0.05,
                help="Customers at or above this probability will be classified as high risk."
            )

            # Generate churn predictions
            prediction_results = predict_customer_churn(
            pipeline,
            customer_data
            )
            # Preserve customer IDs if available
            if "customerID" in df.columns:
                prediction_results.insert(
                    0,
                    "customerID",
                    df["customerID"].values
                )

            st.success("Churn predictions generated successfully!")

            st.subheader("Prediction Results")
            st.dataframe(prediction_results.head())
                        # Generate retention recommendations
            final_results = add_recommendations(
                prediction_results
            )

            st.subheader("Retention Recommendations")

            st.dataframe(
                final_results[
                    [
                        "Churn Prediction",
                        "Churn Probability",
                        "Retention Recommendations"
                    ]
                ].head()
            )
            # Calculate dashboard metrics
            total_customers = len(final_results)

            high_risk_customers = (
                final_results["Churn Prediction"] == "Yes"
            ).sum()

            low_risk_customers = (
                final_results["Churn Prediction"] == "No"
            ).sum()

            churn_percentage = (
                high_risk_customers / total_customers
            ) * 100

            # Dashboard
            st.divider()
            st.subheader("📊 Customer Churn Dashboard")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Customers",
                total_customers
            )

            col2.metric(
                "High-Risk Customers",
                high_risk_customers
            )

            col3.metric(
                "Low-Risk Customers",
                low_risk_customers
            )

            col4.metric(
                "High-Risk Percentage",
                f"{churn_percentage:.2f}%"
            )
                        # Churn risk distribution chart
            st.subheader("Customer Risk Distribution")

            risk_counts = final_results[
                "Churn Prediction"
            ].value_counts()

            st.bar_chart(risk_counts)

                        # Download prediction results
            st.divider()
            st.subheader("📥 Download Results")

            csv_data = final_results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Predictions as CSV",
                data=csv_data,
                file_name="customer_churn_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")