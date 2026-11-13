AI-Powered Customer Churn Prediction and Retention System

An end-to-end Machine Learning application that predicts customer churn, identifies customers who are at risk of leaving a service, estimates their churn probability, and generates personalized retention recommendations.

The application uses Python, Scikit-learn, Pandas, and Streamlit to provide an interactive interface for customer churn analysis and prediction.

---

1. Project Overview

Customer churn refers to the situation in which customers stop using a company's products or services.

Customer retention is important for businesses because acquiring new customers can require more resources than retaining existing ones. Identifying customers who may leave allows businesses to take preventive action.

This project uses supervised Machine Learning to analyze customer information and predict whether a customer is likely to leave a service.

The system provides:

- Customer churn prediction.
- Churn probability estimation.
- Identification of high-risk and low-risk customers.
- Customer risk distribution.
- Personalized retention recommendations.
- Interactive dashboard metrics.
- Downloadable prediction reports.

The application is built using Streamlit, allowing users to interact with the Machine Learning model through a web-based interface without writing Python code.

2. Project Objectives

The main objectives are:

1. Develop a Machine Learning-based customer churn prediction system.
2. Identify customers who have a higher probability of leaving a service.
3. Estimate churn probability for individual customers.
4. Classify customers into high-risk and low-risk categories.
5. Generate retention recommendations based on customer characteristics.
6. Display prediction results through an interactive dashboard.
7. Allow users to upload customer datasets in CSV format.
8. Provide downloadable prediction results for further analysis.

3. Problem Statement

Businesses often face difficulties identifying customers who are likely to discontinue their services.

Traditional approaches may identify churn only after a customer has already left.

This project addresses the problem by using historical customer information and a trained Machine Learning model to estimate the likelihood of future churn.

The predictions can help businesses identify customers who need attention and take suitable retention measures.

4. Technology Stack

Technology| Purpose
Python| Core programming language
Pandas| Data manipulation and analysis
NumPy| Numerical operations
Scikit-learn| Machine Learning and preprocessing
Streamlit| Interactive web application
Joblib| Saving and loading trained ML objects
Matplotlib| Data visualization
Seaborn| Statistical visualization
Plotly| Interactive visualization support
XGBoost| Additional Machine Learning algorithm support
SHAP| Model explainability support
Imbalanced-learn| Handling imbalanced datasets
SciPy| Scientific computing
OpenPyXL| Excel file support in Python
Jupyter Notebook| Development and experimentation

5. Project Architecture

The project is organized into separate directories for application code, datasets, trained models, and supporting modules.

Customer_Churn_AI/
│
├── app.py
│
├── data/
│
├── models/
│   ├── churn_model.pkl
│   ├── churn_pipeline.pkl
│   └── preprocessing.pkl
│
├── src/
│   ├── prediction.py
│   └── recommendations.py
│
├── requirements.txt
│
├── run_app.bat
│
├── test_install.py
│
├── test_preprocessing.py
│
├── .gitignore
│
└── .gitattributes

Directory and File Description

app.py

The main application file.

Responsibilities:

- Configure the Streamlit page.
- Display the application title and introduction.
- Accept customer datasets through a CSV uploader.
- Read uploaded data using Pandas.
- Validate required customer columns.
- Prepare customer information for prediction.
- Load the trained Machine Learning pipeline.
- Generate churn predictions.
- Display prediction results.
- Generate retention recommendations.
- Calculate customer risk metrics.
- Display a risk distribution chart.
- Allow users to download prediction results.

data/

Stores project datasets used for development, testing, and analysis.

models/

Stores previously trained Machine Learning objects.

- "churn_model.pkl": Saved churn model artifact. Its exact internal object and role should be verified before using it independently.
- "churn_pipeline.pkl": Saved prediction pipeline loaded by the current application.
- "preprocessing.pkl": Saved preprocessing artifact. Its exact contents and relationship to the pipeline should be verified.

These files are serialized using Joblib and are intended to avoid unnecessary model retraining during normal application use.

src/

Contains reusable Python modules.

"prediction.py" handles model loading and churn prediction.

"recommendations.py" generates customer retention recommendations based on customer information and predicted churn status.

requirements.txt

Contains the Python packages required to run the project.

run_app.bat

A Windows batch file intended to simplify application startup.

test_install.py

A supporting installation or dependency testing script.

test_preprocessing.py

A supporting preprocessing testing script.

6. Machine Learning Approach

The current project implements a supervised Machine Learning classification workflow.

Supervised Learning

Supervised Learning is a Machine Learning approach in which a model learns patterns from labeled training data.

In this project, the model learns from historical customer information and churn labels.

The target variable is:

"Churn"

Possible target values:

- Yes: Customer has churned.
- No: Customer has not churned.

The model learns relationships between customer characteristics and churn behavior.

After training, it can predict churn for new customer records.

Classification

This is a binary classification problem because the model predicts one of two possible outcomes:

- Churn: Yes
- Churn: No

The prediction module also calculates the probability of churn.

7. Dataset Requirements

The current application expects a customer dataset containing the following 19 feature columns.

Column| Description
gender| Customer gender
SeniorCitizen| Indicates whether the customer is a senior citizen
Partner| Whether the customer has a partner
Dependents| Whether the customer has dependents
tenure| Number of months the customer has stayed
PhoneService| Whether phone service is active
MultipleLines| Whether multiple phone lines are available
InternetService| Type of internet service
OnlineSecurity| Whether online security is subscribed to
OnlineBackup| Whether online backup is subscribed to
DeviceProtection| Whether device protection is subscribed to
TechSupport| Whether technical support is subscribed to
StreamingTV| Whether streaming TV is subscribed to
StreamingMovies| Whether streaming movies are subscribed to
Contract| Type of customer contract
PaperlessBilling| Whether paperless billing is enabled
PaymentMethod| Customer payment method
MonthlyCharges| Monthly customer charges
TotalCharges| Total customer charges

The application also preserves the optional "customerID" column if it is available.

Dataset Format

The current application accepts CSV files.

Example:

customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges

The example shows the expected column structure, not a complete customer record.

Important Dataset Conditions

- Column names must match the expected names.
- All 19 required feature columns must be present.
- "TotalCharges" is converted to numeric values by the application.
- Missing or invalid values may require preprocessing compatible with the saved pipeline.
- The feature meanings and category values should be consistent with the training dataset.
- The "Churn" target column is not required when generating predictions for new customers.

A dataset with unrelated columns or a different feature structure cannot be used directly with the current churn model.

8. Application Workflow

The current application follows this workflow:

User Opens Application
          |
          v
Upload Customer CSV
          |
          v
Read Dataset Using Pandas
          |
          v
Display Dataset Preview
          |
          v
Validate Required Columns
          |
          v
Prepare Customer Features
          |
          v
Load Saved Churn Pipeline
          |
          v
Generate Churn Probabilities
          |
          v
Classify Customers
          |
          v
Generate Retention Recommendations
          |
          v
Calculate Dashboard Metrics
          |
          v
Display Results and Risk Chart
          |
          v
Download Prediction Report

Step 1: Upload Dataset

The user uploads a CSV file through the Streamlit file uploader.

Step 2: Dataset Preview

The application displays:

- A preview of the first five records.
- Number of rows.
- Number of columns.

Step 3: Column Validation

The application checks whether all required customer feature columns are present.

If any required columns are missing, the application displays an error and lists the missing columns.

Step 4: Data Preparation

The application selects the required customer feature columns and converts "TotalCharges" into a numeric format.

Step 5: Model Loading

The saved pipeline is loaded from:

"models/churn_pipeline.pkl"

The project uses Joblib for loading the saved object.

Step 6: Churn Probability Prediction

The prediction module calculates the probability of churn using:

"predict_proba()"

The model returns a probability for the positive churn class.

Step 7: Customer Classification

Customers are classified using a probability threshold.

The default threshold in the prediction function is 0.30.

The classification logic is:

if churn_probability >= threshold:
    prediction = "Yes"
else:
    prediction = "No"

Step 8: Retention Recommendations

The recommendation module generates actions based on customer characteristics.

Step 9: Dashboard

The application calculates:

- Total customers.
- High-risk customers.
- Low-risk customers.
- High-risk customer percentage.

It also displays a bar chart showing the distribution of predicted customer risk.

Step 10: Download Results

Users can download the final prediction results as:

"customer_churn_predictions.csv"

9. Prediction Module

File:

"src/prediction.py"

The prediction module contains two main functions.

load_churn_pipeline()

Purpose:

Loads the saved Machine Learning pipeline using Joblib.

Default model path:

"models/churn_pipeline.pkl"

predict_customer_churn()

Purpose:

Generates churn probabilities and classifications.

Inputs:

- "pipeline": Loaded Machine Learning pipeline.
- "customer_data": Customer feature DataFrame.
- "threshold": Probability threshold, defaulting to 0.30.

Outputs:

A DataFrame containing the original customer features and two additional columns:

Output Column| Description
Churn Probability| Estimated probability of churn
Churn Prediction| Predicted churn category

The prediction function uses the model's positive-class probability and NumPy to apply the classification threshold.

10. Retention Recommendation Module

File:

"src/recommendations.py"

The recommendation module provides customer retention suggestions.

It analyzes selected customer characteristics and generates recommendations for customers predicted to churn.

Recommendation Rules

1. Contract Type

If the customer has a month-to-month contract, the system recommends offering a longer-term contract discount.

2. Customer Tenure

If customer tenure is less than 12 months, the system recommends an onboarding benefit.

3. Monthly Charges

If monthly charges are greater than 80, the system recommends a pricing review or cost-saving plan.

4. Technical Support

If technical support is not subscribed to, the system recommends offering technical support.

5. Online Security

If online security is not subscribed to, the system recommends providing information about online security services.

6. Default Recommendation

If none of the above conditions apply, the system recommends regular engagement check-ins.

Recommendation Output

The module adds a new column:

"Retention Recommendations"

For customers predicted not to churn, the system displays:

"No immediate retention action required."

These recommendations are rule-based business suggestions, not predictions generated by a separate recommendation Machine Learning model.

11. Dashboard Metrics

The application displays four main metrics.

Total Customers

The total number of customer records processed.

High-Risk Customers

The number of customers classified as likely to churn.

Low-Risk Customers

The number of customers classified as unlikely to churn.

High-Risk Percentage

Calculated using:

[
\text{High Risk Percentage} =
\frac{\text{High Risk Customers}}{\text{Total Customers}}
\times 100
]

Risk Distribution

A bar chart displays the number of customers in each predicted churn category.

12. Installation and Setup

Prerequisites

Install the following before running the project:

- Python 3.9 or a compatible version for the installed dependencies.
- pip.
- Git (optional, for cloning the repository).
- Visual Studio Code or another Python IDE.

The exact Python version compatibility should be checked against the installed package versions and saved model artifacts.

Step 1: Clone the Repository

git clone https://github.com/kumarharsh3012-netizen/Customer_Churn_AI.git

Step 2: Open the Project Directory

cd Customer_Churn_AI

Step 3: Create a Virtual Environment

Windows:

python -m venv venv

Step 4: Activate the Virtual Environment

Windows Command Prompt:

venv\Scripts\activate

Windows PowerShell:

.\venv\Scripts\Activate.ps1

Step 5: Install Dependencies

python -m pip install --upgrade pip

pip install -r requirements.txt

Step 6: Run the Application

streamlit run app.py

The application will start locally.

Streamlit normally provides a local address:

"http://localhost:8501"

Open this address in your browser.

Alternative Windows Startup

The project also contains:

"run_app.bat"

This file is intended to provide a convenient Windows startup method. Its exact behavior depends on its current implementation.

13. Dependencies

The current "requirements.txt" includes:

pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
shap
joblib
imbalanced-learn
scipy
openpyxl
xgboost
plotly
jupyter
ipykernel

These packages support data manipulation, visualization, Machine Learning, model serialization, and application development.

14. Model Storage and Serialization

The project stores trained Machine Learning artifacts in the "models/" directory.

The current application loads:

"models/churn_pipeline.pkl"

The other two saved artifacts are:

- "churn_model.pkl"
- "preprocessing.pkl"

Their exact classes, internal components, and relationship to the main pipeline should be verified before independently loading or modifying them.

Joblib is used because it supports serialization of Python Machine Learning objects, including many Scikit-learn estimators and pipelines.

Important: Only load pickle files from trusted sources. Serialized Python objects can execute code during loading.

15. Current Features and Limitations

Implemented Features

- Streamlit-based application.
- CSV customer dataset upload.
- Dataset preview and shape display.
- Required-column validation.
- Customer data preparation.
- Saved churn pipeline loading.
- Churn probability prediction.
- Binary churn classification.
- Rule-based retention recommendations.
- Customer risk dashboard.
- Risk distribution chart.
- CSV prediction report download.

Current Limitations

1. The existing churn model expects a specific set of 19 telecom customer features.
2. The current uploader accepts CSV files only.
3. The application is designed for binary customer churn classification.
4. The application does not currently provide a general-purpose classification and regression training interface.
5. The current prediction dashboard does not display a complete model evaluation report.
6. Retention recommendations use predefined business rules.
7. Model explainability is not currently integrated into the main application workflow.
8. Predictions depend on the compatibility of uploaded data with the saved pipeline.
9. The threshold slider is displayed in the application, but its selected value is not currently passed to the prediction function. This should be corrected so the slider actually controls the classification threshold.

16. Future Enhancements and Development Roadmap

The following features are proposed for future versions. They should not be considered implemented until the corresponding code is added and tested.

A. Premium Dark-Themed Interface

- Modern dark dashboard.
- Professional sidebar navigation.
- Improved metric cards.
- Interactive Plotly visualizations.
- Better dataset previews.
- Responsive layouts.

B. Multiple Dataset Support

Allow users to upload different datasets instead of restricting the application to the original telecom churn dataset.

Support for CSV and Excel files is planned.

C. General Supervised Machine Learning

Add two machine learning workflows:

Classification

- Logistic Regression.
- Decision Tree Classifier.
- Random Forest Classifier.
- Support Vector Machine.
- K-Nearest Neighbors.
- Gradient Boosting.
- XGBoost, where appropriate.

Regression

- Linear Regression.
- Decision Tree Regressor.
- Random Forest Regressor.
- Gradient Boosting Regressor.

D. Automated Data Preprocessing

- Missing value handling.
- Numerical feature processing.
- Categorical encoding.
- Feature selection.
- Train-test splitting.
- Reusable preprocessing pipelines.

E. Model Comparison

Compare multiple algorithms using appropriate evaluation metrics.

Classification metrics:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC, where applicable.

Regression metrics:

- Mean Absolute Error (MAE).
- Mean Squared Error (MSE).
- Root Mean Squared Error (RMSE).
- R-squared.

F. Model Explainability

Integrate explainability tools such as SHAP to help users understand which features influence model predictions.

G. Model Saving and Reuse

Allow users to save trained models and use them for future predictions on compatible datasets.

H. Advanced Analytics Dashboard

- Feature importance.
- Confusion matrix.
- ROC curve.
- Regression evaluation plots.
- Prediction distribution.
- Model performance comparison.

17. Project Security and Data Handling

- Do not upload confidential customer information to public repositories.
- Remove personally identifiable information from sample datasets before sharing.
- Keep credentials and private configuration values out of source code.
- Use trusted model artifacts only.
- Avoid committing large private datasets or unnecessary generated files.
- Validate uploaded files before processing them.

18. Running the Application

After installation, run:

streamlit run app.py

Upload a compatible customer CSV file.

The application will validate the data, generate churn predictions, produce retention recommendations, and display the results through its dashboard.

19. Expected Output

The application generates:

1. Uploaded customer dataset preview.
2. Churn probability for each customer.
3. Predicted churn category.
4. Retention recommendation.
5. Total customer count.
6. High-risk customer count.
7. Low-risk customer count.
8. High-risk customer percentage.
9. Customer risk distribution chart.
10. Downloadable CSV prediction report.

20. Conclusion

The AI-Powered Customer Churn Prediction and Retention System demonstrates how supervised Machine Learning can help businesses identify customers who may discontinue the
