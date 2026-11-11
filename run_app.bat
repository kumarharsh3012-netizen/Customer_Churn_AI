@echo off
cd /d "%~dp0"

echo Starting Customer Churn Prediction App...
echo Please wait...

.venv\Scripts\python.exe -m streamlit run app.py

echo.
echo Application stopped or an error occurred.
pause