ATM Withdrawal Forecasting Dashboard — Basic Version

Run on Windows from this folder:
1. python -m venv .venv
2. .venv\Scripts\activate
3. python -m pip install --upgrade pip
4. python -m pip install -r requirements.txt
5. python -m streamlit run app.py

Required CSV columns:
Timestamp, ATM_ID, Withdrawal_Amount, Transaction_ID, ATM_Type, Day_of_Week, Is_Weekend

This simplified app.py does not import NumPy directly and calculates MAE/RMSE using basic Python arithmetic. However, pandas and scikit-learn use NumPy internally, so this code cannot itself eliminate an underlying NumPy binary incompatibility. If that error persists, the environment/packages must be corrected.

Use data covering many hours or days for meaningful evaluation. A few transactions from one hour are not enough to train a forecasting model.
