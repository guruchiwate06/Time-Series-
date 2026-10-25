import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="ATM Withdrawal Forecasting", layout="wide")
st.title("ATM Withdrawal Forecasting Dashboard")
st.write("Upload ATM transaction data to aggregate hourly withdrawals and compare actual values with model predictions.")

REQUIRED_COLUMNS = [
    "Timestamp", "ATM_ID", "Withdrawal_Amount", "Transaction_ID",
    "ATM_Type", "Day_of_Week", "Is_Weekend"
]

uploaded_file = st.file_uploader("Upload ATM transaction CSV", type=["csv"])

if uploaded_file is not None:
    try:
        transactions = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Could not read the CSV file. Please check that it is a valid CSV.")
        st.stop()

    missing = [column for column in REQUIRED_COLUMNS if column not in transactions.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.write("Required columns:", REQUIRED_COLUMNS)
        st.stop()

    transactions["Timestamp"] = pd.to_datetime(transactions["Timestamp"], errors="coerce")
    transactions["Withdrawal_Amount"] = pd.to_numeric(transactions["Withdrawal_Amount"], errors="coerce")
    transactions = transactions.dropna(subset=["Timestamp", "Withdrawal_Amount"])

    if transactions.empty:
        st.error("No valid rows remain after checking Timestamp and Withdrawal_Amount.")
        st.stop()

    atm_ids = sorted(transactions["ATM_ID"].dropna().astype(str).unique().tolist())
    if not atm_ids:
        st.error("No ATM_ID values were found.")
        st.stop()

    selected_atm = st.selectbox("Select ATM", atm_ids)
    transactions = transactions[transactions["ATM_ID"].astype(str) == selected_atm].copy()
    transactions = transactions.sort_values("Timestamp")

    st.subheader("Uploaded transactions")
    st.dataframe(transactions, use_container_width=True)

    # Aggregate transaction records into hourly withdrawal totals.
    transactions = transactions.set_index("Timestamp")
    hourly = transactions["Withdrawal_Amount"].resample("h").sum().to_frame()
    hourly = hourly.rename(columns={"Withdrawal_Amount": "Total_Withdrawal"})

    # Add hours without transactions and fill them with zero.
    if not hourly.empty:
        full_hours = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
        hourly = hourly.reindex(full_hours, fill_value=0)
        hourly.index.name = "Timestamp"

    hourly["Hour"] = hourly.index.hour
    hourly["Day_of_Week_Num"] = hourly.index.dayofweek
    hourly["Is_Weekend_Calculated"] = (hourly["Day_of_Week_Num"] >= 5).astype(int)
    hourly["Lag_1"] = hourly["Total_Withdrawal"].shift(1)
    hourly["Lag_2"] = hourly["Total_Withdrawal"].shift(2)
    hourly["Lag_3"] = hourly["Total_Withdrawal"].shift(3)

    st.subheader("Hourly withdrawal totals")
    st.dataframe(hourly.reset_index(), use_container_width=True)
    st.download_button(
        "Download hourly data",
        data=hourly.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="atm_hourly_data.csv",
        mime="text/csv",
    )

    model_data = hourly.dropna(subset=["Lag_1", "Lag_2", "Lag_3"]).copy()
    feature_columns = ["Lag_1", "Lag_2", "Lag_3", "Hour", "Day_of_Week_Num", "Is_Weekend_Calculated"]

    if len(model_data) < 10:
        st.warning("Not enough hourly observations to train a model. Upload data covering more hours or days.")
        st.stop()

    test_fraction = st.slider("Test data proportion", min_value=0.2, max_value=0.4, value=0.2, step=0.05)
    split_at = int(len(model_data) * (1 - test_fraction))
    if split_at < 5 or len(model_data) - split_at < 2:
        st.warning("Not enough observations for this train/test split. Upload more data or adjust the test proportion.")
        st.stop()

    model_name = st.selectbox("Choose model", ["Linear Regression", "Random Forest"])
    X_train = model_data[feature_columns].iloc[:split_at]
    y_train = model_data["Total_Withdrawal"].iloc[:split_at]
    X_test = model_data[feature_columns].iloc[split_at:]
    y_test = model_data["Total_Withdrawal"].iloc[split_at:]

    if model_name == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    try:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    except Exception as error:
        st.error("The model could not be trained. Check the uploaded data and installed packages.")
        st.code(str(error))
        st.stop()

    actual_values = y_test.tolist()
    predicted_values = [float(value) for value in predictions]
    absolute_errors = [abs(actual - predicted) for actual, predicted in zip(actual_values, predicted_values)]
    squared_errors = [(actual - predicted) ** 2 for actual, predicted in zip(actual_values, predicted_values)]
    mae = sum(absolute_errors) / len(absolute_errors)
    rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5

    st.subheader("Model evaluation")
    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mae:,.2f}")
    col2.metric("RMSE", f"{rmse:,.2f}")

    results = pd.DataFrame({
        "Timestamp": model_data.index[split_at:],
        "Actual_Withdrawal": actual_values,
        "Predicted_Withdrawal": predicted_values,
    }).set_index("Timestamp")

    st.subheader("Actual vs predicted withdrawals")
    st.line_chart(results[["Actual_Withdrawal", "Predicted_Withdrawal"]])

    st.subheader("Prediction results")
    st.dataframe(results.reset_index(), use_container_width=True)
    st.download_button(
        "Download predictions",
        data=results.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="atm_withdrawal_predictions.csv",
        mime="text/csv",
    )

    st.caption("This forecasts hourly withdrawal demand, not remaining ATM cash balance. It evaluates the later part of the uploaded data and does not forecast beyond the uploaded time range.")
else:
    st.info("Upload a CSV file to begin.")
    st.write("Required columns:", REQUIRED_COLUMNS)
