#python -m pip uninstall -y numpy pandas scikit-learn
#python -m pip install --no-cache-dir numpy pandas scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="ATM Withdrawal Forecasting", page_icon="🏧", layout="wide")

st.title(" ATM Cash Withdrawal Forecasting Dashboard")
st.write(
    "Upload ATM transaction data, aggregate it hourly, create lag features, "
    "train a machine-learning model, and inspect test-period predictions."
)

st.info(
    "Expected CSV columns: Timestamp, ATM_ID, Withdrawal_Amount, Transaction_ID, "
    "ATM_Type, Day_of_Week, Is_Weekend. This dashboard predicts hourly withdrawal "
    "demand; it does not estimate the cash remaining inside the ATM."
)

with st.sidebar:
    st.header("Dashboard settings")
    model_choice = st.selectbox(
        "Choose model",
        ["Random Forest", "Linear Regression"]
    )
    test_fraction = st.slider(
        "Test-set proportion",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05
    )
    st.caption("The split is chronological; observations are not shuffled.")

uploaded_file = st.file_uploader("Upload ATM transaction CSV", type=["csv"])

if uploaded_file is None:
    st.subheader("Sample upload file")
    st.write("Use the sample CSV provided with this project to try the dashboard.")
    st.download_button(
        "Download sample ATM transaction CSV",
        data=open("sample_atm_transactions.csv", "rb").read(),
        file_name="sample_atm_transactions.csv",
        mime="text/csv"
    )
    st.stop()

try:
    transactions = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

required_columns = [
    "Timestamp", "ATM_ID", "Withdrawal_Amount", "Transaction_ID",
    "ATM_Type", "Day_of_Week", "Is_Weekend"
]
missing = [col for col in required_columns if col not in transactions.columns]
if missing:
    st.error("The uploaded file is missing these required columns: " + ", ".join(missing))
    st.stop()

transactions["Timestamp"] = pd.to_datetime(transactions["Timestamp"], errors="coerce")
transactions["Withdrawal_Amount"] = pd.to_numeric(
    transactions["Withdrawal_Amount"], errors="coerce"
)
transactions = transactions.dropna(subset=["Timestamp", "Withdrawal_Amount"])
transactions = transactions.sort_values("Timestamp").reset_index(drop=True)

if transactions.empty:
    st.error("No valid transaction rows were found.")
    st.stop()

atm_ids = transactions["ATM_ID"].dropna().unique()
if len(atm_ids) > 1:
    chosen_atm = st.selectbox("Select ATM to model", sorted(atm_ids))
    transactions = transactions[transactions["ATM_ID"] == chosen_atm].copy()
else:
    chosen_atm = atm_ids[0] if len(atm_ids) else "ATM_001"
    st.write(f"**ATM selected:** {chosen_atm}")

st.subheader("1. Uploaded transaction data")
st.caption(f"{len(transactions):,} transactions | {transactions['Timestamp'].min()} to {transactions['Timestamp'].max()}")
st.dataframe(transactions.head(20), use_container_width=True)

# Aggregate individual transactions into hourly totals.
transactions["Hour_Timestamp"] = transactions["Timestamp"].dt.floor("h")
hourly = transactions.groupby("Hour_Timestamp").agg(
    Total_Withdrawal=("Withdrawal_Amount", "sum"),
    Transaction_Count=("Transaction_ID", "count"),
    Avg_Withdrawal=("Withdrawal_Amount", "mean")
)

# Add missing hours as zero-withdrawal hours.
all_hours = pd.date_range(
    transactions["Timestamp"].min().floor("h"),
    transactions["Timestamp"].max().floor("h"),
    freq="h"
)
hourly = hourly.reindex(all_hours, fill_value=0)
hourly.index.name = "Timestamp"
hourly = hourly.reset_index()
hourly["Avg_Withdrawal"] = hourly["Avg_Withdrawal"].fillna(0)
hourly["Hour"] = hourly["Timestamp"].dt.hour
hourly["Day_of_Week_Num"] = hourly["Timestamp"].dt.dayofweek
hourly["Is_Weekend"] = (hourly["Timestamp"].dt.dayofweek >= 5).astype(int)

st.subheader("2. Hourly aggregated data")
st.dataframe(hourly.head(20), use_container_width=True)
st.download_button(
    "Download hourly aggregated data",
    data=hourly.to_csv(index=False).encode("utf-8"),
    file_name="atm_hourly_aggregated.csv",
    mime="text/csv"
)

# Lag features: use only information available before the target hour.
model_data = hourly.copy()
model_data["Lag_1"] = model_data["Total_Withdrawal"].shift(1)
model_data["Lag_2"] = model_data["Total_Withdrawal"].shift(2)
model_data["Lag_1_Count"] = model_data["Transaction_Count"].shift(1)
model_data = model_data.dropna().reset_index(drop=True)

if len(model_data) < 20:
    st.error("There are too few hourly records to train and evaluate this model. Upload a longer time series.")
    st.stop()

features = [
    "Lag_1", "Lag_2", "Lag_1_Count",
    "Hour", "Day_of_Week_Num", "Is_Weekend"
]
target = "Total_Withdrawal"

split_index = int(len(model_data) * (1 - test_fraction))
if split_index < 10 or len(model_data) - split_index < 2:
    st.error("Not enough records for the selected train/test split. Try a smaller test-set proportion.")
    st.stop()

train_data = model_data.iloc[:split_index].copy()
test_data = model_data.iloc[split_index:].copy()

X_train, y_train = train_data[features], train_data[target]
X_test, y_test = test_data[features], test_data[target]

if model_choice == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2
    )

model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

st.subheader("3. Model performance")
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Model", model_choice)
metric2.metric("MAE", f"{mae:,.2f}")
metric3.metric("RMSE", f"{rmse:,.2f}")
metric4.metric("Test rows", f"{len(test_data):,}")

st.caption("Lower MAE and RMSE indicate smaller prediction errors. Metrics are calculated in the original withdrawal units.")

# Build the prediction table.
results = test_data[["Timestamp", "Total_Withdrawal"]].copy()
results = results.rename(columns={"Total_Withdrawal": "Actual_Withdrawal"})
results["Predicted_Withdrawal"] = predictions
results["Prediction_Error"] = results["Actual_Withdrawal"] - results["Predicted_Withdrawal"]
results["Absolute_Error"] = results["Prediction_Error"].abs()

st.subheader("4. Actual vs predicted withdrawals")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(results["Timestamp"], results["Actual_Withdrawal"], label="Actual", linewidth=2)
ax.plot(results["Timestamp"], results["Predicted_Withdrawal"], label="Predicted", linewidth=1.7)
ax.set_xlabel("Timestamp")
ax.set_ylabel("Hourly total withdrawal")
ax.set_title(f"{chosen_atm}: Actual vs Predicted ({model_choice})")
ax.legend()
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("5. Prediction dataframe")
st.dataframe(
    results.style.format({
        "Actual_Withdrawal": "{:,.2f}",
        "Predicted_Withdrawal": "{:,.2f}",
        "Prediction_Error": "{:,.2f}",
        "Absolute_Error": "{:,.2f}"
    }),
    use_container_width=True
)
st.download_button(
    "Download predictions CSV",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="atm_withdrawal_predictions.csv",
    mime="text/csv"
)

st.subheader("6. Download prepared lagged dataset")
st.download_button(
    "Download lagged modelling data",
    data=model_data.to_csv(index=False).encode("utf-8"),
    file_name="atm_hourly_lagged_data.csv",
    mime="text/csv"
)

with st.expander("How this dashboard works"):
    st.markdown(
        """
        1. Reads transaction-level records from the uploaded CSV.
        2. Aggregates withdrawals into hourly totals, counts, and average amounts.
        3. Adds Lag 1 and Lag 2 features using previous hourly withdrawals.
        4. Splits the data chronologically into training and testing sets.
        5. Trains the selected model and predicts withdrawals for the test period.
        6. Displays MAE/RMSE, an actual-versus-predicted chart, and a downloadable prediction table.
        """
    )
