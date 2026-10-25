import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Building Material const forecasting")
st.title("Building Material Price Forecasting Dashborad ")

st.write(
    "Uplode the transaction data, "
)
st.info(
    "Expected CSV columns:date,year,month,week_of_year,region,product_category,sku,channel,customer_type,units,unit_price,revenue,housing_starts_index,lumber_price_index,mortgage_rate"
)

with st.sidebar:
    st.header("Dashboard Setting")
    moedl_choice = st.sekectbox(
        "Choose model",
        ["",""]
    )