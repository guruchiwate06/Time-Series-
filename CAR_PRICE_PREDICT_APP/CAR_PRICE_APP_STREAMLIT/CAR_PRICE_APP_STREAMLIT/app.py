import streamlit as st
import pickle
import numpy as np

# Title and Image
st.image("images/price.png", width=350)
st.title("Car Price Prediction")
st.write("This app predicts the estimated selling price of a car")

# Load model
model = pickle.load(open("models/lr_model.pkl", "rb"))

# User Inputs
oprice = st.number_input("Enter original price (in Lakhs)", min_value=1.0, max_value=100.0, value=5.0)
kms = st.number_input("Enter kilometers driven", min_value=100, max_value=500000, value=25000)
age = st.number_input("Enter car age (in years)", min_value=0, max_value=30, value=5)
fuel_type = st.selectbox("Select fuel type", ["Petrol", "Diesel", "CNG"])
transmission = st.selectbox("Select transmission", ["Manual", "Automatic"])

# Encoding fuel type
if fuel_type == "Petrol":
    fuel = [0.0, 0.0, 1.0]
elif fuel_type == "Diesel":
    fuel = [0.0, 1.0, 0.0]
elif fuel_type == "CNG":
    fuel = [1.0, 0.0, 0.0]

# Encoding transmission
if transmission == "Automatic":
    trans = [0.0, 1.0]
else:
    trans = [1.0, 0.0]

# Predict button
if st.button("Predict Price"):
    data = np.array([[oprice, kms, age, fuel[0], fuel[1], fuel[2], trans[0], trans[1]]])
    result = np.round(model.predict(data)[0], 2)
    st.write("Results = " + str(result) + " Lakhs")
