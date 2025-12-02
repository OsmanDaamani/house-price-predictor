import streamlit as st
import numpy as np
import joblib

# load model and scaler
model = joblib.load('linear_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🏠 House Price Predictor")
st.write("Fill the details below to predict the price of a house.")

# inputs
area = st.number_input("Area (m²)", min_value=30, max_value=500, value=120)
rooms = st.number_input("Rooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)
age = st.number_input("Building Age (years)", min_value=0, max_value=40, value=10)
floor = st.number_input("Floor", min_value=0, max_value=20, value=2)
location_score = st.number_input("Location Score (1–10)", min_value=1, max_value=10, value=7)
distance_to_center = st.number_input("Distance to Center (km)", min_value=0.1, max_value=30.0, value=5.0)
parking = st.selectbox("Parking", [0, 1])
elevator = st.selectbox("Elevator", [0, 1])

if st.button("Predict Price"):
    x = np.array([[area, rooms, bathrooms, age, floor, location_score, distance_to_center, parking, elevator]])
    x_scaled = scaler.transform(x)
    
    predicted_price = model.predict(x_scaled)[0]

    st.success(f"Estimated Price: **${predicted_price:,.0f}**")
   
