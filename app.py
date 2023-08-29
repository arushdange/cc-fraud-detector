import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import joblib

# Direct link to the hosted model file (replace with your link)
hosted_model_url = "https://github.com/arushdange/cc-fraud-detector/raw/main/trained_model.joblib"

# Load the hosted model
rf = joblib.load(hosted_model_url)

# Load Streamlit interface
st.title("Credit Card Fraud Detection")

# Create input fields for user input (adjust features accordingly)
input_features = st.text_input("Enter features separated by commas (V1,V2,...,Amount):")
input_features = input_features.split(',')

# Process user input
if len(input_features) == 30:  # Adjust the number of features accordingly
    input_features = [float(val.strip()) for val in input_features]

    # Make prediction using the trained model
    prediction = rf.predict([input_features])

    st.write("Predicted Class:", prediction[0])
else:
    st.write("Please enter the correct number of features.")
