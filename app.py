import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Load your dataset
data = None
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

# Split the data into features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Apply Random Under-Sampling to balance classes
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Scale the "Amount" column
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# Split the resampled and scaled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled_scaled, y_resampled, test_size=0.2, random_state=42)

# Create a RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Fit the model to the training data
rf.fit(X_train, y_train)

# Load Streamlit interface
st.title("Credit Card Fraud Detection")

# Create input fields for user input
input_features = st.text_input("Enter features separated by commas (V1,V2,...,Amount):")
input_features = input_features.split(',')

# Process user input
if len(input_features) == X_train.shape[1]:
    input_features = [float(val.strip()) for val in input_features]
    scaled_input = scaler.transform([input_features])
    
    # Make prediction using the trained model
    prediction = rf.predict(scaled_input)
    
    st.write("Predicted Class:", prediction[0])
else:
    st.write("Please enter the correct number of features.")

# Display model evaluation metrics
st.write("Model Evaluation Metrics:")
st.write("ROC AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))
st.write("Precision:", precision_score(y_test, rf.predict(X_test)))
st.write("Recall:", recall_score(y_test, rf.predict(X_test)))
st.write("F1-Score:", f1_score(y_test, rf.predict(X_test)))
