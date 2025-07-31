
import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("Sleep Disorder Classification")

def user_input_features():
    snoring = st.selectbox("Snoring Rate", [0, 1, 2])
    age = st.slider("Age", 18, 90)
    bmi = st.slider("BMI", 10.0, 50.0)
    physical_activity = st.slider("Physical Activity Level (hrs/week)", 0, 20)
    alcohol = st.selectbox("Alcohol Consumption", ["Low", "Medium", "High"])
    smoking = st.selectbox("Smoking Status", ["Yes", "No"])
    stress = st.slider("Stress Level", 1, 10)
    
    data = {
        "Snoring Rate": snoring,
        "Age": age,
        "BMI": bmi,
        "Physical Activity Level": physical_activity,
        "Alcohol Consumption": alcohol,
        "Smoking Status": smoking,
        "Stress Level": stress
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Encode categorical fields
for col in input_df.columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Predict
if st.button("Predict Sleep Disorder"):
    prediction = model.predict(input_df)[0]
    disorder_label = label_encoders["Sleep Disorder"].inverse_transform([prediction])[0]
    st.success(f"Predicted Sleep Disorder: {disorder_label}")
