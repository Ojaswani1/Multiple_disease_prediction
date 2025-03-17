import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load trained models
diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))
heart_model = pickle.load(open('heart_model.pkl', 'rb'))
parkinson_model = pickle.load(open('parkinson_model.pkl', 'rb'))

# Load scalers (if needed)
diabetes_scaler = pickle.load(open('diabetes_scaler.pkl', 'rb'))
parkinson_scaler = pickle.load(open('parkinson_scaler.pkl', 'rb'))

# Streamlit UI
st.title("Multiple Disease Prediction System 🏥")
st.sidebar.header("Select a Disease to Predict")
disease_option = st.sidebar.selectbox("Choose a Disease:", ["Diabetes", "Heart Disease", "Parkinson’s Disease"])

if disease_option == "Diabetes":
    st.subheader("Diabetes Prediction 🩸")
    Pregnancies = st.number_input("Pregnancies", min_value=0)
    Glucose = st.number_input("Glucose Level", min_value=0)
    BloodPressure = st.number_input("Blood Pressure", min_value=0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0)
    Insulin = st.number_input("Insulin Level", min_value=0)
    BMI = st.number_input("BMI", min_value=0.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    Age = st.number_input("Age", min_value=0)

    if st.button("Predict Diabetes"):
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        input_data_scaled = diabetes_scaler.transform(input_data)
        prediction = diabetes_model.predict(input_data_scaled)

        if prediction[0] == 1:
            st.error("The person is likely to have Diabetes.")
        else:
            st.success("The person is unlikely to have Diabetes.")

elif disease_option == "Heart Disease":
    st.subheader("Heart Disease Prediction ❤️")
    age = st.number_input("Age", min_value=0)
    sex = st.selectbox("Sex", [0, 1])  # 0 = Female, 1 = Male
    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0)
    chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    restecg = st.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0)
    exang = st.selectbox("Exercise-Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by state of rest", min_value=0.0)
    slope = st.number_input("Slope of Peak Exercise ST Segment", min_value=0, max_value=2)
    ca = st.number_input("Major Vessels (0-3)", min_value=0, max_value=3)
    thal = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3)

    if st.button("Predict Heart Disease"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = heart_model.predict(input_data)

        if prediction[0] == 1:
            st.error("The person is likely to have Heart Disease.")
        else:
            st.success("The person is unlikely to have Heart Disease.")


elif disease_option == "Parkinson’s Disease":
    st.subheader("Parkinson’s Disease Prediction 🧠")
    Fundamental_Frequency = st.number_input("Average vocal fundamental frequency", min_value=0.0)
    Max_Frequency = st.number_input("Maximum vocal fundamental frequency", min_value=0.0)
    Min_Frequency = st.number_input("Minimum vocal fundamental frequency", min_value=0.0)
    Jitter_Percent = st.number_input("Jitter (%)", min_value=0.0,format="%.6f")
    Jitter_Abs = st.number_input("Jitter (Abs)", min_value=0.0,format="%.6f")
    RAP = st.number_input("RAP", min_value=0.0,format="%.6f")
    PPQ = st.number_input("PPQ", min_value=0.0,format="%.6f")
    Jitter_DDP = st.number_input("DDP", min_value=0.0,format="%.6f")
    Shimmer = st.number_input("Shimmer (%)", min_value=0.0,format="%.6f")
    Shimmer_dB = st.number_input("Shimmer (dB)", min_value=0.0,format="%.6f")
    Shimmer_APQ3 = st.number_input("APQ3", min_value=0.0,format="%.6f")
    Shimmer_APQ5 = st.number_input("APQ5", min_value=0.0,format="%.6f")
    MDVP_APQ = st.number_input("APQ", min_value=0.0,format="%.6f")
    Shimmer_DDA = st.number_input("Shimmer_DDA", min_value=0.0,format="%.6f")
    NHR = st.number_input("NHR", min_value=0.0,format="%.6f")
    HNR = st.number_input("HNR (Harmonics-to-Noise Ratio)", min_value=0.0,format="%.6f")
    RPDE  = st.number_input("RPDE", min_value=0.0,format="%.6f")
    DFA = st.number_input("DFA", min_value=0.0,format="%.6f")
    spread1 = st.number_input("Spread1", min_value=-100.0,format="%.6f")
    spread2 = st.number_input("Spread2", min_value=0.0,format="%.6f")
    D2  = st.number_input("D2", min_value=0.0,format="%.6f")
    PPE = st.number_input("PPE", min_value=0.0,format="%.6f")

    if st.button("Predict Parkinson’s Disease"):
        input_data = np.array([[Fundamental_Frequency, Max_Frequency, Min_Frequency, Jitter_Percent, Jitter_Abs,RAP,PPQ,Jitter_DDP, Shimmer,Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,NHR, HNR , RPDE,DFA,spread1,spread2,D2,PPE]])
        input_data_scaled = parkinson_scaler.transform(input_data)
        prediction = parkinson_model.predict(input_data_scaled)

        if prediction[0] == 1:
            st.error("The person is likely to have Parkinson’s Disease.")
        else:
            st.success("The person is unlikely to have Parkinson’s Disease.")
