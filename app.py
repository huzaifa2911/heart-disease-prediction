
import streamlit as st
import joblib
import pandas as pd

st.title("Heart Disease Prediction System")
st.write("Enter the four attributes below.")

saved = joblib.load("heart_model.pkl")
model = saved["model"]
scaler = saved["scaler"]
features = saved["features"]

age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex (dataset coding)", [0, 1])
trestbps = st.number_input(
    "Resting blood pressure (mm Hg)",
    min_value=50, max_value=250, value=120
)
thalach = st.number_input(
    "Maximum heart rate achieved",
    min_value=50, max_value=250, value=150
)

if st.button("Predict"):
    data = pd.DataFrame(
        [[age, sex, trestbps, thalach]],
        columns=features
    )

    scaled_data = scaler.transform(data)
    prediction = int(model.predict(scaled_data)[0])

    if prediction == 1:
        st.warning("The model predicts the positive class.")
    else:
        st.success("The model predicts the negative class.")

    st.caption(
        "Educational demonstration only; not a medical diagnosis."
    )
