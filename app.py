import streamlit as st
import numpy as np
import joblib

# Loading trained model
model = joblib.load("model.pkl")

# Streamlit Web UI
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌺",
    layout="wide"
)
st.title("Iris Flower Predictor 🌺")
st.write("*Enter flower measurements to predict the species.*")

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.6)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 2.5)

    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.2)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Preparing input
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
target_names = ["Setosa", "Versicolor", "Virginica"]

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"🌺 Predicted Flower: **{target_names[prediction[0]]}**")