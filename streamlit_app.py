import streamlit as st
import requests
from PIL import Image
import io

st.title("AgroScan – Plant Disease Detector")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/predict", files=files)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Disease: {result['class']}")
            st.info(f"Confidence: {result['confidence']*100:.2f}%")
            # st.warning(f"Treatment Advice: {result['treatment']}")
        else:
            st.error("Prediction failed.")
