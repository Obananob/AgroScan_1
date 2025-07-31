import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from twilio.rest import Client
import os
import requests
from dotenv import load_dotenv

load_dotenv()

FastAPI_URL = "https://agroscan1-production.up.railway.app" 

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE_NUMBER")

twilio_client = Client(TWILIO_SID, TWILIO_AUTH)

# STREAMLIT SETUP
st.set_page_config(page_title="AgroScan", page_icon="🌿")
st.title("🌿 AgroScan: Plant Disease Detector")
st.markdown("Upload a leaf image to detect possible plant diseases and optionally send results via SMS.")

#  IMAGE UPLOAD 
uploaded_file = st.file_uploader("📷 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    #PREDICTION 
    with st.spinner("🔍 Analyzing image..."):
        response = requests.post(
            FastAPI_URL,
            files={"file": ("leaf.png", img_bytes, "image/png")}
        )

    if response.status_code == 200:
        result = response.json()
        st.success(f" Disease Detected: **{result['class']}**")
        st.info(f"Confidence: `{result['confidence']:.2f}`")

        if result["class"].lower() == "uncertain":
            st.warning("⚠️ Unable to confidently identify the disease. Try a clearer image.")
        else:
            #  SMS SECTION 
            st.markdown("### Send Diagnosis via SMS")
            phone_number = st.text_input("Enter recipient phone number (e.g. +234...)", max_chars=15)

            if st.button("Send SMS"):
                sms_text = (
                    f"🌿 AgroScan Diagnosis:\n"
                    f"Disease: {result['class']}\n"
                    f"Confidence: {result['confidence']:.2f}"
                )
                try:
                    twilio_client.messages.create(
                        body=sms_text,
                        from_=TWILIO_PHONE,
                        to=phone_number
                    )
                    st.success("SMS sent successfully!")
                except Exception as e:
                    st.error(f" SMS failed: {str(e)}")
    else:
        st.error("Error processing image. Please try again.")

#  FOOTER
st.caption("AgroScan powered by FastAPI, TensorFlow, and Twilio 🌍")
