import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from twilio.rest import Client

# --- Title and Description ---
st.title("🌿 AgroScan – Plant Disease Detector")
st.markdown("Upload or snap a leaf image to detect the disease. Optionally, send the diagnosis via SMS.")
st.markdown("---")

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("agroscan_model.keras")

model = load_model()

class_names = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Target_Spot",
    "Tomato___healthy"
]
  

# --- Image Preprocessing ---
def preprocess_image(image):
    img = image.resize((160, 160))  

    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- Image Upload / Capture ---
uploaded_image = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("Or take a picture")

image = None
if uploaded_image:
    image = Image.open(uploaded_image)
elif camera_image:
    image = Image.open(camera_image)

# --- Prediction ---
if image:
    st.image(image, caption="Uploaded Leaf", use_container_width=True)
    with st.spinner("Analyzing the image..."):
        processed = preprocess_image(image)
        diagnosis = model.predict(processed)[0]
        detected_class = class_names[np.argmax(diagnosis)]
        confidence = round(100 * np.max(diagnosis), 2)

    st.success(f"**Disease Detected**: {detected_class}")
    st.info(f"**Confidence**: {confidence}%")

    # --- Optional SMS ---
    st.markdown("---")
    st.markdown("### 📩 Send Diagnosis via SMS (optional)")
    send_sms = st.checkbox("Send result via SMS")
    
    if send_sms:
        phone_number = st.text_input("Recipient phone number (e.g. +234...)", max_chars=15)
        if st.button("Send SMS"):
            if phone_number:
                try:
                    # Twilio Credentials from Streamlit secrets
                    client = Client(st.secrets["TWILIO_SID"], st.secrets["TWILIO_AUTH"])
                    message = client.messages.create(
                        body=f"AgroScan Result: {detected_class} ({confidence}%)",
                        from_=st.secrets["TWILIO_PHONE"],
                        to=phone_number
                    )
                    st.success("SMS sent successfully ✅")
                except Exception as e:
                    st.error(f"Error sending SMS: {e}")
            else:
                st.warning("Enter a valid phone number.")

# --- Footer ---
st.markdown(---)
st.caption("**AgroScan by ByteBuilders** 🚀")
