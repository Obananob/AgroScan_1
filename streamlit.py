import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
<<<<<<< HEAD
import io
from twilio.rest import Client
import os
from dotenv import load_dotenv

load_dotenv()

FastAPI_URL = "http://localhost:8000/predict" 

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
        st.success(f"✅ Disease Detected: **{result['class']}**")
        st.info(f"🧪 Confidence: `{result['confidence']:.2f}`")

        if result["class"].lower() == "uncertain":
            st.warning("⚠️ Unable to confidently identify the disease. Try a clearer image.")
        else:
            #  SMS SECTION 
            st.markdown("### 📱 Send Diagnosis via SMS")
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
                    st.success("✅ SMS sent successfully!")
                except Exception as e:
                    st.error(f"❌ SMS failed: {str(e)}")
    else:
        st.error("❌ Error processing image. Please try again.")

#  FOOTER
st.caption("AgroScan powered by FastAPI, TensorFlow, and Twilio 🌍")
=======
from transformers import T5Tokenizer, T5ForConditionalGeneration
from googletrans import Translator
from fpdf import FPDF
import io

# ------------------ INIT ------------------

st.set_page_config(page_title="AgroScan: Plant Doctor", layout="centered")

# Load logo
logo = Image.open("logo.png")

# Display in header
st.image(logo, width=120)
st.title("AgroScan: AI-Powered Plant Disease Diagnosis")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("agroscan_model.keras")

@st.cache_resource
def load_llm():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    return tokenizer, model

cnn_model = load_cnn_model()
tokenizer, llm_model = load_llm()
translator = Translator()

# ------------------ HELPER FUNCS ------------------

def preprocess_image(image, target_size=(160, 160)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:  # Remove alpha if present
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    return image

CLASS_NAMES = [
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

def predict_disease(image):
    img_array = preprocess_image(image)
    prediction = cnn_model.predict(img_array)[0]
    pred_index = np.argmax(prediction)
    return CLASS_NAMES[pred_index]

def generate_treatment(disease, follow_up):
    prompt = f"""
You are an agricultural expert assisting rural farmers. Given the plant disease: '{disease}', provide a clear, organic, low-cost treatment advice specifically for *plants*, not humans.

Avoid medical equipment or non-farm tools. Only mention substances available to farmers (e.g., neem oil, copper spray, pruning techniques, etc.).

Disease: {disease}
Treatment advice: Consider: {follow_up}"""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_pdf(disease, treatment):
    pdf = FPDF()
    pdf.add_page()

    try:
        pdf.image("logo.png", x=10, y=8, w=30)
    except:
        pass  # Skip if logo file not found

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 40, "AgroScan Disease Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Disease: {disease}\n\nTreatment Advice: {treatment}")

    pdf_output = io.BytesIO()
    pdf.output(pdf_output, 'S')
    pdf_output.seek(0)
    return pdf_output

def translate_text(text, lang):
    try:
        return translator.translate(text, dest=lang).text
    except Exception:
        return text  # fallback if translation fails

# ------------------ UI ------------------

st.subheader("📷 Upload Leaf Image")
img_file = st.file_uploader("Upload a photo of the leaf", type=["jpg", "jpeg", "png"])

follow_up = st.text_input("🌱 Any specific concerns or follow-up questions?")
language = st.selectbox("🌍 Preferred Language", ["English", "Yoruba", "Hausa", "Igbo"])

if st.button("🔍 Diagnose") and img_file:
    image = Image.open(img_file)
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    disease = predict_disease(image)
    st.success(f"🩺 Detected Disease: {disease}")

    treatment = generate_treatment(disease, follow_up)

    if language != "English":
        lang_code = language.lower()[:2]
        disease_translated = translate_text(disease, lang_code)
        treatment_translated = translate_text(treatment, lang_code)
    else:
        disease_translated = disease
        treatment_translated = treatment

    st.markdown("### 💊 Treatment Advice")
    st.info(treatment_translated)

    pdf = generate_pdf(disease_translated, treatment_translated)
    st.download_button("📄 Download PDF Report", data=pdf, file_name="AgroScan_Report.pdf")
>>>>>>> cdbc4c75ae06127061212396becf336a98171a17
