import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import zipfile
from googletrans import Translator
from fpdf import FPDF
import io
import datetime
import os

# Constants
MODEL_ZIP_PATH = "model.zip"
EXTRACT_DIR = "unzipped_model"
MODEL_PATH = os.path.join(EXTRACT_DIR, "agroscan.keras")

# Step 1 – Unzip the model
if not os.path.exists(MODEL_PATH):
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

# -------------------- APP CONFIG --------------------
st.set_page_config(page_title="AgroScan – AI-Powered Plant Disease Detector", layout="centered")
st.title("🌿 AgroScan – AI-Powered Plant Disease Detector")
st.markdown("Upload a plant leaf image or take a photo to detect diseases using AI.")

# -------------------- MODEL & CLASSES --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

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

# -------------------- IMAGE PREPROCESSING --------------------
def preprocess_image(image):
    image = image.resize((160, 160))  # match training size
    image = np.array(image) / 255.0   # normalize
    if image.shape[-1] != 3:
        image = np.stack((image,) * 3, axis=-1)  # ensure RGB
    image = np.expand_dims(image, axis=0)       # add batch dimension
    return image

# -------------------- DISEASE TREATMENT ADVICE --------------------
def get_treatment_advice(disease):
    treatments = {
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Apply strobilurins or triazoles. Practice crop rotation and good sanitation.",
        "Corn_(maize)___Common_rust_": "Use resistant hybrids. Apply tebuconazole-based fungicide if needed.",
        "Corn_(maize)___healthy": "No disease. Maintain good agronomic practices.",
        "Pepper,_bell___Bacterial_spot": "Use copper-based bactericides. Avoid overhead watering. Rotate crops.",
        "Pepper,_bell___healthy": "Healthy plant. Monitor for early signs.",
        "Potato___Late_blight": "Apply chlorothalonil or mancozeb. Remove infected foliage.",
        "Potato___healthy": "No disease. Maintain weed control and scout regularly.",
        "Tomato___Bacterial_spot": "Use copper sprays. Avoid wet foliage. Start with clean seeds.",
        "Tomato___Early_blight": "Apply azoxystrobin. Rotate crops and remove infected leaves.",
        "Tomato___Late_blight": "Use systemic fungicides. Destroy infected plants quickly.",
        "Tomato___Leaf_Mold": "Improve ventilation. Use sulfur-based sprays.",
        "Tomato___Target_Spot": "Apply azoxystrobin. Avoid excessive watering.",
        "Tomato___healthy": "Plant is healthy. Keep monitoring and maintain soil health."
    }
    return treatments.get(disease, "No treatment advice available.")

# -------------------- PDF REPORT GENERATION --------------------
def generate_pdf(disease, advice):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"AgroScan Report – {datetime.date.today()}", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Diagnosis: {disease}\n\nRecommended Treatment:\n{advice}")
    pdf_output = pdf.output(dest='S').encode('latin-1')
    buffer = io.BytesIO(pdf_output)
    buffer.seek(0)
    return buffer

# -------------------- TRANSLATOR --------------------
translator = Translator()

# -------------------- STREAMLIT INTERFACE --------------------
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])
camera_input = st.camera_input("📷 Or take a photo")

if uploaded_file or camera_input:
    image = Image.open(uploaded_file if uploaded_file else camera_input).convert("RGB")
    st.image(image, caption="Analyzing this image...", use_column_width=True)

    with st.spinner("Processing..."):
        processed = preprocess_image(image)
        predictions = model.predict(processed)
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions)) * 100

    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    advice = get_treatment_advice(predicted_class)
    st.markdown(f"**Treatment Advice:** {advice}")

    # Language translation
    lang = st.selectbox("🌍 Translate to:", ["English", "French", "Arabic", "Hausa", "Yoruba", "Igbo"])
    if lang != "English":
        translated_text = translator.translate(advice, dest=lang.lower()).text
        st.markdown(f"**Translated Advice ({lang}):** {translated_text}")

    # PDF Download
    if st.button("📄 Generate PDF Report"):
        pdf = generate_pdf(predicted_class, advice)
        st.download_button(label="Download Report", data=pdf, file_name="agroscan_report.pdf", mime="application/pdf")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("**AgroScan Roadmap**: Offline AI | WhatsApp Bot | Multilingual Support | Smart Crop Advice")
st.caption("© 2025 AgroScan | Built with ❤️ for African Farmers")