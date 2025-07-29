import streamlit as st 
import numpy as np 
import tensorflow as tf 
from PIL import Image 
from googletrans import Translator 
from fpdf import FPDF 
import io 
import datetime


model = tf.keras.models.load_model("agroscan_model.keras") 

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

translator = Translator()


def preprocess_image(image): 
image = image.resize((160, 160)) 
image = np.array(image) / 255.0 
image = np.expand_dims(image, axis=0) return image


def generate_pdf(disease, advice): 

pdf = FPDF() 
pdf.add_page() 
pdf.set_font("Arial", size=12) pdf.cell(200, 10, txt=f"AgroScan Disease Report - {datetime.date.today()}", ln=True, align='C') pdf.ln(10) 
pdf.multi_cell(0, 10, f"Prediction: {disease}\n\nTreatment Advice: {advice}") buffer = io.BytesIO() 
pdf.output(buffer) 
buffer.seek(0)
return buffer


def get_treatment_advice(disease): treatments = { "Bacterial Spot": "Remove affected leaves. Apply copper-based fungicide.", "Early Blight": "Use fungicides with chlorothalonil. Rotate crops.", "Late Blight": "Destroy infected plants. Apply mancozeb-based spray.", "Leaf Mold": "Ensure ventilation. Use sulfur-based fungicide.", "Septoria Leaf Spot": "Remove infected leaves. Use fungicides like azoxystrobin.", "Spider Mites": "Spray with neem oil. Control weeds nearby.", "Target Spot": "Apply fungicides and reduce leaf wetness.", "Yellow Leaf Curl Virus": "Control whiteflies. Use resistant varieties.", "Mosaic Virus": "Remove infected plants. Use virus-free seeds.", "Healthy": "No treatment needed. Continue good care." } return treatments.get(disease, "Consult an agricultural extension officer.")


st.set_page_config(page_title="AgroScan – AI-Powered Plant Disease Detector", layout="centered") st.title("\U0001F331 AgroScan – AI-Powered Plant Disease Detector") 
st.markdown("Upload a leaf image or use your camera. For best results, choose your back camera manually if on mobile.")

image_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"]) camera_input = st.camera_input("Or take a photo")

if image_file or camera_input: 
image = Image.open(image_file if image_file else camera_input) st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

with st.spinner("Analyzing the image..."):
    input_data = preprocess_image(image)
    prediction = model.predict(input_data)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction)) * 100

st.success(f"**Detected:** {predicted_class} ({confidence:.2f}%)")

advice = get_treatment_advice(predicted_class)
st.info(f"**Advice:** {advice}")

language = st.selectbox("Translate to language", ["English", "Hausa", "Yoruba", "Igbo", "French", "Arabic"])
if language != "English":
    translated_advice = translator.translate(advice, dest=language.lower()).text
    st.markdown(f"**Translated Advice ({language}):** {translated_advice}")

# PDF generation
if st.button("📄 Download PDF Report"):
    pdf_buffer = generate_pdf(predicted_class, advice)
    st.download_button("Download Report", data=pdf_buffer, file_name="agroscan_report.pdf", mime="application/pdf")



st.markdown("---") 
st.markdown("Future Work & Roadmap: Offline mode | WhatsApp bot | Auto-treatment | Local language support | Farm dashboard") 
st.caption("© 2025 AgroScan | Built for Africa Deep Tech Challenge")

