import streamlit as st 
import tensorflow as tf 
import numpy as np 
from PIL import Image 
from transformers import T5Tokenizer, T5ForConditionalGeneration 
from googletrans import Translator 
from fpdf import FPDF 
import io

------------------ INIT ------------------

st.set_page_config(page_title="AgroScan: Plant Doctor", layout="centered")

st.title("🌿 AgroScan: AI-Powered Plant Disease Detection")

------------------ LOAD MODEL ------------------

@st.cache_resource def load_cnn_model(): return tf.keras.models.load_model("plant_disease_model.keras")

@st.cache_resource def load_llm(): tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small") model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small") return tokenizer, model

cnn_model = load_cnn_model() tokenizer, llm_model = load_llm() translator = Translator()

------------------ HELPER FUNCS ------------------

def predict_disease(image): img = image.resize((160, 160)) img_array = np.expand_dims(np.array(img)/255.0, axis=0) prediction = cnn_model.predict(img_array)[0] class_names = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Healthy'] pred_index = np.argmax(prediction) return class_names[pred_index]

def generate_treatment(disease, follow_up): prompt = f"What is the treatment for {disease} in tomato? Also consider: {follow_up}" inputs = tokenizer(prompt, return_tensors="pt") outputs = llm_model.generate(**inputs, max_new_tokens=100) return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_pdf(disease, treatment): pdf = FPDF() pdf.add_page() pdf.set_font("Arial", size=12) pdf.cell(200, 10, txt="AgroScan Diagnosis Report", ln=1, align="C") pdf.ln(10) pdf.multi_cell(0, 10, f"Disease: {disease}\n\nTreatment Advice: {treatment}") pdf_output = io.BytesIO() pdf.output(pdf_output) pdf_output.seek(0) return pdf_output

def translate_text(text, lang): return translator.translate(text, dest=lang).text

------------------ UI ------------------

st.subheader("📷 Upload Leaf Image") img_file = st.file_uploader("Upload a photo of the leaf", type=["jpg", "jpeg", "png"])

follow_up = st.text_input("🌱 Any specific concerns or follow-up questions?") language = st.selectbox("🌍 Preferred Language", ["English", "Yoruba", "Hausa", "Igbo"])

if st.button("🔍 Diagnose") and img_file: image = Image.open(img_file) st.image(image, caption="Uploaded Leaf", use_column_width=True)

disease = predict_disease(image)
st.success(f"🩺 Detected Disease: {disease}")

treatment = generate_treatment(disease, follow_up)

if language != "English":
    disease_translated = translate_text(disease, language.lower()[:2])
    treatment_translated = translate_text(treatment, language.lower()[:2])
else:
    disease_translated, treatment_translated = disease, treatment

st.markdown("### 💊 Treatment Advice")
st.info(treatment_translated)

pdf = generate_pdf(disease_translated, treatment_translated)
st.download_button("📄 Download PDF Report", data=pdf, file_name="AgroScan_Report.pdf")

