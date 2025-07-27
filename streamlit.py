import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
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
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_disease(image):
    img_array = preprocess_image(image)
    prediction = cnn_model.predict(img_array)[0]

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

    pred_index = np.argmax(prediction)
    return CLASS_NAMES[pred_index]

def generate_treatment(disease, follow_up):
    prompt = f"What is the treatment for {disease}? Consider: {follow_up}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_pdf(disease, treatment):
    pdf = FPDF()
    pdf.add_page()

    # Add logo
    pdf.image("A_vector_graphic_logo_design_for_AgroScan_features.png", x=10, y=8, w=30)

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 40, "AgroScan Disease Report", ln=True, align="C")

    # Body
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Disease: {disease}\n\nTreatment Advice: {treatment}")

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

def translate_text(text, lang):
    return translator.translate(text, dest=lang).text

# ------------------ UI ------------------
st.subheader("📷 Upload Leaf Image")
img_file = st.file_uploader("Upload a photo of the leaf", type=["jpg", "jpeg", "png"])

follow_up = st.text_input("🌱 Any specific concerns or follow-up questions?")
language = st.selectbox("🌍 Preferred Language", ["English", "Yoruba", "Hausa", "Igbo"])

if st.button("🔍 Diagnose") and img_file:
    image = Image.open(img_file)
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    disease = predict_disease(image)
    st.success(f"🩺 Detected Disease: {disease}")

    treatment = generate_treatment(disease, follow_up)

    if language != "English":
        lang_code = language.lower()[:2]
        disease_translated = translate_text(disease, lang_code)
        treatment_translated = translate_text(treatment, lang_code)
    else:
        disease_translated, treatment_translated = disease, treatment

    st.markdown("### 💊 Treatment Advice")
    st.info(treatment_translated)

    pdf = generate_pdf(disease_translated, treatment_translated)
    st.download_button("📄 Download PDF Report", data=pdf, file_name="AgroScan_Report.pdf")