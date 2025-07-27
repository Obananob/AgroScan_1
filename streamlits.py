import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from googletrans import Translator
from fpdf import FPDF
import openai
import io

# ------------------ INIT ------------------

st.set_page_config(page_title="AgroScan: Plant Doctor", layout="centered")

# Load logo
logo = Image.open("logo.png")
st.image(logo, width=120)
st.title("AgroScan: AI-Powered Plant Disease Diagnosis")

# ------------------ OPENAI API KEY ------------------

openai.api_key = st.secrets["OPENAI_API_KEY"]  # Store this in .streamlit/secrets.toml

# ------------------ LOAD MODEL ------------------

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("agroscan_model.keras")

cnn_model = load_cnn_model()
translator = Translator()

# ------------------ DISEASE CLASSES ------------------

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

# ------------------ HELPER FUNCS ------------------

def preprocess_image(image, target_size=(160, 160)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.expand_dims(image, axis=0)

def predict_disease(image):
    img_array = preprocess_image(image)
    prediction = cnn_model.predict(img_array)[0]
    pred_index = np.argmax(prediction)
    return CLASS_NAMES[pred_index]

def get_openai_treatment_advice(disease, follow_up):
    prompt = f"""
You are an agricultural assistant for rural farmers. Provide an organic, cost-effective treatment for the plant disease: '{disease}'. 
Avoid human medicine or tools. Mention local substances (e.g., neem oil, ash, pruning) that farmers can use. 
Also consider: '{follow_up}'.
Respond in under 100 words.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful plant health expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI Error: {e}")
        return "⚠️ Could not fetch treatment advice."

def translate_text(text, lang):
    try:
        return translator.translate(text, dest=lang).text
    except:
        return text  # fallback

def generate_pdf(disease, treatment):
    pdf = FPDF()
    pdf.add_page()
    try:
        pdf.image("logo.png", x=10, y=8, w=30)
    except:
        pass
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 40, "AgroScan Disease Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Disease: {disease}\n\nTreatment Advice: {treatment}")
    pdf_output = io.BytesIO()
    pdf.output(pdf_output, 'S')
    pdf_output.seek(0)
    return pdf_output

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

    treatment = get_openai_treatment_advice(disease, follow_up)

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