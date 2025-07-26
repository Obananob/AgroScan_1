import streamlit as st
import requests
from PIL import Image
import base64

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AgroScan – Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# -------------------- TITLE --------------------
st.title("🌿 AgroScan – Smart Plant Doctor")

# -------------------- LANGUAGE TOGGLE --------------------
language = st.selectbox("🌍 Select Language", ["English", "Yoruba", "Hausa", "Pidgin"])

# -------------------- FILE UPLOADER --------------------
uploaded_file = st.file_uploader("📷 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing image..."):
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post("http://localhost:8000/predict", files=files)
            except Exception as e:
                st.error(f"Server connection failed: {e}")
                st.stop()

        if response.status_code == 200:
            result = response.json()
            disease = result['class']
            confidence = result['confidence']
            st.success(f"✅ Disease: **{disease}**")
            st.info(f"🔎 Confidence: **{confidence*100:.2f}%**")

            # -------------------- TREATMENT ADVICE --------------------
            with st.spinner("Fetching treatment recommendation..."):
                advice_payload = {"disease": disease, "language": language}
                advice_response = requests.post("http://localhost:8000/treatment", json=advice_payload)

            if advice_response.status_code == 200:
                advice = advice_response.json()["advice"]
                st.markdown(f"💊 **Treatment Advice:**\n\n{advice}")
            else:
                st.warning("⚠️ Could not fetch treatment advice.")

            # -------------------- FOLLOW-UP QUESTION --------------------
            with st.expander("💬 Ask a follow-up question"):
                user_question = st.text_input("🧠 Ask the plant doctor:")
                if st.button("Submit Question"):
                    follow_up_payload = {"question": user_question, "disease": disease}
                    follow_up_response = requests.post("http://localhost:8000/ask", json=follow_up_payload)

                    if follow_up_response.status_code == 200:
                        reply = follow_up_response.json()["response"]
                        st.markdown(f"🤖 **AgroScan says:** {reply}")
                    else:
                        st.error("❌ Follow-up failed. Try again later.")

            # -------------------- PDF DOWNLOAD --------------------
            with st.spinner("Preparing your report..."):
                pdf_payload = {
                    "disease": disease,
                    "confidence": confidence,
                    "advice": advice,
                    "language": language,
                    "image_bytes": base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
                }
                pdf_response = requests.post("http://localhost:8000/generate-pdf", json=pdf_payload)

            if pdf_response.status_code == 200:
                st.download_button(
                    "📥 Download Report",
                    pdf_response.content,
                    file_name="AgroScan_Report.pdf"
                )
            else:
                st.error("📄 Failed to generate PDF.")

            # -------------------- WHATSAPP SHARE --------------------
            whatsapp_text = f"AgroScan detected *{disease}* with {confidence*100:.2f}% confidence.\nAdvice: {advice}"
            whatsapp_url = f"https://wa.me/?text={whatsapp_text.replace(' ', '%20')}"
            st.markdown(f"[📤 Share on WhatsApp]({whatsapp_url})")

        else:
            st.error("❌ Prediction failed. Ensure your FastAPI server is running.")
