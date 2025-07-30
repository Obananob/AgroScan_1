# 🌿 AgroScan –  Plant Disease Detection 

**AgroScan** is a deep learning-powered solution that detects crop leaf diseases from images. The model is designed to be deployed via a **WhatsApp chatbot** using Twilio, making it accessible to farmers and agricultural workers in remote or low-connectivity areas.

AgroScan is developed as part of the **Africa Deep Tech Challenge 2025**.

---

## 🎯 Project Goals

- Build an image classifier that detects plant diseases from leaf photos
- Deploy the model through a WhatsApp bot for easy farmer access
- Promote early detection and minimize crop loss in agriculture

---

## 🧠 Model Details

- Model: **Convolutional Neural Network (CNN)**
- Trained on: PlantVillage Dataset (Kaggle)
- Output: Disease classification (e.g., Early Blight, Late Blight, Healthy)
- Evaluation: Accuracy, Confusion Matrix, Precision/Recall
- Streamlit frontend for real-time image upload and predictio
- FastAPI backend to serve predictions and handle WhatsApp/SMS webhooks
- Twilio integration for diagnosis alerts and treatment advice

---
## 🖥️ How It Works

1. User uploads a plant leaf image via the Streamlit app
2. The model classifies the disease (or identifies healthy leaves)
3. The user receives instant diagnosis plus treatment advice
4. SMS is sent to the farmer (via Twilio)
   
## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- Pandas, NumPy
- Twilio API (WhatsApp and SMS integration)
- FastAPI for hosting
- Streamlit
- Railway
- Github

---

## 📊 Dataset

The model is trained using the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease), a labeled image dataset of healthy and diseased crop leaves.


## 👨‍👩‍👧‍👦 Team – ByteBuilders

| Name                          | Role                       | Email                            | Phone         |
|-------------------------------|----------------------------|----------------------------------|---------------|
| **Oni-Bashir Atiatunnasir Arike** | Team Lead / ML Engineer     | obananob91@gmail.com             | 09059624948   |
| **Oladejo Luqman Kehinde**       | Backend Developer / DevOps | oladejoluqman6351@gmail.com      | 08163510869   |
| **Abdulrasaq Yusuf Alabi**       | Frontend & QA Testing      | hollarmheeitconsult@gmail.com    | 07039137111   |

## 📦 Installation (Dev Mode)

```bash
git clone https://github.com/yourusername/agroscan.git
cd agroscan
pip install -r requirements.txt
Uvicorn main:app -- reload <--- fastAPI should be running to run streamlit app
streamlit run app.py
