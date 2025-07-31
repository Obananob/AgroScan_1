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
- Twilio integration for diagnosis and easy access

---
## 🖥️ How It Works

1. User uploads a plant leaf image via the Streamlit app
2. The model classifies the disease (or identifies healthy leaves)
3. The user receives instant diagnosis plus treatment advice
4. SMS is sent to the farmer (via Twilio)

---
## Leaf Diseases Trained On
- Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
- Corn_(maize)___Common_rust_
- Corn_(maize)___healthy
- Pepper,_bell___Bacterial_spot
- Pepper,_bell___healthy
- Potato___Late_blight
- Potato___healthy
- Tomato___Bacterial_spot
- Tomato___Early_blight
- Tomato___Late_blight
- Tomato___Leaf_Mold
- Tomato___Target_Spot
- Tomato___healthy
---
## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- Pandas, NumPy
- Twilio API (WhatsApp and SMS integration)
- FastAPI for hosting
- Streamlit
- Railway
- Github
- Postman

---
## Link to Prototype 
- [YouTube link](https://youtu.be/A26UjghBPvY?si=htANUpi3FVPVC9S-)
- [Streamlit URL](https://agroscan1-bytebuilders.streamlit.app)
- [FastAPI URL](https://agroscan1-production.up.railway.app)
---
## 📊 Dataset

The model is trained using the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease), a labeled image dataset of healthy and diseased crop leaves.


## 👨‍👩‍👧‍👦 Team – ByteBuilders

| Name                          | Role                       | Email                            | Phone         |
|-------------------------------|----------------------------|----------------------------------|---------------|
| **Oni-Bashir Atiatunnasir Arike** | Team Lead / ML Engineer     | obananob91@gmail.com             | 09059624948   |
| **Oladejo Luqman Kehinde**       | Backend Developer / DevOps | oladejoluqman6351@gmail.com      | 08163510869   |
| **Abdulrasaq Yusuf Alabi**       | Project Documentation      | hollarmheeitconsult@gmail.com    | 07039137111   |

## 📦 Installation

```bash
# Step 1: Clone the GitHub repository
git clone https://github.com/Obananob/AgroScan_1.git
cd AgroScan_1

# Step 2: Create and activate a virtual environment
python -m venv .venv
# For Linux/macOS
source .venv/bin/activate
# For Windows
.venv\Scripts\activate

# Step 3: Install required dependencies
pip install -r requirements.txt

# Step 4: Start the FastAPI backend
uvicorn main:app --reload
# The FastAPI server will start at http://127.0.0.1:8000
# Swagger UI is available at http://127.0.0.1:8000/docs

# Step 5: In a new terminal, launch the Streamlit frontend
streamlit run app.py
# This opens the AgroScan UI where users can upload an image,
# receive diagnosis, and get an SMS notification.
# To receive SMS notification you have to create a twilio account then copy your Twilio SID, AUTH TOKEN to your .env file
