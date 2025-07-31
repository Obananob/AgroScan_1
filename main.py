from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

app = FastAPI(title="AgroScan – AI Powered Plant Doctor")

# MODEL LOAD 
MODEL = tf.keras.models.load_model("agroscan_model.keras")
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

# TWILIO SETUP
load_dotenv()
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

#  HUGGING FACE LLM 
# HF_TOKEN = os.getenv("HF_TOKEN")
# HF_MODEL = "featherless-ai/LawToken-0.3B-v2"  # use a light model <1B
# hf_client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)

# UTILS 
def preprocess_image(image: Image.Image):
    image = image.resize((160, 160))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# def generate_prompt(disease: str) -> str:
#     return (
#         f"Think carefully. A plant has been diagnosed with the disease: {disease}. "
#         f"What is a simple, actionable, and locally relevant treatment recommendation? "
#         f"Respond in the same language as the disease name."
#     )

# # def get_treatment_recommendation(disease: str) -> str:
# #     prompt = generate_prompt(disease)
# #     try:
# #         response = hf_client.text_generation(prompt=prompt, max_new_tokens=150, temperature=0.6)
# #         return response.strip()
# #     except Exception as e:
# #         return f"⚠️ LLM Error: {str(e)}"

# ENDPOINTS 

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    if confidence < 0.7:
        return {"class": "Uncertain", "confidence": float(confidence)}

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

# @app.post("/treat")
# async def treat(file: UploadFile = File(...)):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)

#     prediction = MODEL.predict(img_batch)
#     predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
#     confidence = np.max(prediction[0])

#     if confidence < 0.7:
#         return {
#             "class": "Uncertain",
#             "confidence": float(confidence),
#            # "treatment": "⚠️ Please upload a clearer image."
#         }

#    treatment = get_treatment_recommendation(predicted_class)

    return {
        "class": predicted_class,
        "confidence": float(confidence),
       # "treatment": treatment
    }

@app.post("/hook", response_class=PlainTextResponse)
async def whatsapp_hook(request: Request):
    data = await request.form()
    user_msg = data.get("Body", "").strip().lower()

    if "hi" in user_msg or "hello" in user_msg:
        reply = "👋 Welcome to AgroScan! Send me a plant leaf image and I’ll tell you if it’s sick and what to do."
    else:
        reply = "📸 Please upload a clear plant leaf image for analysis."

    response = MessagingResponse()
    response.message(reply)
    return PlainTextResponse(str(response))

@app.get("/")
def root():
    return {"message": "🌱 Welcome to AgroScan FastAPI backend!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
