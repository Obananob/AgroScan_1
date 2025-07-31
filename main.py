from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import Response
from fastapi.responses import PlainTextResponse, JSONResponse
from urllib.parse import parse_qs
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

# Whatsapp webhook endpoint
@app.post("/hook")
async def whatsapp_hook(request: Request):
    raw_body = await request.body()
    data = parse_qs(raw_body.decode())

    user_msg = data.get("Body", [""])[0].strip().lower()
    num_media = int(data.get("NumMedia", ["0"])[0])

    response = MessagingResponse()

    if num_media > 0:
        media_url = data.get("MediaUrl0", [""])[0]
        content_type = data

    response = MessagingResponse()

    if num_media > 0:
    media_url = data.get("MediaUrl0", [""])[0]
    content_type = data.get("MediaContentType0", [""])[0]

    if content_type and content_type.startswith("image/"):
        try:
            img_response = requests.get(media_url, auth=(TWILIO_SID, TWILIO_AUTH_TOKEN))
            if img_response.status_code != 200:
                reply = "❌ Couldn't download the image. Please try again."
            else:
                image = read_file_as_image(img_response.content)
                img_batch = preprocess_image(Image.fromarray(image))

                prediction = MODEL.predict(img_batch)
                predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
                confidence = np.max(prediction[0])

                if confidence < 0.7:
                    reply = "⚠️ Unable to confidently identify the disease. Please try a clearer image."
                else:
                    reply = f"🌿 Disease Detected: *{predicted_class}*\nConfidence: `{confidence:.2f}`"
        except Exception as e:
            reply = f"❌ Error processing image: {str(e)}"
    else:
        reply = "⚠️ Please send a valid image of a plant leaf."
else:
    if "hi" in user_msg or "hello" in user_msg:
        reply = "👋 Welcome to AgroScan! Send me a plant leaf image and I’ll tell you if it’s sick and what to do."
    else:
        reply = "📸 Please upload a clear plant leaf image for analysis."

    response.message(reply)
    return Response(content=str(response), media_type="application/xml")

@app.get("/")
def root():
    return {"message": "🌱 Welcome to AgroScan FastAPI backend!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
