from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Initialize the FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins. Replace "*" with specific origins if needed.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the model and class names
MODEL_PATH = r"C:/Temp_D/Progreming_Lerning/AI_ML/TensorFlowp/converted_keras/keras_model.h5"
LABELS_PATH = r"C:/Temp_D/Progreming_Lerning/AI_ML/TensorFlowp/converted_keras/labels.txt"

model = load_model(MODEL_PATH, compile=False)
class_names = [line.strip() for line in open(LABELS_PATH, "r").readlines()]

# Define the image size expected by the model
IMAGE_SIZE = (224, 224)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Load the uploaded image
        image = Image.open(file.file).convert("RGB")
        
        # Preprocess the image
        image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # Make prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = float(prediction[0][index])
        
        # Return prediction and confidence score
        return JSONResponse(content={
            "class_name": class_name[2:],  # Remove the leading index if present
            "confidence_score": confidence_score
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
