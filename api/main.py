from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
from io import BytesIO
import tensorflow as tf
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:3001",  # Allow your React app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading the model
MODEL = tf.keras.models.load_model(r"C:\Users\safac\potato-disease\saved_model\model_version_1\model_version_1.keras")
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@app.get("/ping")
async def ping():
    return {"message": "Hello, I'm alive"}

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((224, 224))  # Resize image if the model expects a specific input size
    image = np.array(image) / 255.0  # Normalize pixel values if required by the model
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())  # Use 'await' because file.read() is async

    # The model expects a batch of images, so we expand dims
    img_batch_np = np.expand_dims(image, axis=0)

    # Get predictions from the model
    prediction = MODEL.predict(img_batch_np)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
