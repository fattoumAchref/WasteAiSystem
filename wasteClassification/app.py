# app.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input
import io
from PIL import Image

app = FastAPI()

# 1. Load your model ONCE when starting the app
MODEL_PATH = 'efficient1waste15.keras'
CLASS_NAMES = [
    'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes', 'clothing', 'food_waste',
    'glass_beverage_bottles', 'glass_cosmetic_containers', 'office_paper', 'paper_cups',
    'plastic_detergent_bottles', 'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws', 'plastic_water_bottles'
]

# Load model (prefer CPU for safety)
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# 2. Image preparation function
def prepare_image(uploaded_file):
    img = Image.open(io.BytesIO(uploaded_file)).convert('RGB')
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_processed = preprocess_input(img_array.copy())
    return np.expand_dims(img_processed, axis=0)


# 3. Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})

    try:
        contents = await file.read()
        img_ready = prepare_image(contents)

        preds = model.predict(img_ready)[0]
        pred_class = CLASS_NAMES[np.argmax(preds)]
        confidence = float(np.max(preds))

        return {
            "prediction": pred_class,
            "confidence": f"{confidence:.2%}"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
