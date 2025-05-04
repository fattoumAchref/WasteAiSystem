from fastapi import FastAPI, UploadFile, File
import uvicorn
import tensorflow as tf
import numpy as np
import consul
import socket
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input
import io
from PIL import Image

app = FastAPI()
service_name = "wasteclassification-service"
service_id = f"{service_name}-{socket.gethostname()}"
service_port = 8002  # Different port

c = consul.Consul(host="localhost", port=8500)

# Load your model
MODEL_PATH = 'efficient1waste15.keras'
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
CLASS_NAMES = ['aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 
               'cardboard_boxes', 'clothing', 'food_waste', 
               'glass_beverage_bottles', 'glass_cosmetic_containers', 
               'office_paper', 'paper_cups', 'plastic_detergent_bottles', 
               'plastic_shopping_bags', 'plastic_soda_bottles', 
               'plastic_straws', 'plastic_water_bottles']

@app.on_event("startup")
def register_service():
    c.agent.service.register(
        name=service_name,
        service_id=service_id,
        address=socket.gethostbyname(socket.gethostname()),
        port=service_port,
        tags=["classification", "tensorflow", "waste"]
    )
    print(f"✅ Registered {service_name} to Consul")

@app.on_event("shutdown")
def deregister_service():
    c.agent.service.deregister(service_id)
    print(f"❌ Deregistered {service_name} from Consul")

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    
    img_processed = preprocess_input(img_array.copy())
    img_ready = np.expand_dims(img_processed, axis=0)

    preds = model.predict(img_ready)[0]
    print("Predictions:", preds)  # Debug

    pred_class = CLASS_NAMES[np.argmax(preds)]
    confidence = float(np.max(preds))

    return {"class": pred_class, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run("wasteclassification_service:app", host="0.0.0.0", port=service_port, reload=True)
