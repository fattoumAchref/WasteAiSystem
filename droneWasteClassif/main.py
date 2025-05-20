
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io
import requests


app = FastAPI()
model = YOLO("droneBest.pt")

@app.on_event("startup")
def register_service_with_consul():
    consul_url = "http://localhost:8500/v1/agent/service/register"
    service_info = {
        "Name": "yolo-service",
        "ID": "yolo-service-1",
        "Address": "localhost",
        "Port": 8001,
        "Tags": ["yolo", "object-detection"],
        "Check": {
            "HTTP": "http://localhost:8001/docs",
            "Interval": "10s"
        }
    }

    try:
        response = requests.put(consul_url, json=service_info)
        print("✅ Service registered in Consul:", response.status_code)
    except Exception as e:
        print("❌ Failed to register service in Consul:", e)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    results = model(image)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        detections.append({
            "label": label,
            "confidence": conf,
            "bbox": xyxy
        })

    return {"detections": detections}
