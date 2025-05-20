from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import clip
from ultralytics import YOLO
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import requests
from fastapi.middleware.cors import CORSMiddleware
import base64 
from fastapi import Request 
import logging
import tempfile
import json

os.environ["TRANSFORMERS_NO_TF"] = "1"

app = FastAPI()

# Add this at the VERY TOP of your middleware chain
@app.middleware("http")
async def add_cors_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:4200"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/detect")
async def options_detect():
    return {"message": "OK"}, 200, {
        "Access-Control-Allow-Origin": "http://localhost:4200",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }

# ======== Enregistrement dans Consul =========
@app.on_event("startup")
def register_service_with_consul():
    consul_url = "http://localhost:8500/v1/agent/service/register"
    service_info = {
        "Name": "ai-waste-detection-service",
        "ID": "ai-waste-detection-service-001",
        "Address": "localhost",
        "Port": 5001,
        "Tags": ["ai", "waste-detection", "yolo", "image-analysis"],
        "Check": {
            "HTTP": "http://localhost:5001/docs",
            "Interval": "10s",
            "Timeout": "5s"
        }
    }

    try:
        response = requests.put(consul_url, json=service_info)
        print("✅ Service registered in Consul:", response.status_code)
    except Exception as e:
        print("❌ Failed to register service in Consul:", e)

# ======== Configuration GPU ========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True  # Optimisation pour les convolutions

# ======== Chemins des modèles ========
YOLO_MODEL_PATH = 'yolov8m2_taco.pt'

# ======== Chargement des modèles avec optimisation GPU ========
# YOLO
try:
    model = YOLO(YOLO_MODEL_PATH).to(device)
    model.fuse()  # Fusion des couches pour meilleures performances
except Exception as e:
    raise RuntimeError(f"Erreur de chargement YOLO: {str(e)}")

# CLIP
try:
    clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)
    clip_model.eval()
except Exception as e:
    raise RuntimeError(f"Erreur de chargement CLIP: {str(e)}")

# BLIP
try:
    processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model_blip = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    model_blip.eval()
except Exception as e:
    raise RuntimeError(f"Erreur de chargement BLIP: {str(e)}")

# LLM (Phi-1.5 optimisé pour GPU)
try:
    model_id = "microsoft/phi-1_5"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_phi = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model_phi.eval()
    
    llm = pipeline(
        "text-generation",
        model=model_phi,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto"
    )
except Exception as e:
    raise RuntimeError(f"Erreur de chargement Phi-1.5: {str(e)}")

# Import des constantes et fonctions du pipeline
from pipeline_nour_v1000 import (
 process_image, generate_report_with_phi2
)


# Configurer le logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    return obj

@app.post("/detect")
async def detect_waste(file: UploadFile = File(...)):
    temp_path = None
    try:
        logger.info("🔴 Démarrage du traitement de l'image")
        
        # Validate file size
        img_bytes = await file.read()
        if len(img_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image trop volumineuse")

        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(img_bytes)
            temp_path = temp_file.name

        logger.info("🔵 Traitement de l'image...")
        summary = process_image(temp_path)
        logger.info("🔴 Fin de Traitement de l'image...")

        # Generate report
        logger.info("🔵 Generation du report avec phi2...")
        report = generate_report_with_phi2(summary["objects"], phi2_pipeline=llm)
        logger.info("🔴 Fin de generation du report avec phi2")

        response_data = {
            "detections": [convert_numpy_types(obj) for obj in summary["objects"]],
            "report": report,
            "original_image": base64.b64encode(img_bytes).decode('utf-8') if img_bytes else None
        }

        # Double-check conversion
        json.dumps(response_data)

        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={"Access-Control-Allow-Origin": "http://localhost:4200"}
        )
    finally:
        torch.cuda.empty_cache()
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.error(f"⚠️ Erreur nettoyage temporaire: {e}")

from fastapi import Request

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"📍 Erreur survenue: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": f"Une erreur est survenue: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5001) 