from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import clip
from PIL import Image
from ultralytics import YOLO
import numpy as np
import io
import cv2
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import requests
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO

os.environ["TRANSFORMERS_NO_TF"] = "1"

app = FastAPI()

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======== Configuration GPU ========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True  # Optimisation pour les convolutions

# ======== Chemins des modèles ========
YOLO_MODEL_PATH = 'yolov8m2_taco.pt'

# ======== Chargement des modèles avec optimisation GPU ========

# YOLO (optimisé pour GPU)
try:
    model = YOLO(YOLO_MODEL_PATH).to(device)
    model.fuse()  # Fusion des couches pour meilleures performances
except Exception as e:
    raise RuntimeError(f"Erreur de chargement YOLO: {str(e)}")

# CLIP (optimisé pour GPU)
try:
    clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)
    clip_model.eval()
except Exception as e:
    raise RuntimeError(f"Erreur de chargement CLIP: {str(e)}")

# BLIP (version locale avec TF→PyTorch sur GPU)
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
    class_names, material_map, state_descriptions, contam_descriptions,
    material_density_map, state_volume_multipliers, class_volume_profiles,
    class_size_profiles, CALIBRATION_FACTOR, CAMERA_DISTANCE_METERS,
    FOCAL_LENGTH_MM, SENSOR_HEIGHT_MM,
    generate_caption, generate_material_prompts_with_hermes,
    generate_prompts, predict_clip_states, iou, is_close,
    estimate_volume_from_box, get_state_multiplier, estimate_volume,
    estimate_weight, process_image, generate_report_with_phi2
)

@app.post("/detect")
async def detect_waste(file: UploadFile = File(...)):
    try:
        # Lire l'image téléchargée
        img_bytes = await file.read()
        if len(img_bytes) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Image trop volumineuse")
            
        img = Image.open(BytesIO(img_bytes))
        
        # Sauvegarder temporairement l'image pour le traitement
        temp_path = "temp_upload.jpg"
        img.save(temp_path)
        
        # Traiter l'image avec le pipeline
        results = process_image(temp_path)
        
        # Nettoyer le fichier temporaire
        os.remove(temp_path)
        
        # Générer le rapport
        if "objects" in results:
            report = generate_report_with_phi2(results["objects"], llm)
            results["report"] = report
        
        # Nettoyage GPU
        torch.cuda.empty_cache()
        
        return results

    except Exception as e:
        torch.cuda.empty_cache()
        print(f"🚨 Erreur pendant le traitement : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5001) 