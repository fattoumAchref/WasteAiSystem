# api_gateway.py

from fastapi import FastAPI, HTTPException, Request,File, UploadFile
from pydantic import BaseModel
import requests
import consul
import httpx
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

c = consul.Consul(host="localhost", port=8500)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

class PredictRequest(BaseModel):
    question: str

class PlanResponse(BaseModel):
    tasks: list
    tools: list

def discover_service(service_name: str):
    services = c.catalog.service(service_name)[1]
    if not services:
        raise Exception(f"Service '{service_name}' not found in Consul.")
    return services[0]['Address'], services[0]['ServicePort']

@app.post("/chat", response_model=ChatResponse)
def forward_chat(request: ChatRequest):
    try:
        host, port = discover_service("ragwaste-service")
        response = requests.post(f"http://{host}:{port}/chat", json=request.dict())
        return ChatResponse(**response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/proxy/classify")
async def proxy_classify(request: Request):
    try:
        # Forward the multipart request to the classification service
        form = await request.form()
        files = {'file': (form['file'].filename, await form['file'].read(), form['file'].content_type)}

        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:8002/classify", files=files)
            return JSONResponse(content=response.json(), status_code=response.status_code)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": "Error proxying request", "error": str(e)}
        )
    
@app.post("/predict")
async def proxy_droneClassify(file: UploadFile = File(...)):
    try:
        # Get actual image dimensions
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data))
        width, height = img.size
        
        # Reset file pointer for forwarding
        file.file.seek(0)
        
        host, port = discover_service("yolo-service")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"http://{host}:{port}/predict",
                files={"file": (file.filename, image_data, file.content_type)}
            )
            response.raise_for_status()
            
            return {
                "image_width": width,
                "image_height": height,
                "detections": response.json().get("detections", []),
                "error": None
            }

    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="YOLO service unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))