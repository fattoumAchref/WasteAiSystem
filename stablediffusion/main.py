from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from generator import generate_description_with_mistral, generate_image
from io import BytesIO
import base64
import socket
import consul
from contextlib import asynccontextmanager

# Service Info
service_name = "recycled-image-service"
service_port = 8003
service_id = f"{service_name}-{socket.gethostname()}"
consul_client = consul.Consul(host="localhost", port=8500)



# Lifespan handler for registering/deregistering with Consul
@asynccontextmanager
async def lifespan(app: FastAPI):
    ip = socket.gethostbyname(socket.gethostname())
    consul_client.agent.service.register(
        name=service_name,
        service_id=service_id,
        address=ip,
        port=service_port,
        tags=["image", "recycling"],
        check={
            "http": f"http://{ip}:{service_port}/health",
            "interval": "10s"
        }
    )
    print(f"✅ Registered {service_name} to Consul")

    yield

    consul_client.agent.service.deregister(service_id)
    print(f"❌ Deregistered {service_name} from Consul")

# Create app with lifespan
app = FastAPI(title="Recycled Product Generator API", lifespan=lifespan)

# Health check route
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Request model
class MaterialRequest(BaseModel):
    pred_class: str

# Image generation endpoint
@app.post("/generate-image")
def generate_product_image(request: MaterialRequest):
    description = generate_description_with_mistral(request.pred_class)
    image = generate_image(description)

    img_io = BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)
    image_bytes = img_io.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return JSONResponse({
        "description": description,
        "image_base64": image_base64
    })
