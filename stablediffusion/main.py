from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO
from generator import generate_description_with_mistral, generate_image
from fastapi.responses import JSONResponse, Response
import base64

app = FastAPI(title="Recycled Product Generator API")

class MaterialRequest(BaseModel):
    pred_class: str

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
