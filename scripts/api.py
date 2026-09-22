"""HTTP wrapper around DiagnosisPipeline, so the Spring Boot backend can call the model over the network
instead of shelling out to the CLI. The model loads once at process startup, not per-request.

Run:
  uvicorn api:app --app-dir scripts --host 0.0.0.0 --port 8000

Backend side: POST multipart/form-data to /diagnose with an `image` file part and a `crop` field
(one of pepper/strawberry/lettuce/cucumber/tomato); response body is the same JSON full_pipeline.py prints.
"""
import io
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from diagnosis_pipeline import CROPS, DiagnosisPipeline

app = FastAPI(title="smartfarm-ai diagnosis service")

pipeline = DiagnosisPipeline(
    detector_path=os.environ.get("DETECTOR_PATH", "models/rfdetr-s-synthetic-v5/last_ema.pth"),
    categories_path=os.environ.get("CATEGORIES_PATH", "models/rfdetr-s-synthetic-v5/categories.json"),
    knowledge_path=os.environ.get("KNOWLEDGE_PATH", "data/disease_knowledge.json"),
    severity_model_path=os.environ.get("SEVERITY_MODEL_PATH"),
    resolution=int(os.environ.get("DETECTOR_RESOLUTION", "640")),
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/diagnose")
async def diagnose(
    image: UploadFile = File(...),
    crop: str = Form(...),
    threshold: float = Form(0.15),
    tiles: int = Form(1),
):
    if crop not in CROPS:
        raise HTTPException(status_code=400, detail=f"unknown crop '{crop}', expected one of {CROPS}")

    raw = await image.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="uploaded file is not a readable image")

    return pipeline.diagnose(img, crop, threshold=threshold, tiles=tiles)
