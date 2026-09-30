"""HTTP wrapper around DiagnosisPipeline, so the Spring Boot backend can call the model over the network
instead of shelling out to the CLI. The model loads once at process startup, not per-request.

Run:
  uvicorn api:app --app-dir scripts --host 0.0.0.0 --port 8000

Backend side: POST multipart/form-data to /diagnose with an `image` file part and a `crop` field
(one of pepper/strawberry/lettuce/cucumber/tomato); response body is the same JSON full_pipeline.py prints.

Auth: set the API_KEY environment variable to require it in an `X-API-Key` header on /diagnose. Do this
whenever the server is reachable from outside the local machine (e.g. through a tunnel); /health stays open.
"""
import io
import logging
import os
import secrets

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import Image

from diagnosis_pipeline import CROPS, DiagnosisPipeline

logger = logging.getLogger("uvicorn.error")

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    logger.warning("API_KEY is not set: /diagnose accepts unauthenticated requests. Do not expose this server.")


def require_api_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and not (x_api_key and secrets.compare_digest(x_api_key.encode(), API_KEY.encode())):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


app = FastAPI(title="smartfarm-ai diagnosis service")

pipeline = DiagnosisPipeline(
    detector_path=os.environ.get("DETECTOR_PATH", "models/rfdetr-s-synthetic-v5/last_ema.pth"),
    categories_path=os.environ.get("CATEGORIES_PATH", "models/rfdetr-s-synthetic-v5/categories.json"),
    knowledge_path=os.environ.get("KNOWLEDGE_PATH", "data/disease_knowledge.json"),
    severity_dir=os.environ.get("SEVERITY_DIR", "models/severity-v2"),
    resolution=int(os.environ.get("DETECTOR_RESOLUTION", "640")),
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ping", dependencies=[Depends(require_api_key)])
def ping():
    """Like /health but behind the API key, so a caller can confirm its key is accepted."""
    return {"status": "ok"}


@app.post("/diagnose", dependencies=[Depends(require_api_key)])
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
