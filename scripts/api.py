"""HTTP wrapper around DiagnosisPipeline, so the Spring Boot backend can call the model over the network
instead of shelling out to the CLI. The model loads once at process startup, not per-request.

Run:
  uvicorn api:app --app-dir scripts --host 0.0.0.0 --port 8000

Backend side: POST multipart/form-data to /diagnose with an `image` file part and a `crop` field
(one of pepper/strawberry/lettuce/cucumber/tomato); response body is the same JSON full_pipeline.py prints.

Auth: set the API_KEY environment variable to require it in an `X-API-Key` header on /diagnose. Do this
whenever the server is reachable from outside the local machine (e.g. through a tunnel); /health stays open.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
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


_inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diagnosis")


@asynccontextmanager
async def lifespan(app):
    try:
        yield
    finally:
        _inference_executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="smartfarm-ai diagnosis service", lifespan=lifespan)

pipeline = DiagnosisPipeline(
    detector_path=os.environ.get("DETECTOR_PATH", "models/rfdetr-s-synthetic-v5/last_ema.pth"),
    categories_path=os.environ.get("CATEGORIES_PATH", "models/rfdetr-s-synthetic-v5/categories.json"),
    knowledge_path=os.environ.get("KNOWLEDGE_PATH", "data/disease_knowledge.json"),
    severity_dir=os.environ.get("SEVERITY_DIR", "models/severity-v2"),
    resolution=int(os.environ.get("DETECTOR_RESOLUTION", "640")),
)

# Stored by the backend in diagnosis.model_version (VARCHAR(30)), so results can be traced to the model that made
# them. Change it whenever DETECTOR_PATH or SEVERITY_DIR points at a different model.
MODEL_VERSION = os.environ.get("MODEL_VERSION", "rfdetr-v5_severity-v2")


def _diagnose_image(file, crop, threshold, tiles):
    img = None
    try:
        img = Image.open(file)
        img.load()  # Decode before inference so truncated uploads remain a 400.
    except Exception as error:
        if img is not None:
            img.close()
        raise HTTPException(status_code=400, detail="uploaded file is not a readable image") from error

    with img:
        return pipeline.diagnose(img, crop, threshold=threshold, tiles=tiles)


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
    threshold: float = Form(0.15, ge=0, le=1, allow_inf_nan=False),
    tiles: int = Form(1, ge=1),
):
    if crop not in CROPS:
        raise HTTPException(status_code=400, detail=f"unknown crop '{crop}', expected one of {CROPS}")

    # Keep the spooled upload while waiting; read/decode it only in the single
    # inference worker, without occupying the shared HTTP worker pool.
    result = await asyncio.get_running_loop().run_in_executor(
        _inference_executor, _diagnose_image, image.file, crop, threshold, tiles
    )
    return {**result, "model_version": MODEL_VERSION}
