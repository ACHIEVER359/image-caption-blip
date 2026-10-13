"""Standalone BLIP image-captioning web application."""
import io
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_ID = os.getenv("MODEL_ID", "Salesforce/blip-image-captioning-base")
DEVICE = "cuda" if torch.cuda.is_available() and os.getenv("FORCE_CPU", "0") != "1" else "cpu"
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_MB", "12")) * 1024 * 1024
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "8"))
processor = None
model = None
model_error: Optional[str] = None
model_lock = threading.Lock()


def load_model():
    global processor, model, model_error
    with model_lock:
        if model is not None:
            return
        try:
            processor = BlipProcessor.from_pretrained(MODEL_ID)
            model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
            model.to(DEVICE)
            model.eval()
            model_error = None
        except Exception as exc:
            model_error = str(exc)
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load lazily on the first request so the web server can start while the
    # model weights are downloading, and health checks can still respond.
    yield

app = FastAPI(title="Image Caption Studio", description="Generate image captions with Salesforce BLIP.", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_id": MODEL_ID, "device": DEVICE, "error": model_error}

@app.post("/api/caption")
async def caption_images(files: list[UploadFile] = File(...), prompt: str = Form(default="")):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image.")
    if len(files) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"Please upload no more than {MAX_IMAGES} images at once.")
    try:
        load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"The captioning model could not load: {exc}") from exc

    results = []
    for upload in files:
        raw = await upload.read()
        name = upload.filename or "image"
        if len(raw) > MAX_IMAGE_BYTES:
            results.append({"filename": name, "error": f"Image exceeds the {MAX_IMAGE_BYTES // (1024*1024)} MB limit."})
            continue
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            kwargs = {"images": image, "return_tensors": "pt"}
            if prompt.strip():
                kwargs["text"] = prompt.strip()
            inputs = processor(**kwargs).to(DEVICE)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=60)
            caption = processor.decode(output[0], skip_special_tokens=True).strip()
            results.append({"filename": name, "caption": caption, "width": image.width, "height": image.height,
                            "image_data": "data:image/jpeg;base64," + __import__("base64").b64encode(_jpeg_bytes(image)).decode("ascii")})
        except (UnidentifiedImageError, OSError):
            results.append({"filename": name, "error": "This file could not be read as an image. Try JPG, PNG, or WEBP."})
        except Exception as exc:
            results.append({"filename": name, "error": f"Caption generation failed: {exc}"})
    return {"results": results, "model": MODEL_ID, "device": DEVICE}


def _jpeg_bytes(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=88, optimize=True)
    return out.getvalue()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
