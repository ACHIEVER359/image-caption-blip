# Image Caption Studio

A standalone web app based on the uploaded Google Colab notebook. It uses **Salesforce BLIP (`Salesforce/blip-image-captioning-base`)** to generate captions for uploaded images, supports an optional starting prompt, and processes multiple images in one batch.

## Features
- Responsive, custom HTML/CSS/JavaScript UI (no Gradio dependency)
- Multi-image upload, drag-and-drop, previews, per-image captions, copy-one and copy-all
- Optional prompt conditioning (examples: `a photo of`, `this image shows`, `there are`)
- FastAPI backend; model inference runs on the server
- Health endpoint, upload limits, Dockerfile, Render configuration, and GitHub-safe `.gitignore`
- Images are held in memory for processing and aren't intentionally saved to disk

## Run locally

Requires Python 3.10 or 3.11 and enough RAM for PyTorch and the BLIP model. First-time startup/request downloads the model weights from Hugging Face and needs an internet connection.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8000`.

## Deploy to Render

1. Create a new **public GitHub repository** and upload the contents of this folder (not the ZIP itself).
2. In Render, create a Web Service and connect that repository.
3. Choose Python runtime; build command: `pip install -r requirements.txt`; start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`.
4. Choose a plan with sufficient memory. BLIP + PyTorch can require several GB of RAM; small/free instances may run out of memory, and CPU captioning can be slow. The model must download on the first request, so allow for cold-start time.
5. After deployment, check `/api/health` and then upload a test image.

`render.yaml` is included as a deployment starting point; its `standard` plan may incur charges. Review current platform pricing before deploying.

## Docker

```bash
docker build -t image-caption-studio .
docker run --rm -p 8000:8000 image-caption-studio
```

Then open `http://localhost:8000`.

## Configuration

Optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_ID` | `Salesforce/blip-image-captioning-base` | Hugging Face model ID |
| `PORT` | `8000` locally | HTTP port |
| `MAX_IMAGE_MB` | `12` | Per-image upload limit |
| `MAX_IMAGES` | `8` | Batch size limit |
| `FORCE_CPU` | `0` | Set to `1` to force CPU inference |

No API key is required for the public model. Uploaded images are sent to your own deployed server for inference; don't upload sensitive images to a public instance you don't control.

## API

- `GET /api/health` — server/model status
- `POST /api/caption` — multipart form fields `files` (one or more image files) and optional `prompt`

## Model notes

This app follows the notebook's BLIP image-captioning model and prompt-based generation. Captions can be imperfect; inspect results before relying on them. The notebook's local Colab example images aren't bundled because they were external runtime files, not embedded in the notebook.
