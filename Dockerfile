# Dockerfile (repo root)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- System packages ---
# - tesseract-ocr (+ eng data) for OCR
# - libgl1, libglib2.0-0 (OpenCV runtime)
# - libsm6, libxext6, libxrender1 (some OpenCV builds want these)
# - build-essential (gcc, g++) in case a wheel falls back to source build)
# - curl, git are just handy during pip resolves
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    build-essential curl git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps ---
COPY backend/requirements.txt .

# Make sure we have the latest build tooling so wheels resolve correctly
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# --- App code ---
COPY backend/ ./

# Render provides $PORT; gunicorn will bind to it.
ENV PORT=8080
CMD ["bash","-lc","gunicorn app:app --bind 0.0.0.0:${PORT}"]
