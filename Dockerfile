# Dockerfile (repo root)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- System packages ----
# - tesseract-ocr (+ eng language)
# - libgl1 & libglib2.0-0 are needed by opencv-python-headless at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn

# ---- App code ----
COPY backend/ ./

# Render provides $PORT. Bind to it.
ENV PORT=8080
CMD ["bash","-lc","gunicorn app:app --bind 0.0.0.0:${PORT}"]
