# ── BirdSense Backend ── FastAPI + PyTorch (CPU) ──
FROM python:3.11-slim

WORKDIR /app

# System deps for librosa/soundfile and model download
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install (CPU-only PyTorch for smaller image)
COPY 08_Deployment/Backend/requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy backend source code (excluding large model files)
COPY 08_Deployment/Backend/config.py 08_Deployment/Backend/main.py 08_Deployment/Backend/db_utils.py ./
COPY 08_Deployment/Backend/routers ./routers
COPY 08_Deployment/Backend/services ./services

# Create required directories (models downloaded at runtime via start.sh)
RUN mkdir -p /app/uploads /app/models/checkpoints /app/labels

EXPOSE 8000

# Copy and run startup script
COPY 08_Deployment/Backend/start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
