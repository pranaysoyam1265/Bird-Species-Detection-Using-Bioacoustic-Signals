# ── BirdSense Backend ── FastAPI + PyTorch ──
FROM python:3.11-slim

WORKDIR /app

# System deps for librosa/soundfile and model download
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy source code (excluding large model files)
COPY config.py main.py db_utils.py ./
COPY routers ./routers
COPY services ./services

# Create required directories (models downloaded at runtime via start.sh)
RUN mkdir -p /app/uploads /app/models/checkpoints /app/labels

EXPOSE 8000

COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
