# ── BirdSense Backend ── FastAPI + PyTorch (CPU) ──
FROM python:3.11-slim

WORKDIR /app

# System deps for librosa/soundfile and model download
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install (CPU-only PyTorch for smaller image)
COPY 08_Deployment/Backend/requirements.txt .

# Install PyTorch CPU packages first with special index
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0

# Install remaining dependencies from PyPI
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code and all necessary packages
COPY 08_Deployment/Backend/config.py 08_Deployment/Backend/main.py 08_Deployment/Backend/db_utils.py ./
COPY 08_Deployment/Backend/routers ./routers
COPY 08_Deployment/Backend/services ./services
COPY 08_Deployment/Backend/models ./models
COPY 08_Deployment/Backend/labels ./labels

# Create required directories
RUN mkdir -p /app/uploads /app/models/checkpoints /app/labels

# ── Anti-Thrashing Optimization ──
# Restrict PyTorch to 2 threads to prevent massive CPU deadlocks on HF free tier
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2

EXPOSE 8000

# Copy and run startup script
COPY 08_Deployment/Backend/start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
