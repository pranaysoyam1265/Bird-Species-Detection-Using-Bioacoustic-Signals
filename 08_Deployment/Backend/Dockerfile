# ── BirdSense Backend ── FastAPI + PyTorch (CPU) ──
FROM python:3.11-slim

WORKDIR /app

# System deps for librosa/soundfile and model download
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU packages first with special index
# Using --extra-index-url adds to PyPI, not replaces it
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0

# Copy and install remaining dependencies from PyPI
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (excluding large model files)
COPY config.py main.py db_utils.py ./
COPY routers ./routers
COPY services ./services
COPY models ./models
COPY labels ./labels

# Create required directories (models downloaded at runtime via start.sh)
RUN mkdir -p /app/uploads /app/models/checkpoints /app/labels

# ── Anti-Thrashing Optimization ──
# Render only provides a fraction of a CPU on the free tier, but PyTorch
# will attempt to spawn 32+ threads based on the host physical machine.
# This strictly limits PyTorch to 2 threads, preventing total server lockup.
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2

EXPOSE 8000

COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
