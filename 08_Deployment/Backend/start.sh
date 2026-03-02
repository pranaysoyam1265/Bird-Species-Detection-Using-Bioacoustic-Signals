#!/bin/bash
set -e

echo "Starting BirdSense Backend..."

# Download model if not present and MODEL_URL is set
if [ ! -f "./models/checkpoints/best_model_v3.pth" ]; then
  if [ -z "$MODEL_URL" ]; then
    echo "ERROR: Model file not found and MODEL_URL not set"
    echo "Provide MODEL_URL environment variable with download link"
    exit 1
  fi
  echo "Downloading model from $MODEL_URL..."
  mkdir -p ./models/checkpoints
  curl -sSL "$MODEL_URL" -o ./models/checkpoints/best_model_v3.pth
  if [ ! -f "./models/checkpoints/best_model_v3.pth" ]; then
    echo "ERROR: Failed to download model"
    exit 1
  fi
  echo "✓ Model downloaded successfully"
fi

echo "Starting FastAPI server on port ${PORT:-8000}..."
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
