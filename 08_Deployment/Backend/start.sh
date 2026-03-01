#!/bin/bash
set -e

echo "Starting BirdSense Backend..."
pip install --no-cache-dir -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
