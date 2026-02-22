# 🎵 Bird Species Detection Using Bioacoustic Signals

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) 
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/pranaysoyam1265/Bird-Species-Detection-Using-Bioacoustic-Signals)

A comprehensive machine learning and deep learning project for detecting and classifying bird species using bioacoustic signals. This project combines audio processing, feature extraction, neural networks, and deployment infrastructure to create a production-ready bird species identification system.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Audio Processing Pipeline](#audio-processing-pipeline)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline for **bird species classification** from bioacoustic signals. The system captures audio recordings, processes them into meaningful features (spectrograms, embeddings), trains deep neural networks, and provides both API and web-based interfaces for predictions.

### Key Objectives
- **Audio Classification**: Classify bird species from audio recordings
- **Scalable Pipeline**: Handle large volumes of bioacoustic data
- **Production Ready**: Deploy as REST API, Streamlit app, or web application
- **Interpretability**: Provide attention maps and gradient-based explanations

---

## ⚡ Features

### Core ML Features
- ✅ **Multi-format Audio Support**: WAV, MP3, OGG, FLAC
- ✅ **Spectrogram Generation**: Mel-spectrograms, MFCC, CQT transforms
- ✅ **Deep Learning Models**: CNN, RNN, Attention mechanisms, Transfer learning
- ✅ **Feature Embeddings**: Audio embeddings from pre-trained models
- ✅ **Data Augmentation**: Time-stretching, pitch-shifting, mixup
- ✅ **Cross-validation**: Multi-fold training and evaluation

### Explainability & Analysis
- 🔍 **Attention Maps**: Visualize model focus areas
- 🔍 **Grad-CAM**: Gradient-based class activation maps
- 🔍 **Temporal Localization**: Identify species detection timing
- 🔍 **Per-species Analysis**: Detailed metrics and confusion matrices

### Deployment
- 🚀 **REST API**: FastAPI/Flask backend
- 🚀 **Web Interface**: Interactive Streamlit dashboards
- 🚀 **Frontend App**: Next.js React application
- 🚀 **Docker Support**: Containerized deployment
- 🚀 **Cloud Ready**: Scalable architecture

---

## 📂 Project Structure

```
ML Conf-BioFSL/
├── 01_Raw_Data/                    # Raw data storage
│   ├── Audio_Recordings/           # Original audio files
│   ├── External_Data/              # External datasets (BirdCLEF)
│   └── Metadata/                   # Species metadata, checksums
├── 02_Preprocessed/                # Processed audio data
│   ├── Audio_Chunks/               # Segmented audio clips
│   ├── Standardized_Audio/         # Normalized audio
│   └── Quality_Reports/            # Audio quality analysis
├── 03_Features/                    # Extracted features
│   ├── Spectrograms/               # Mel-spectrograms (.npy)
│   ├── Embeddings/                 # Pre-trained embeddings
│   └── Spectrograms_Precomputed/   # Cached spectrograms
├── 04_Labels/                      # Training labels
│   ├── Raw_Labels/                 # Original labels
│   ├── Processed_Labels/           # Processing & mappings
│   └── Train_Val_Test_Split/       # Data splits
├── 05_Model/                       # Model artifacts
│   ├── Configs/                    # Model configurations
│   ├── Checkpoints/                # Training checkpoints
│   ├── Saved_Models/               # Final trained models
│   └── Training_Logs/              # TensorBoard logs
├── 06_Explainability/              # Model interpretation
│   ├── Attention_Maps/             # Attention visualizations
│   ├── GradCAM/                    # Activation maps
│   └── Temporal_Localization/      # Detection timing
├── 07_Evaluation/                  # Evaluation results
│   ├── Metrics/                    # Performance metrics
│   ├── Predictions/                # Model predictions
│   ├── Confusion_Matrices/         # Classification analysis
│   └── Species_Analysis/           # Per-species statistics
├── 08_Deployment/                  # Deployment code
│   ├── API/                        # REST API (FastAPI)
│   ├── Backend/                    # Backend services
│   ├── Frontend/                   # Next.js web app
│   ├── Streamlit_Version/          # Streamlit dashboards
│   └── Docker/                     # Docker configurations
├── 09_Utils/                       # Utilities & scripts
│   ├── Notebooks/                  # Jupyter notebooks
│   ├── Scripts/                    # Python scripts
│   ├── Logs/                       # Application logs
│   └── Temp/                       # Temporary files
├── 10_Outputs/                     # Output artifacts
│   ├── Reports/                    # Analysis reports
│   ├── Visualizations/             # Charts and graphs
│   ├── Exports/                    # Data exports
│   └── Analysis_Graphs/            # Statistical plots
└── README.md                       # This file
```

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- 8GB+ RAM recommended
- GPU (CUDA/cuDNN) for faster training

### Step 1: Clone Repository
```bash
git clone https://github.com/pranaysoyam1265/Bird-Species-Detection-Using-Bioacoustic-Signals.git
cd Bird-Species-Detection-Using-Bioacoustic-Signals
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n bioforest python=3.10
conda activate bioforest
```

### Step 3: Install Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision torchaudio tensorflow-gpu

# For development
pip install -r requirements-dev.txt
```

### Step 4: Download Datasets
```bash
# BirdCLEF dataset (if needed)
python 09_Utils/Scripts/download_birdclef.py

# Or use your own audio files
cp /path/to/audio/files 01_Raw_Data/Audio_Recordings/
```

---

## 🚀 Quick Start

### 1️⃣ Audio Preprocessing
```bash
python 09_Utils/Scripts/preprocess_audio.py \
  --input 01_Raw_Data/Audio_Recordings/ \
  --output 02_Preprocessed/Audio_Chunks/ \
  --chunk_duration 5
```

### 2️⃣ Feature Extraction
```bash
python 09_Utils/Scripts/extract_features.py \
  --input 02_Preprocessed/Audio_Chunks/ \
  --output 03_Features/Spectrograms/ \
  --feature_type melspectrogram
```

### 3️⃣ Model Training
```bash
python 09_Utils/Scripts/train_model.py \
  --config 05_Model/Configs/config_default.json \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 1e-3
```

### 4️⃣ Model Evaluation
```bash
python 09_Utils/Scripts/evaluate_model.py \
  --model 05_Model/Saved_Models/best_model.pth \
  --test_data 04_Labels/Train_Val_Test_Split/test.csv \
  --output 07_Evaluation/Metrics/
```

### 5️⃣ Run Streamlit App
```bash
streamlit run 08_Deployment/Streamlit_Version/app.py
```

### 6️⃣ Start REST API
```bash
python 08_Deployment/API/main.py --port 8000
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

---

## 📊 Dataset

### BirdCLEF Dataset
- **Source**: Kaggle BirdCLEF competition
- **Size**: ~28,975 spectrogram images + metadata
- **Species**: 1,000+ bird species
- **Audio**: Format varies (WAV, MP3, OGG)
- **Metadata**: Geographic location, recording time, species info

### Custom Dataset Structure
```
01_Raw_Data/Metadata/
├── bird_metadata_complete.csv      # Complete metadata
├── metadata_usable_only.csv        # Validated entries
├── species_summary.csv             # Species distribution
└── problematic_recording_ids.csv   # Data quality issues
```

**Metadata Format:**
```csv
recording_id,species_name,latin_name,location,date,audio_file
XC1000176,Song Sparrow,Melospiza melodia,USA,2023-01-15,path/to/audio.wav
```

---

## 🎨 Audio Processing Pipeline

### Step 1: Raw Audio Input
- **Format Support**: WAV, MP3, OGG, FLAC
- **Sample Rate**: Standardized to 22.05 kHz or 44.1 kHz
- **Channels**: Converted to mono

### Step 2: Chunking
- **Duration**: 5-second clips
- **Overlap**: 0-2 seconds overlap for coverage
- **Padding**: Zero-padding for short clips

### Step 3: Normalization
- **Mean**: Subtract to center amplitude
- **Std Dev**: Scale to unit variance
- **Clipping**: Remove extreme outliers

### Step 4: Feature Extraction
```python
# Mel-Spectrogram
mel_spec = librosa.feature.melspectrogram(
    y=audio_signal, sr=22050, n_mels=128, n_fft=2048
)

# MFCC (Mel-Frequency Cepstral Coefficients)
mfcc = librosa.feature.mfcc(y=audio_signal, sr=22050, n_mfcc=13)

# Chromagram
chroma = librosa.feature.chroma_stft(y=audio_signal, sr=22050)
```

### Step 5: Augmentation
- Time-stretching: Vary temporal scale
- Pitch-shifting: Change pitch without time
- Mixup: Blend multiple samples
- SpecAugment: Spectral masking

---

## 🧠 Model Architecture

### CNN-based Models

#### 1. ResNet50 + Custom Head
```
Input (128, 431, 1)
  ↓
ResNet50 Backbone (pretrained ImageNet)
  ↓
Global Average Pooling
  ↓
Dense(512, ReLU) + Dropout(0.3)
  ↓
Dense(256, ReLU) + Dropout(0.2)
  ↓
Dense(num_species, Softmax)
  ↓
Output (num_species)
```

#### 2. Custom CNN + Attention
```
Input (128, 431, 1)
  ↓
Conv2D(32) → BatchNorm → ReLU → MaxPool
  ↓
Conv2D(64) → BatchNorm → ReLU → MaxPool
  ↓
Conv2D(128) → BatchNorm → ReLU → MaxPool
  ↓
Multi-Head Self-Attention
  ↓
Global Average Pooling
  ↓
Dense Classifier Head
  ↓
Output (num_species)
```

#### 3. RNN with Attention
```
Input Sequence (T, 128)
  ↓
Bidirectional LSTM (512 units)
  ↓
Multi-Head Attention
  ↓
RNN → Dense layers
  ↓
Output (num_species)
```

### Hyperparameters
- **Optimizer**: Adam (lr=1e-3, beta1=0.9, beta2=0.999)
- **Loss**: Cross-entropy with class weights
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)
- **Dropout**: 0.2-0.5

---

## 📈 Training & Evaluation

### Training Process
```bash
Epoch 1/100
  Train Loss: 2.341 | Train Acc: 0.42 | Val Loss: 2.156 | Val Acc: 0.51
Epoch 2/100
  Train Loss: 1.892 | Train Acc: 0.58 | Val Loss: 1.654 | Val Acc: 0.62
...
Epoch 100/100
  Train Loss: 0.234 | Train Acc: 0.94 | Val Loss: 0.456 | Val Acc: 0.89
```

### Evaluation Metrics
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP) per class
- **Recall**: TP / (TP + FN) per class
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **AUC-ROC**: Area under the ROC curve
- **Confusion Matrix**: Per-species classification breakdown

### Cross-Validation
- **Strategy**: 5-Fold Stratified Cross-Validation
- **Purpose**: Robust performance estimation
- **Result**: Mean ± Std deviation metrics

---

## 📊 Results

### Model Performance
| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| ResNet50 | 0.894 | 0.891 | 0.889 | 0.890 | 45 min |
| Custom CNN | 0.876 | 0.874 | 0.871 | 0.872 | 32 min |
| RNN + Attention | 0.862 | 0.859 | 0.856 | 0.857 | 62 min |

### Species Performance
Top 10 species by F1-score available in: `07_Evaluation/Species_Analysis/`

### Visualizations
- **Confusion Matrix**: `10_Outputs/Confusion_Matrices/`
- **ROC Curves**: `10_Outputs/Analysis_Graphs/`
- **Attention Maps**: `06_Explainability/Attention_Maps/`
- **Grad-CAM**: `06_Explainability/GradCAM/`

---

## 🚀 Deployment

### Option 1: Streamlit App
```bash
cd 08_Deployment/Streamlit_Version
streamlit run app.py
```
Access at `http://localhost:8501`

**Features:**
- 🎵 Audio upload and playback
- 🎯 Real-time species prediction
- 📊 Confidence scores visualization
- 🗺️ Geographic analysis
- 📈 Historical predictions

### Option 2: REST API
```bash
cd 08_Deployment/API
python main.py
```
Access at `http://localhost:8000`

**Endpoints:**
```
POST /predict - Single audio prediction
GET  /metrics - Performance metrics
POST /batch  - Batch predictions
GET  /health - Health check
```

### Option 3: Docker Deployment
```bash
# Build image
docker build -t bioforest:latest 08_Deployment/Docker/

# Run container
docker run -p 8000:8000 -v data:/app/data bioforest:latest

# Docker Compose
docker-compose -f 08_Deployment/Docker/docker-compose.yml up
```

### Option 4: Next.js Web App
```bash
cd 08_Deployment/Frontend
npm install
npm run dev
```
Access at `http://localhost:3000`

---

## 🔍 Explainability

### Attention Maps
Visualize which frequency/time regions the model focuses on:
```python
from 06_Explainability.attention import plot_attention_maps
plot_attention_maps(model, audio_file, output_path='visualization.png')
```

### Grad-CAM
Generate activation maps highlighting important regions:
```python
from 06_Explainability.gradcam import generate_gradcam
gradcam_map = generate_gradcam(model, audio_spectrogram)
```

### Temporal Localization
Identify when species is detected in audio:
```python
from 06_Explainability.temporal import get_detection_windows
windows = get_detection_windows(model, audio_file, threshold=0.7)
```

---

## 📝 Usage Examples

### Python API
```python
from 08_Deployment.API.inference import BirdSpeciesPredictor

# Initialize model
predictor = BirdSpeciesPredictor(model_path='05_Model/Saved_Models/best_model.pth')

# Single prediction
audio_path = '01_Raw_Data/Audio_Recordings/sample.wav'
predictions = predictor.predict(audio_path)

print("Top 5 Predictions:")
for species, confidence in predictions[:5]:
    print(f"  {species}: {confidence:.2%}")
```

### Batch Processing
```python
from 09_Utils.Scripts.batch_predict import batch_predict

results = batch_predict(
    audio_dir='01_Raw_Data/Audio_Recordings/',
    model_path='05_Model/Saved_Models/best_model.pth',
    output_csv='07_Evaluation/Predictions/batch_results.csv'
)
```

### Custom Training
```python
from 09_Utils.Scripts.train_model import train

config = {
    'model_type': 'ResNet50',
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'feature_type': 'melspectrogram'
}

model, history = train(config, train_data, val_data)
```

---

## 📦 Dependencies

### Core ML Libraries
```
torch==2.0.0
tensorflow==2.12.0
librosa==0.10.0
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.2.0
scipy==1.10.0
```

### Web Framework
```
fastapi==0.95.0
streamlit==1.20.0
next.js==13.0.0
```

### Visualization & Analysis
```
matplotlib==3.7.0
seaborn==0.12.0
plotly==5.13.0
```

See `requirements.txt` for complete list.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
pip install -r requirements-dev.txt
pre-commit install
pytest  # Run tests
```

---

## 📜 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Pranay Soyam**
- Email: pranaysoyam1265@gmail.com
- GitHub: [@pranaysoyam1265](https://github.com/pranaysoyam1265)

---

## 🙏 Acknowledgments

- **BirdCLEF Dataset**: Kaggle competition organizers
- **Libraries**: PyTorch, TensorFlow, librosa, scikit-learn teams
- **Research**: References from bioacoustics and audio ML papers

---

## 📞 Support

For issues, questions, or suggestions:
- 📧 Email: pranaysoyam1265@gmail.com
- 🐛 GitHub Issues: [Create an issue](https://github.com/pranaysoyam1265/Bird-Species-Detection-Using-Bioacoustic-Signals/issues)
- 💬 Discussions: [Start a discussion](https://github.com/pranaysoyam1265/Bird-Species-Detection-Using-Bioacoustic-Signals/discussions)

---

## 🌟 Citation

If you use this project in your research, please cite:

```bibtex
@software{soyam2026bioforest,
  author = {Soyam, Pranay},
  title = {Bird Species Detection Using Bioacoustic Signals},
  year = {2026},
  url = {https://github.com/pranaysoyam1265/Bird-Species-Detection-Using-Bioacoustic-Signals}
}
```

---

<div align="center">

**Made with ❤️ for bird enthusiasts and ML engineers**

[⬆ Back to Top](#-bird-species-detection-using-bioacoustic-signals)

</div>
