# 🐦 Bird Detection Project - Folder Structure

## Project: Confidence-Aware, Explainable Multi-Species Bird Detection

**Created:** 2026-02-06 16:41:28

---

## 📁 Directory Tree

```
ML Conf-BioFSL/
│
├── 01_Raw_Data/
│   ├── Audio_Recordings/
│   ├── Metadata/
│   └── External_Data/
│
├── 02_Preprocessed/
│   ├── Standardized_Audio/
│   ├── Audio_Chunks/
│   └── Quality_Reports/
│
├── 03_Features/
│   ├── Spectrograms/
│   └── Embeddings/
│
├── 04_Labels/
│   ├── Raw_Labels/
│   ├── Processed_Labels/
│   └── Train_Val_Test_Split/
│
├── 05_Model/
│   ├── Checkpoints/
│   ├── Saved_Models/
│   ├── Training_Logs/
│   └── Configs/
│
├── 06_Explainability/
│   ├── GradCAM/
│   ├── Attention_Maps/
│   └── Temporal_Localization/
│
├── 07_Evaluation/
│   ├── Metrics/
│   ├── Confusion_Matrices/
│   └── Predictions/
│
├── 08_Deployment/
│   ├── API/
│   ├── Frontend/
│   └── Docker/
│
├── 09_Utils/
│   ├── Scripts/
│   ├── Notebooks/
│   ├── Logs/
│   └── Temp/
│
└── 10_Outputs/
    ├── Reports/
    ├── Visualizations/
    └── Exports/
```

---

## 📋 Folder Descriptions

### 📂 01_Raw_Data

| Subfolder | Purpose |
|-----------|----------|
| `Audio_Recordings/` | Original 4521 WAV files from Xeno-Canto |
| `Metadata/` | CSV files with recording metadata |
| `External_Data/` | External datasets (BirdCLEF spectrograms, etc.) |

### 📂 02_Preprocessed

| Subfolder | Purpose |
|-----------|----------|
| `Standardized_Audio/` | Resampled, mono audio files (22050 Hz) |
| `Audio_Chunks/` | Fixed-length 5-second segments |
| `Quality_Reports/` | Audio analysis and quality reports |

### 📂 03_Features

| Subfolder | Purpose |
|-----------|----------|
| `Spectrograms/` | Mel-spectrograms generated from our audio |
| `Embeddings/` | Audio embeddings (if using pretrained models) |

### 📂 04_Labels

| Subfolder | Purpose |
|-----------|----------|
| `Raw_Labels/` | Original label files from metadata |
| `Processed_Labels/` | Multi-label encoded files |
| `Train_Val_Test_Split/` | Data split information |

### 📂 05_Model

| Subfolder | Purpose |
|-----------|----------|
| `Checkpoints/` | Model checkpoints during training |
| `Saved_Models/` | Final trained models (.pth, .h5) |
| `Training_Logs/` | TensorBoard logs, training history |
| `Configs/` | Model configuration YAML/JSON files |

### 📂 06_Explainability

| Subfolder | Purpose |
|-----------|----------|
| `GradCAM/` | Grad-CAM visualizations |
| `Attention_Maps/` | Attention heatmaps |
| `Temporal_Localization/` | Time-based detection results |

### 📂 07_Evaluation

| Subfolder | Purpose |
|-----------|----------|
| `Metrics/` | Performance metrics and reports |
| `Confusion_Matrices/` | Confusion matrix visualizations |
| `Predictions/` | Model predictions on test set |

### 📂 08_Deployment

| Subfolder | Purpose |
|-----------|----------|
| `API/` | FastAPI backend code |
| `Frontend/` | Streamlit/Gradio UI code |
| `Docker/` | Docker configuration files |

### 📂 09_Utils

| Subfolder | Purpose |
|-----------|----------|
| `Scripts/` | All Python utility scripts |
| `Notebooks/` | Jupyter notebooks for exploration |
| `Logs/` | General processing logs |
| `Temp/` | Temporary files (can be deleted) |

### 📂 10_Outputs

| Subfolder | Purpose |
|-----------|----------|
| `Reports/` | Generated reports (PDF, HTML) |
| `Visualizations/` | Charts, graphs, figures |
| `Exports/` | Exported data for sharing |

---

## 🗺️ Phase to Folder Mapping

| Phase | Primary Folders |
|-------|----------------|
| Phase 0: Metadata | `01_Raw_Data/Metadata/` |
| Phase 1: Preprocessing | `01_Raw_Data/` → `02_Preprocessed/` |
| Phase 2: Augmentation | `02_Preprocessed/` → `03_Features/` |
| Phase 3: Labels | `04_Labels/` |
| Phase 4-5: Model | `05_Model/` |
| Phase 6: Explainability | `06_Explainability/` |
| Phase 7: Evaluation | `07_Evaluation/` |
| Phase 8: Deployment | `08_Deployment/` |

