# 🚀 Deployment

This folder contains deployment-related files for the Bird Species Detection System.

## 📁 Structure

### Streamlit_Version/
Rapid prototype built with Streamlit for quick testing and demonstration.

**To run:**
```bash
cd Streamlit_Version
streamlit run app.py
```

**Features:**
- 🎵 Audio upload & analysis
- 📊 Detection results with confidence scores
- 🔬 Expert tools for detailed analysis  
- 🐦 Species browser & information
- ⚙️ Settings & configuration
- Export results as CSV/JSON

**Pages:**
1. `1_🎵_Analyze.py` - Main audio analysis interface
2. `2_📊_Results.py` - Detailed detection results
3. `3_🔬_Expert_Tools.py` - Advanced analysis tools
4. `4_🐦_Species_Browser.py` - Browse all species
5. `5_⚙️_Settings.py` - Application settings

**Key Files:**
- `app.py` - Streamlit main application
- `config.py` - Configuration & paths
- `utils/` - Helper modules (model, audio, visualization)
- `assets/` - Static assets & icons
- `temp/` - Temporary files cache

### Frontend/
Production frontend (React/Vue/other framework).

**To integrate your frontend:**
1. Place frontend files in this directory
2. Configure API endpoints to connect to backend
3. Update documentation below

### Backend/
API backend for production deployment (FastAPI/Flask).

**To add API backend:**
1. Create `api.py` or similar
2. Implement endpoints for model inference
3. Coordinate with frontend configuration

---

## 🔧 Requirements

### Streamlit Version
```bash
pip install streamlit plotly librosa soundfile noisereduce pandas numpy torch timm pillow
```

### Production
(To be defined based on your frontend framework)

---

## 📊 Model Integration

All versions use the same model located at:
```
../05_Model/Saved_Models/best_model_v3.pth
```

**Model specs:**
- Architecture: EfficientNet-B2 (`tf_efficientnet_b2_ns`)
- Species Detected: 87
- Input: Mel spectrogram (128 × 216 pixels)
- Validation Accuracy: 95.90%
- Test Accuracy: ~85%+ (on verified test set)

**Labels:**
- Location: `../04_Labels/Processed_Labels/label_mapping_v3.json`
- Format: JSON with species scientific names → {index, english_name}

---

## 🎯 Quick Start

### Running the Streamlit Prototype
```bash
# Navigate to project root
cd "C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"

# Run Streamlit app
cd 08_Deployment\Streamlit_Version
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🔄 Development Workflow

### Adding a New Streamlit Page
1. Create new file in `Streamlit_Version/pages/` 
2. Name format: `N_📌_Page_Name.py` (Streamlit auto-orders by number)
3. Import from `utils/`:
   ```python
   import streamlit as st
   from pathlib import Path
   import sys
   
   # Add parent dirs to path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   
   from config import PROJECT_ROOT, MODEL_CONFIG
   from utils.model import load_model
   ```
4. Use shared utilities from `utils/` for consistency

### Adding to Utilities
1. New modules go in `utils/` 
2. Create as separate `.py` files
3. Import in main scripts:
   ```python
   from utils.module_name import function_name
   ```

---

## 🛠️ Troubleshooting

### Model not found
- Check `05_Model/Saved_Models/best_model_v3.pth` exists
- Verify `config.py` paths are correct (use `Path(__file__).parent` for relative paths)

### Missing labels
- Ensure `04_Labels/Processed_Labels/label_mapping_v3.json` exists
- Verify JSON format: `{"species_scientific_name": {"index": 0, "english_name": "..."}}`

### Streamlit won't start
- Install requirements: `pip install streamlit plotly librosa soundfile noisereduce pandas numpy torch timm pillow`
- Check Python version (3.8+)
- Verify config paths are accessible

### Audio upload fails
- Check `TEMP_DIR` exists and has write permissions
- Verify `max_file_size_mb` in config.py
- Supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`

---

## 📝 Path Reference

From `Streamlit_Version/config.py`:
```
STREAMLIT_DIR    = Streamlit_Version/
DEPLOYMENT_DIR   = 08_Deployment/
PROJECT_ROOT     = ML Conf-BioFSL/

MODEL_PATH       = PROJECT_ROOT/05_Model/Saved_Models/best_model_v3.pth
LABEL_MAPPING    = PROJECT_ROOT/04_Labels/Processed_Labels/label_mapping_v3.json
TEST_RESULTS     = PROJECT_ROOT/05_Model/Training_Logs/test_results_v3_FINAL.json

ASSETS_DIR       = STREAMLIT_DIR/assets/
TEMP_DIR         = STREAMLIT_DIR/temp/
```

---

## ✅ Deployment Checklist

- [ ] Run `streamlit run app.py` and verify all pages load
- [ ] Test audio upload & analysis on sample file
- [ ] Verify model predictions are reasonable (>50% confidence)
- [ ] Test export functionality (CSV/JSON)
- [ ] Check all paths resolve correctly in logs
- [ ] Review model accuracy metrics in Results page
- [ ] Test species search/browser
- [ ] Verify settings persist across sessions

---

## 📞 Next Steps

1. **Production Frontend** - Add React/Vue/Svelte app to `Frontend/`
2. **API Backend** - Create FastAPI endpoints in `Backend/`
3. **Containerization** - Add Docker configs for deployment
4. **Documentation** - Generate API docs with Swagger/OpenAPI
5. **Testing** - Add unit tests for all modules

---

**Last Updated:** February 21, 2026  
**Version:** 1.0 - Streamlit Prototype  
**Status:** ✅ Active Development
