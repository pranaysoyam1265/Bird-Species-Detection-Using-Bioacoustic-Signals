# 🎵 QUICK REFERENCE - Long Test Samples

## 📁 Where Are They?

```
C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\10_Outputs\Test_Samples_Long\
```

## ⚡ Quick Start (Copy & Paste)

### Load One Sample
```python
import librosa
audio, sr = librosa.load("10_Outputs/Test_Samples_Long/test_composite_001.wav", sr=22050)
print(f"Duration: {len(audio)/sr:.1f} seconds")
```

### Load All Samples
```python
from test_sample_helper import TestSampleHelper
helper = TestSampleHelper("10_Outputs/Test_Samples_Long")
all_samples = helper.load_all_samples()  # Returns list of (audio, filename) tuples
```

### Test Your Model
```python
from batch_test_runner import BatchTestRunner

def my_predict(audio, sr):
    # Your model here
    return model.predict(audio, sr)

runner = BatchTestRunner()
results = runner.run_test(my_predict, model_name="my_model")
# Saves: 10_Outputs/Test_Results/results_my_model_*.json
```

---

## 📊 What You Got

| Aspect | Details |
|--------|---------|
| **Samples** | 20 files |
| **Duration** | 5 minutes each (300 sec) |
| **Format** | WAV, 22.05 kHz, Mono |
| **Total Size** | ~1 GB |
| **Source** | 7,884 unused Xeno-Canto recordings |
| **Training Overlap** | 0% (completely unused) |

---

## 🛠️ 3 Helper Scripts

### 1️⃣ `test_sample_helper.py`
```python
helper = TestSampleHelper()
audio, sr = helper.load_sample(1)          # Load sample 1
stats = helper.get_statistics(audio, sr)   # Get stats
features = helper.extract_features(audio, sr)  # Extract features
chunks = helper.chunk_sample(audio, sr)    # Create chunks
helper.print_summary()                      # Print table
```

### 2️⃣ `batch_test_runner.py`
```python
runner = BatchTestRunner()
results = runner.run_test(predict_fn, "model_name")  # Test all 20 samples
# Auto-saves JSON, CSV, TXT results
```

### 3️⃣ `generate_long_test_samples.py`
```bash
python generate_long_test_samples.py  # Creates more samples if needed
```

---

## 📈 Example: Test All 20 Samples in 10 Lines

```python
from batch_test_runner import BatchTestRunner
import librosa

def predict(audio, sr):
    # your model prediction
    return "Common_Blackbird"  # dummy output

runner = BatchTestRunner("10_Outputs/Test_Samples_Long")
results = runner.run_test(predict, "my_model")
print(f"✓ Tested {len(results)} samples")
print(f"✓ Results saved to: 10_Outputs/Test_Results/")
```

---

## 📋 Important Files

| File | Purpose |
|------|---------|
| `test_composite_001.wav` - `test_composite_020.wav` | 20 audio samples (5 min each) |
| `test_samples_metadata.json` | Complete metadata |
| `test_samples_manifest.csv` | Quick reference |
| `GENERATION_REPORT.txt` | Full details |
| `README_TEST_SAMPLES.md` | Detailed guide |

---

## ✅ Verify Installation

```bash
# Check samples exist
ls 10_Outputs/Test_Samples_Long/*.wav | wc -l
# Should print: 20

# Check Python dependencies
python -c "import librosa, soundfile; print('✓ OK')"

# Run example
python test_sample_helper.py --example
```

---

## 🎯 3-Step Testing

```
Step 1: Load Sample
    audio, sr = librosa.load("test_composite_001.wav", sr=22050)

Step 2: Run Model  
    prediction = model.predict(audio, sr)

Step 3: Get Results
    print(f"Prediction: {prediction}")
```

---

## 🔥 Most Common Tasks

### Task: Process One Sample
```python
import librosa
audio, sr = librosa.load("10_Outputs/Test_Samples_Long/test_composite_001.wav", sr=22050)
result = your_model(audio)
```

### Task: Process All Samples
```python
from batch_test_runner import BatchTestRunner
runner = BatchTestRunner()
results = runner.run_test(your_model_fn, "model_name")
```

### Task: Get Statistics
```python
from test_sample_helper import TestSampleHelper
helper = TestSampleHelper()
helper.print_summary()  # Prints summary table
```

### Task: Extract Features
```python
from test_sample_helper import TestSampleHelper
helper = TestSampleHelper()
audio, sr = helper.load_sample(1)
features = helper.extract_features(audio, sr)
# Returns: MFCC, mel-spec, ZCR, spectral centroid, energy
```

### Task: Create Chunks
```python
from test_sample_helper import TestSampleHelper
helper = TestSampleHelper()
audio, sr = helper.load_sample(1)
chunks = helper.chunk_sample(audio, sr, chunk_duration=30)
# Returns: List of 30-second overlapping chunks
```

---

## 🐛 Quick Fixes

**Error: "ModuleNotFoundError: No module named 'librosa'"**
```bash
pip install librosa soundfile
```

**Error: "FileNotFoundError: test_composite_001.wav"**
```python
import os
print(os.getcwd())  # Check current directory
os.chdir("C:/Users/prana/OneDrive/Desktop/ML Conf-BioFSL")
```

**Error: "MemoryError"**
```python
# Instead of loading all at once:
# ✗ all = helper.load_all_samples()

# Load one at a time:
# ✓ audio, sr = helper.load_sample(1)
```

---

## 📞 Documentation Map

```
🏠 Project Root/
├── LONG_TEST_SAMPLES_SUMMARY.md ← Comprehensive guide (THIS FOLDER)
├── test_sample_helper.py ← Load & explore samples
├── batch_test_runner.py ← Test all 20 samples
├── generate_long_test_samples.py ← Create more samples
│
└── 10_Outputs/Test_Samples_Long/ ← AUDIO FILES ARE HERE
    ├── test_composite_001.wav → test_composite_020.wav
    ├── README_TEST_SAMPLES.md ← Detailed usage guide
    ├── test_samples_metadata.json
    ├── test_samples_manifest.csv
    └── GENERATION_REPORT.txt
```

---

## 🎵 Sample Overview

### Composition
Each sample = 2-10 Xeno-Canto recordings + 0.5s silence gaps = 5 min

### Example
```
Sample 1:
  XC123456.mp3 (45.2s) + [gap] 
  + XC234567.mp3 (52.1s) + [gap]
  + XC345678.mp3 (38.9s) + [gap]
  + ... (7 total)
  = 300 sec (5.0 min)
```

### Quality
- ✓ All recordings from unused XC database
- ✓ No training/validation/test overlap
- ✓ Mixed species for real-world testing
- ✓ Consistent 5-minute duration

---

## 🚀 Production Workflow

```
1. Load sample
   ↓
2. Preprocess (normalize, extract features)
   ↓
3. Run model inference
   ↓
4. Collect prediction
   ↓
5. Save results
   ↓
6. Compare metrics
```

**DO IT AUTOMATICALLY**:
```python
runner = BatchTestRunner()
results = runner.run_test(my_model, "model_v1")
# Everything done! Check 10_Outputs/Test_Results/
```

---

## 📊 Metadata Quick Access

Get info about a sample:
```python
from test_sample_helper import TestSampleHelper
helper = TestSampleHelper()
info = helper.get_sample_info(1)

print(f"Recordings: {info['num_recordings_used']}")
for rec in info['source_recordings']:
    print(f"  • {rec['filename']}")
```

---

**🎉 YOU'RE READY TO TEST!**

Start with: `python test_sample_helper.py --example`

Questions? Check: `10_Outputs/Test_Samples_Long/README_TEST_SAMPLES.md`
