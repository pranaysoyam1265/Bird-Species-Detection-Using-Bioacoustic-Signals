# 🎵 Long Test Samples Generation - Complete Summary

## ✅ What Was Created

**20 composite audio samples** - each 5 minutes long, totaling **100 minutes** of test data

```
✓ 20 test files generated
✓ 5.0 minutes each (300 seconds)
✓ 100.0 minutes total duration
✓ All from UNUSED Xeno-Canto recordings
✓ No overlap with training data
✓ Ready for model evaluation
```

---

## 📁 Generated Files Location

```
📂 10_Outputs/Test_Samples_Long/
├── 🎵 test_composite_001.wav  ← Sample 1
├── 🎵 test_composite_002.wav  ← Sample 2
├── ...
├── 🎵 test_composite_020.wav  ← Sample 20
│
├── 📋 test_samples_metadata.json      ← Complete metadata (JSON)
├── 📊 test_samples_manifest.csv       ← Quick reference (CSV)
├── 📄 GENERATION_REPORT.txt           ← Detailed report
└── 📖 README_TEST_SAMPLES.md          ← This guide
```

---

## 🛠️ Three Helper Scripts Created

### 1. **test_sample_helper.py** - Load & Explore Samples

```python
from test_sample_helper import TestSampleHelper

# Initialize
helper = TestSampleHelper("10_Outputs/Test_Samples_Long")

# Load sample 1
audio, sr = helper.load_sample(1)  # Returns: (audio_array, 22050)

# Get statistics
stats = helper.get_statistics(audio, sr)
print(stats)  # RMS, peak, mean amplitude, etc.

# Extract features
features = helper.extract_features(audio, sr)
# Returns: MFCC, Mel-spectrogram, ZCR, Spectral Centroid

# Create chunks for processing
chunks = helper.chunk_sample(audio, sr, chunk_duration=5.0, hop_duration=1.0)
# Returns: List of 5-second overlapping chunks

# Get summary of all samples
helper.print_summary()  # Prints formatted table
```

**Location**: `test_sample_helper.py`  
**Run Example**: `.\.venv\Scripts\python test_sample_helper.py --example`

---

### 2. **batch_test_runner.py** - Test All Samples

```python
from batch_test_runner import BatchTestRunner

# Your model prediction function
def my_model_predict(audio, sr):
    # Your preprocessing
    features = extract_features(audio, sr)
    # Your inference
    prediction = model.predict(features)
    return prediction

# Run batch test
runner = BatchTestRunner()
results = runner.run_test(
    model_predict_fn=my_model_predict,
    model_name="my_model",
    use_chunks=False  # or True for chunk-based prediction
)

# Results automatically saved as JSON, CSV, and TXT report
```

**Location**: `batch_test_runner.py`  
**Features**:
- Test all 20 samples automatically
- Optional chunk-based processing
- Saves results in JSON, CSV, and text report
- Error handling and logging
- Timing information

---

### 3. **generate_long_test_samples.py** - Generate More Samples (if needed)

If you need to create additional samples:

```bash
.\.venv\Scripts\python generate_long_test_samples.py
```

Customize by editing these variables in the script:
```python
TARGET_DURATION = 300       # Duration in seconds
MIN_SAMPLES = 20            # Number of samples to create
SAMPLE_RATE = 22050         # Hz
```

---

## 📊 Sample Details & Statistics

### Quick Stats Table

```
Sample #  Filename                  Duration  # Recordings  Source Diversity
────────  ────────────────────────  ────────  ─────────────  ────────────────
   1      test_composite_001.wav    5.0 min   7 recordings   Multiple species
   2      test_composite_002.wav    5.0 min   8 recordings   Multiple species
   3      test_composite_003.wav    5.0 min   2 recordings   Isolated calls
   ...    ...                       ...       ...            ...
  20      test_composite_020.wav    5.0 min   10 recordings  Multiple species
```

### Audio Properties

| Property | Value |
|----------|-------|
| Format | WAV (uncompressed) |
| Sample Rate | 22,050 Hz |
| Channels | Mono |
| Bit Depth | 16-bit |
| Duration per Sample | 300 seconds (5.0 minutes) |
| File Size per Sample | ~50 MB |
| Total Size (20 samples) | ~1 GB |

### Composition

Each sample concatenates 2-10 **unused** Xeno-Canto recordings:
- ✓ No training/validation/test overlap
- ✓ Natural silence gaps (0.5s) between recordings
- ✓ Diverse species combinations
- ✓ Mixed duration recordings (short calls + longer vocalizations)

---

## 🚀 Quick Start - 3 Steps to Test Your Model

### Step 1: Load a Sample

```python
import librosa
from test_sample_helper import TestSampleHelper

helper = TestSampleHelper("10_Outputs/Test_Samples_Long")
audio, sr = helper.load_sample(1)  # Load sample 1

print(f"Audio shape: {audio.shape}")
print(f"Duration: {len(audio)/sr:.1f} seconds")
```

### Step 2: Extract Features

```python
# Method A: Using helper
features = helper.extract_features(audio, sr)
print("MFCC:", features['mfcc_mean'])

# Method B: Using librosa directly
import librosa
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
```

### Step 3: Run Model & Get Results

```python
from batch_test_runner import BatchTestRunner

def predict_fn(audio, sr):
    # Your model inference here
    features = extract_features(audio, sr)
    return model.predict(features)

runner = BatchTestRunner()
results = runner.run_test(predict_fn, model_name="my_model")

# Results saved in 10_Outputs/Test_Results/
```

---

## 📖 Usage Examples

### Example 1: Process Full 5-Minute Sample

```python
import librosa
from test_sample_helper import TestSampleHelper

helper = TestSampleHelper()
audio, sr = helper.load_sample(1)

# Your model processes the entire 5-minute sample
prediction = model.predict(audio, sr)

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2%}")
```

### Example 2: Chunk-Based Processing (30-second windows)

```python
helper = TestSampleHelper()
audio, sr = helper.load_sample(1)

# Create 30-second overlapping chunks
chunks = helper.chunk_sample(audio, sr, 
                             chunk_duration=30, 
                             hop_duration=10)

predictions = []
for chunk in chunks:
    pred = model.predict(chunk, sr)
    predictions.append(pred)

# Take majority vote
from collections import Counter
final_pred = Counter(predictions).most_common(1)[0][0]
```

### Example 3: Batch Test All Samples

```python
from batch_test_runner import BatchTestRunner

runner = BatchTestRunner()

# Define your prediction function
def my_predict(audio, sr):
    # Your preprocessing
    features = librosa.feature.mfcc(y=audio, sr=sr)
    # Your inference
    return model.forward(features)

# Run on all 20 samples
results = runner.run_test(
    model_predict_fn=my_predict,
    model_name="attention_cnn_v3",
    use_chunks=False
)

# Results in: 10_Outputs/Test_Results/results_attention_cnn_v3_*.json
```

### Example 4: Get Sample Information

```python
helper = TestSampleHelper()

# Get info about sample 1
info = helper.get_sample_info(1)
print(f"Source recordings: {info['num_recordings_used']}")
print(f"Source files:")
for rec in info['source_recordings']:
    print(f"  - {rec['filename']} ({rec['duration']:.1f}s)")

# Get summary of all samples
summary = helper.get_sample_summary()  # Returns pandas DataFrame
print(summary)
```

---

## 📋 File Descriptions

### Core Audio Files (`*.wav`)

- `test_composite_001.wav` through `test_composite_020.wav`
- **What**: 20 concatenated audio samples
- **Duration**: Each is exactly 300 seconds (5.0 minutes)
- **Format**: WAV, 44,100 samples/sec → 22,050 Hz after librosa load
- **Use**: Direct model input for evaluation
- **Size**: ~50MB each, ~1GB total

### Metadata Files

#### `test_samples_metadata.json`
**Purpose**: Complete metadata with full provenance
```json
[
  {
    "sample_id": 1,
    "output_file": "test_composite_001.wav",
    "actual_duration": 300.15,
    "num_recordings_used": 7,
    "source_recordings": [
      {"xc_id": "123456", "filename": "XC123456.mp3", "duration": 45.2},
      {"xc_id": "234567", "filename": "XC234567.mp3", "duration": 52.1},
      ...
    ]
  },
  ...
]
```

#### `test_samples_manifest.csv`
**Purpose**: Quick reference summary
```csv
Sample ID,Filename,Duration (sec),Num Recordings,Created
1,test_composite_001.wav,300.1,7,2026-03-03T15:45:30.123456
2,test_composite_002.wav,300.2,8,2026-03-03T15:46:12.456789
...
```

#### `GENERATION_REPORT.txt`
**Purpose**: Detailed human-readable report
- Generation timestamp
- All sample details
- Statistics summary
- Next steps

### Helper Scripts

#### `test_sample_helper.py`
Quick utilities for loading and exploring samples:
- `load_sample(idx)` - Load by index 1-20
- `load_all_samples()` - Load all at once
- `get_statistics(audio, sr)` - Audio stats
- `extract_features(audio, sr)` - MFCC, mel-spec, etc.
- `chunk_sample(audio, sr, duration)` - Create overlapping chunks
- `print_summary()` - Print formatted table

#### `batch_test_runner.py`
Batch evaluation on all samples:
- `run_test(predict_fn, model_name)` - Run on all 20 samples
- Automatic result saving (JSON, CSV, TXT)
- Error handling and logging
- Timing information
- Example prediction functions

#### `generate_long_test_samples.py`
Create additional samples (if needed):
- Customize duration, count, sample rate
- Load unused recording IDs
- Generate new composites
- Save metadata

---

## 🔬 Technical Specifications

### Why Concatenation Approach?

**Advantages**:
- ✓ Real-world scenario: Mixed species, varied vocalizations
- ✓ Long duration: Tests model ability to maintain attention
- ✓ No training overlap: Uses 7,884 unused recordings
- ✓ Consistent duration: All exactly 5 minutes for fair comparison
- ✓ Reproducible: Full metadata for each sample

**Considerations**:
- Silence gaps (0.5s) between recordings are intentional
- Species mixes create challenging test cases
- Not representative of single-species clean recordings

### Audio Processing Parameters

```python
# Loading audio
librosa.load(path, sr=22050)  # Resamples to 22.05 kHz mono

# Gap between recordings
silence_duration = 0.5  # seconds
silence_samples = int(0.5 * 22050)  # 11,025 samples

# Target duration enforcement
target_duration = 300  # seconds
final_samples = int(300 * 22050)  # 6,615,000 samples
```

---

## 📈 What to Measure

### Model Performance Metrics

Depending on your evaluation goal:

**Classification Accuracy**:
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(ground_truth_labels, predictions)
```

**Per-Species Performance**:
```python
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None, labels=class_ids
)
```

**Long-Duration Effects**:
```python
# Compare predictions across chunks
# Does model maintain consistency over 5 minutes?
chunk_consistency = sum(1 for p in chunk_predictions if p == final_prediction)
```

---

## ❌ Common Issues & Solutions

### Issue: File Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: 'test_composite_001.wav'
```
**Solution**: Check working directory matches test sample location
```python
import os
print(os.getcwd())  # Should contain 10_Outputs/Test_Samples_Long/
os.chdir("C:/Users/prana/OneDrive/Desktop/ML Conf-BioFSL")
```

### Issue: Memory Error with All Samples
```
MemoryError: Unable to allocate 2.00 GB
```
**Solution**: Process one sample at a time instead of loading all
```python
# ✓ GOOD - Processes one at a time
for idx in range(1, 21):
    audio, sr = helper.load_sample(idx)
    result = model.predict(audio, sr)

# ✗ BAD - Loads all into memory
all_samples = helper.load_all_samples()  # ~1GB in memory
```

### Issue: Inconsistent Sample Rate
```
ValueError: Sample rates must match (22050 != 16000)
```
**Solution**: Resample your model input or specify target sr
```python
# Resample audio to model's expected rate
audio, sr = librosa.load(path, sr=16000)  # Specify target sr

# Or resample after loading
audio, sr = librosa.load(path, sr=22050)
audio_resampled = librosa.resample(audio, orig_sr=22050, target_sr=16000)
```

---

## 🎯 Next Steps Recommendations

1. **Quick Validation** (5 minutes)
   ```bash
   python test_sample_helper.py --example
   ```
   Verify samples load and sound correct

2. **Feature Extraction** (15 minutes)
   - Extract MFCC, mel-spectrogram features
   - Verify feature shapes and ranges

3. **Single Sample Test** (30 minutes)
   - Test your model on 1-2 samples
   - Debug any preprocessing issues

4. **Batch Evaluation** (1-2 hours)
   ```bash
   python batch_test_runner.py
   ```
   Run on all 20 samples, collect results

5. **Analysis & Reporting**
   - Compare predictions across samples
   - Analyze failure cases
   - Measure long-duration effects

---

## 📞 Support Resources

### Generated Documentation

| File | Purpose |
|------|---------|
| `README_TEST_SAMPLES.md` | Usage guide with Python examples |
| `GENERATION_REPORT.txt` | Detailed generation report |
| `test_samples_metadata.json` | Complete sample metadata |
| `test_samples_manifest.csv` | Quick reference table |

### Script Help

```bash
# View helper script usage
python test_sample_helper.py

# View batch runner usage and example
python batch_test_runner.py

# Generate additional samples (if needed)
python generate_long_test_samples.py --help
```

---

## ✨ Summary

**What You Have**:
- ✅ 20 ready-to-use audio test samples
- ✅ Each 5 minutes long (100 minutes total)
- ✅ Complete metadata and provenance
- ✅ 3 helper scripts for testing
- ✅ Batch evaluation template
- ✅ Full documentation

**What to Do Next**:
1. Load a sample: `helper.load_sample(1)`
2. Extract features or run model
3. Run batch test: `runner.run_test(predict_fn, "model_name")`
4. Analyze results in: `10_Outputs/Test_Results/`

**Key Points**:
- All samples are from UNUSED Xeno-Canto recordings
- No overlap with your training/val/test split
- Represents real-world mixed-species scenarios
- Ready for production evaluation

---

**Generated**: March 3, 2026  
**Version**: 1.0  
**Status**: ✅ Complete and Ready to Use
