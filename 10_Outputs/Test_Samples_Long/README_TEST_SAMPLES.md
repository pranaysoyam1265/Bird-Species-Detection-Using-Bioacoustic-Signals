# 🎵 Long Test Samples - User Guide

## Quick Summary

✅ **20 composite audio samples generated**  
✅ **Each sample: 5 minutes (300 seconds) long**  
✅ **Total test duration: 100 minutes**  
✅ **All samples from unused recordings (NOT in training set)**  

---

## 📂 Generated Files Location

```
10_Outputs/Test_Samples_Long/
├── test_composite_001.wav      # 5-minute sample 1
├── test_composite_002.wav      # 5-minute sample 2
├── test_composite_003.wav      # ... and so on
├── ...
├── test_composite_020.wav      # 5-minute sample 20
│
├── test_samples_metadata.json  # Detailed metadata (JSON format)
├── test_samples_manifest.csv   # Quick reference (CSV format)
├── GENERATION_REPORT.txt       # Full report with details
└── README_TEST_SAMPLES.md      # This file
```

---

## 📋 Sample Details

### Sample Information

| Property | Details |
|----------|---------|
| **Count** | 20 samples |
| **Duration** | 5 minutes (300 seconds) each |
| **Format** | WAV (Waveform Audio) |
| **Sample Rate** | 22,050 Hz (22.05 kHz) |
| **Channels** | Mono |
| **Total Size** | ~100 minutes of audio |

### How Samples Were Created

Each test sample is a **concatenation** of 2-10 unused Xeno-Canto recordings:
- **Silence gaps**: 0.5 seconds between each recording
- **Source diversity**: Each sample uses different combinations of recordings
- **No training overlap**: All source recordings were NOT used in model training

Example (Sample 1):
```
XC123456.mp3 (0.8 min) + [silence] 
  + XC234567.mp3 (0.9 min) + [silence]
  + XC345678.mp3 (0.6 min) + [silence]
  + ... (7 total recordings)
  = 5.0 minute composite sample
```

---

## 🎯 How to Use These Samples

### Option 1: Direct Testing with Your Model

```python
import librosa
import numpy as np
import torch

# Load a long test sample
audio_path = "10_Outputs/Test_Samples_Long/test_composite_001.wav"
audio, sr = librosa.load(audio_path, sr=22050)

# Run inference on the full 5-minute sample
predictions = model(audio)

# Or process in chunks (if your model expects shorter inputs)
chunk_size = 30000  # e.g., 30k samples
for i in range(0, len(audio), chunk_size):
    chunk = audio[i:i+chunk_size]
    chunk_pred = model(chunk)
```

### Option 2: Sliding Window Analysis

```python
import librosa
from collections import Counter

audio_path = "10_Outputs/Test_Samples_Long/test_composite_001.wav"
audio, sr = librosa.load(audio_path, sr=22050)

# Process with sliding window
window_size = sr * 5  # 5-second windows
hop_size = sr * 1     # 1-second hop

all_predictions = []
for start in range(0, len(audio) - window_size, hop_size):
    chunk = audio[start:start + window_size]
    pred = model(chunk)
    all_predictions.append(pred)

# Get most common prediction for the entire 5-minute sample
final_label = Counter(all_predictions).most_common(1)[0][0]
```

### Option 3: Batch Evaluation

```python
import os
import glob
import pandas as pd

# Load all test samples
test_dir = "10_Outputs/Test_Samples_Long"
test_samples = sorted(glob.glob(os.path.join(test_dir, "test_composite_*.wav")))

results = []
for sample_path in test_samples:
    audio, sr = librosa.load(sample_path, sr=22050)
    
    # Get prediction
    prediction = model(audio)
    confidence = model.confidence(audio)
    
    results.append({
        'sample_file': os.path.basename(sample_path),
        'prediction': prediction,
        'confidence': confidence,
        'duration_seconds': len(audio) / sr
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("test_results.csv", index=False)
print(results_df)
```

---

## 📊 Metadata Reference

### JSON Format (test_samples_metadata.json)

```json
[
  {
    "sample_id": 1,
    "output_file": "test_composite_001.wav",
    "output_path": "C:/path/to/test_composite_001.wav",
    "target_duration": 300,
    "actual_duration": 300.15,
    "num_recordings_used": 7,
    "failed_to_load": 0,
    "source_recordings": [
      {
        "xc_id": "123456",
        "duration": 45.2,
        "filename": "XC123456.mp3"
      },
      {
        "xc_id": "234567",
        "duration": 52.1,
        "filename": "XC234567.mp3"
      }
      ...
    ],
    "created_at": "2026-03-03T15:45:30.123456"
  },
  ...
]
```

### CSV Format (test_samples_manifest.csv)

```csv
Sample ID,Filename,Duration (sec),Num Recordings,Created
1,test_composite_001.wav,300.1,7,2026-03-03T15:45:30.123456
2,test_composite_002.wav,300.2,8,2026-03-03T15:46:12.456789
...
```

---

## ✅ Quality Assurance

### Verification Checklist

- ✅ **No training overlap**: All 7,884 source XC recordings were unused
- ✅ **Consistent duration**: All samples are ~5 minutes (±0.1s)
- ✅ **Valid format**: WAV format, 22.05 kHz sample rate
- ✅ **Source diversity**: 2-10 different recordings per sample
- ✅ **Silence gaps**: 0.5s gaps between recordings for clarity
- ✅ **Metadata complete**: Full provenance for each sample

### Potential Considerations

⚠️ **Concatenation Effects**:
- Silence gaps may affect audio feature extraction
- Abrupt transitions between recordings are natural
- Consider this when analyzing performance

⚠️ **Species Diversity**:
- Each sample may contain multiple bird species
- Mixed-species samples test real-world robustness
- Not representative of single-species recordings

---

## 🛠️ Advanced Usage

### Remove Silence Gaps (Optional)

```python
import librosa
import soundfile as sf
import numpy as np

def remove_silence_gaps(audio, sr, threshold_db=-40):
    """Remove silence from concatenated samples"""
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Identify non-silent frames
    noisy = S_db.mean(axis=0) > threshold_db
    
    # Get audio indices
    idx = np.where(noisy)[0]
    audio_frames = librosa.frames_to_samples(idx)
    
    return audio[audio_frames]

# Apply
audio_path = "test_composite_001.wav"
audio, sr = librosa.load(audio_path, sr=22050)
clean_audio = remove_silence_gaps(audio, sr)

# Save
sf.write("test_composite_001_clean.wav", clean_audio, sr)
```

### Normalize Volume

```python
from scipy import signal
import soundfile as sf

def normalize_loudness(audio, target_db=-20):
    """Normalize to target loudness"""
    current_db = 20 * np.log10(np.std(audio) + 1e-5)
    adjustment_db = target_db - current_db
    adjustment_linear = 10 ** (adjustment_db / 20)
    return audio * adjustment_linear

audio, sr = librosa.load("test_composite_001.wav", sr=22050)
normalized = normalize_loudness(audio, target_db=-20)
sf.write("test_composite_001_normalized.wav", normalized, sr)
```

### Extract Features

```python
import librosa
import numpy as np

audio_path = "test_composite_001.wav"
audio, sr = librosa.load(audio_path, sr=22050)

# Extract multiple features
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
zcr = librosa.feature.zero_crossing_rate(audio)[0]

features = {
    'mfcc_mean': mfcc.mean(axis=1),
    'mel_spec_mean': librosa.power_to_db(mel_spec).mean(axis=1),
    'chroma_mean': chroma.mean(axis=1),
    'zcr_mean': zcr.mean()
}

print("Extracted features:", features)
```

---

## 📈 Performance Evaluation Example

```python
import glob
import librosa
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd

# Assuming you have ground truth labels
test_samples_dir = "10_Outputs/Test_Samples_Long"
test_files = sorted(glob.glob(f"{test_samples_dir}/test_composite_*.wav"))

predictions = []
ground_truth = []  # You need to provide this!

for test_file in test_files:
    audio, sr = librosa.load(test_file, sr=22050)
    pred = model.predict(audio)
    predictions.append(pred)

# Calculate metrics
accuracy = accuracy_score(ground_truth, predictions)
precision = precision_score(ground_truth, predictions, average='weighted')
recall = recall_score(ground_truth, predictions, average='weighted')

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
```

---

## 📞 Support Information

### Files in This Directory

| File | Purpose |
|------|---------|
| `test_composite_*.wav` | Audio samples (use these for testing!) |
| `test_samples_metadata.json` | Complete metadata in JSON format |
| `test_samples_manifest.csv` | Quick reference in CSV format |
| `GENERATION_REPORT.txt` | Detailed generation report |
| `README_TEST_SAMPLES.md` | This file |

### Important Notes

1. **Sample Rate**: All samples are 22.05 kHz. Convert if your model expects different SR.
2. **Duration**: All samples are exactly 5 minutes (300 seconds) ± 0.1s
3. **Format**: WAV format, Mono channel
4. **Storage**: Each ~50MB file (20 files ≈ 1GB total)
5. **Portability**: WAV is widely compatible with all audio libraries

---

## 🧪 Testing Workflow

```
1. Load test sample
          ↓
2. Preprocess (normalize, features)
          ↓
3. Run model inference
          ↓
4. Collect predictions
          ↓
5. Compare with ground truth (if available)
          ↓
6. Calculate metrics
          ↓
7. Document results
```

---

**Generated**: 2026-03-03  
**Version**: 1.0  
**Total Samples**: 20 (5 minutes each)  
**Total Duration**: 100 minutes
