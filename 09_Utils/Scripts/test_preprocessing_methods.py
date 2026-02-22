# Save as: test_preprocessing_methods.py

import torch
import torch.nn as nn
import timm
import numpy as np
import pandas as pd
from pathlib import Path
import json
import os

os.chdir(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")

print("="*60)
print("🧪 TESTING DIFFERENT PREPROCESSING METHODS")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
class BirdClassifier(nn.Module):
    def __init__(self, num_classes=87, dropout_rate=0.4):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnet_b2_ns', pretrained=False, 
                                          num_classes=0, global_pool='avg')
        num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.75),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.backbone(x))

model = BirdClassifier(num_classes=87)
checkpoint = torch.load('05_Model/Saved_Models/best_model_v3.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Load label mapping
with open('04_Labels/Processed_Labels/label_mapping_v3.json', 'r') as f:
    label_mapping_raw = json.load(f)
idx_to_species = {v['index']: k for k, v in label_mapping_raw.items()}

# Load test samples
test_df = pd.read_csv('04_Labels/Train_Val_Test_Split/test_v3.csv')
spec_dir = Path('03_Features/Spectrograms')

# Get 30 random test samples
test_samples = []
for _, row in test_df.sample(n=30, random_state=123).iterrows():
    spec_path = Path(row['spectrogram_path'])
    if spec_path.exists():
        test_samples.append({
            'path': spec_path,
            'label': int(row['label']),
            'species': row['species_scientific']
        })

print(f"Loaded {len(test_samples)} test samples")

# Load one spectrogram to check format
sample_spec = np.load(test_samples[0]['path'])
print(f"\nSpectrogram shape: {sample_spec.shape}")
print(f"Spectrogram dtype: {sample_spec.dtype}")
print(f"Spectrogram range: [{sample_spec.min():.3f}, {sample_spec.max():.3f}]")

# Different preprocessing methods to try
def preprocess_v1(spec):
    """Method 1: Simple normalization to 0-1, repeat to 3 channels"""
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    spec = np.stack([spec, spec, spec], axis=0)
    return torch.from_numpy(spec).float()

def preprocess_v2(spec):
    """Method 2: ImageNet normalization after 0-1 scaling"""
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    spec = np.stack([spec, spec, spec], axis=0)
    tensor = torch.from_numpy(spec).float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std

def preprocess_v3(spec):
    """Method 3: No normalization, just repeat channels"""
    spec = np.stack([spec, spec, spec], axis=0)
    return torch.from_numpy(spec).float()

def preprocess_v4(spec):
    """Method 4: Scale to 0-255, then /255, then ImageNet norm"""
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    spec = spec * 255.0
    spec = spec / 255.0
    spec = np.stack([spec, spec, spec], axis=0)
    tensor = torch.from_numpy(spec).float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std

def preprocess_v5(spec):
    """Method 5: dB scale normalization (assuming min is -80dB, max is 0)"""
    # Typical mel spectrogram in dB has range ~[-80, 0]
    spec = (spec + 80) / 80  # Normalize assuming -80dB floor
    spec = np.clip(spec, 0, 1)
    spec = np.stack([spec, spec, spec], axis=0)
    return torch.from_numpy(spec).float()

def preprocess_v6(spec):
    """Method 6: Standard scaling (mean=0, std=1) then to 0-1"""
    spec = (spec - spec.mean()) / (spec.std() + 1e-8)
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    spec = np.stack([spec, spec, spec], axis=0)
    return torch.from_numpy(spec).float()

def preprocess_v7(spec):
    """Method 7: Just use raw values with 3 channels"""
    # Maybe training used raw dB values?
    spec = np.stack([spec, spec, spec], axis=0)
    tensor = torch.from_numpy(spec).float()
    # Normalize to have similar scale as ImageNet
    tensor = tensor / 60.0 + 0.5  # Rough scaling for [-60, 0] range
    return tensor

def preprocess_v8(spec):
    """Method 8: Inverse (flip) the spectrogram"""
    spec = -spec  # Flip sign
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    spec = np.stack([spec, spec, spec], axis=0)
    return torch.from_numpy(spec).float()

preprocessing_methods = [
    ("v1: Simple 0-1", preprocess_v1),
    ("v2: 0-1 + ImageNet", preprocess_v2),
    ("v3: Raw (no norm)", preprocess_v3),
    ("v4: 0-255-0-1 + ImageNet", preprocess_v4),
    ("v5: dB scale (-80,0)->0-1", preprocess_v5),
    ("v6: StandardScale -> 0-1", preprocess_v6),
    ("v7: Raw scaled /60+0.5", preprocess_v7),
    ("v8: Inverted + 0-1", preprocess_v8),
]

print("\n" + "="*60)
print("🧪 TESTING PREPROCESSING METHODS")
print("="*60)

for method_name, preprocess_fn in preprocessing_methods:
    correct = 0
    total = 0
    
    for sample in test_samples:
        spec = np.load(sample['path'])
        true_label = sample['label']
        
        try:
            tensor = preprocess_fn(spec)
            tensor = tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(tensor)
                pred_label = output.argmax(dim=1).item()
            
            if pred_label == true_label:
                correct += 1
            total += 1
            
        except Exception as e:
            pass
    
    acc = 100 * correct / total if total > 0 else 0
    print(f"  {method_name:30s}: {correct:2d}/{total:2d} = {acc:5.1f}%")

print("\n" + "="*60)
print("📋 RECOMMENDATION")
print("="*60)
print("""
The preprocessing method with highest accuracy is likely 
what was used during training.

If all methods give ~0-1% accuracy, the issue is deeper:
- Model may have been saved incorrectly
- Training may have used different data transforms
- There may be a bug in the training pipeline

Next step: Examine the training script to find exact preprocessing.
""")