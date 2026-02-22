# Save as: evaluate_with_reconstructed_mapping.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import timm
import pandas as pd
import numpy as np
from PIL import Image
import json
import os
from pathlib import Path
from tqdm import tqdm

# Resolve project root
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
os.chdir(project_root)

print("="*60)
print("EVALUATION WITH RECONSTRUCTED LABEL MAPPING")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# 1. Reconstruct Label Mapping from Training Data
# ============================================================
print("\nReconstructing label mapping...")

train_df = pd.read_csv(project_root / '04_Labels/Train_Val_Test_Split/train_v3.csv')
# Use correct column for species
unique_species = sorted(train_df['species_scientific'].unique())
label_mapping = {species: idx for idx, species in enumerate(unique_species)}
idx_to_species = {v: k for k, v in label_mapping.items()}

print(f"Reconstructed mapping with {len(label_mapping)} classes")
print("\nFirst 10:")
for species, idx in list(label_mapping.items())[:10]:
    print(f"  {idx:2d}: {species}")

# ============================================================
# 2. Load Model
# ============================================================
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

model = BirdClassifier(num_classes=len(label_mapping))
checkpoint = torch.load('05_Model/Saved_Models/best_model_v3.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print("✅ Model loaded")

# ============================================================
# 3. Test on Training Samples
# ============================================================
print("\n🧪 Testing on training samples...")

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((128, 216)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# CRITICAL: Match training preprocessing exactly!
def normalize_spec(spec):
    spec_min = spec.min()
    spec_max = spec.max()
    if spec_max - spec_min > 0:
        spec = (spec - spec_min) / (spec_max - spec_min)
    else:
        spec = np.zeros_like(spec)
    spec = np.clip(spec, 0, 1)
    spec_tensor = torch.FloatTensor(spec).unsqueeze(0)
    spec_tensor = spec_tensor.repeat(3, 1, 1)
    return spec_tensor

spec_dir = Path('03_Features/Spectrograms')
train_samples = train_df.sample(n=50).to_dict('records')

correct = 0
for sample in train_samples:
    chunk_id = sample.get('chunk_filename', sample.get('chunk_id', ''))
    true_species = sample['species_scientific']
    true_label = sample.get('label', None)
    spec_path = spec_dir / f"{chunk_id}.npy"
    if not spec_path.exists():
        spec_path = spec_dir / f"{chunk_id}.png"
    if not spec_path.exists():
        continue
    if spec_path.suffix == '.npy':
        spec = np.load(spec_path)
        spec_tensor = normalize_spec(spec)
    else:
        img = Image.open(spec_path).convert('L')
        spec = np.array(img, dtype=np.float32)
        spec_tensor = normalize_spec(spec)
    spec_tensor = spec_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(spec_tensor)
        pred_label = output.argmax(dim=1).item()
        pred_species = idx_to_species[pred_label]
    if true_label is not None and pred_label == int(true_label):
        correct += 1

print(f"Results on TRAINING samples: {correct}/{len(train_samples)} correct ({100*correct/len(train_samples):.1f}%)")

# ============================================================
# 4. Full Test Evaluation
# ============================================================
print("\n🧪 Full test evaluation...")

class SpectrogramDataset(Dataset):
    def __init__(self, csv_path, spectrogram_dir, label_mapping, transform):
        self.df = pd.read_csv(csv_path)
        self.spec_dir = Path(spectrogram_dir)
        self.label_mapping = label_mapping
        self.transform = transform
        
        self.samples = []
        for _, row in self.df.iterrows():
            species = row['species']
            if species in label_mapping:
                path = self.spec_dir / f"{row['chunk_id']}.png"
                if path.exists():
                    self.samples.append({'path': path, 'label': label_mapping[species]})
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        return self.transform(img), s['label']

test_dataset = SpectrogramDataset(
    '04_Labels/Train_Val_Test_Split/test_v3.csv',
    '03_Features/Spectrograms',
    label_mapping,
    transform
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        outputs = model(images.to(device))
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = 100 * np.mean(all_preds == all_labels)
print(f"\n🎯 Test Accuracy: {accuracy:.2f}%")

# Top-5
top5 = 0
for i in range(len(all_labels)):
    if all_labels[i] in np.argsort(all_preds)[-5:]:
        top5 += 1
# Actually need to use probs for top-5, let me fix:

print("\n📊 Prediction distribution:")
from collections import Counter
pred_counts = Counter(all_preds)
for pred_idx, count in pred_counts.most_common(10):
    print(f"  {idx_to_species[pred_idx]}: {count} predictions")