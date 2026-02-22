"""
generate_gradcam.py
Generate Grad-CAM visualizations for model explainability

Shows which parts of the spectrogram the model focuses on
for bird species identification.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")

# Paths
TEST_CSV = BASE_DIR / "04_Labels" / "Train_Val_Test_Split" / "test.csv"
TEST_SPEC_DIR = BASE_DIR / "03_Features" / "Spectrograms" / "test"
MODEL_PATH = BASE_DIR / "05_Model" / "Saved_Models" / "best_model.pth"
LABEL_MAPPING = BASE_DIR / "04_Labels" / "Processed_Labels" / "label_mapping.json"

# Output
GRADCAM_DIR = BASE_DIR / "06_Explainability" / "GradCAM"
SAMPLES_DIR = GRADCAM_DIR / "sample_visualizations"

# Settings
NUM_SAMPLES = 30  # Number of samples to visualize
SAMPLES_PER_CLASS = 2  # Samples per species

# ============================================================
# DATASET
# ============================================================

class BirdSpectrogramDataset(Dataset):
    def __init__(self, csv_path, spec_dir):
        self.df = pd.read_csv(csv_path)
        self.spec_dir = Path(spec_dir)
        
        valid_rows = []
        for idx, row in self.df.iterrows():
            spec_file = row['chunk_file'].replace('.wav', '.npy')
            if (self.spec_dir / spec_file).exists():
                valid_rows.append(idx)
        
        self.df = self.df.loc[valid_rows].reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        spec_file = row['chunk_file'].replace('.wav', '.npy')
        spec_path = self.spec_dir / spec_file
        
        spec = np.load(spec_path)
        spec_original = spec.copy()
        spec = self._pad_or_crop(spec, target_time=216)
        
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0)
        spec_tensor = spec_tensor.repeat(3, 1, 1)
        
        return spec_tensor, row['label_id'], row['chunk_file'], row['species'], spec_original
    
    def _pad_or_crop(self, spec, target_time):
        current_time = spec.shape[1]
        if current_time < target_time:
            padding = target_time - current_time
            spec = np.pad(spec, ((0, 0), (0, padding)), mode='constant')
        elif current_time > target_time:
            spec = spec[:, :target_time]
        return spec

# ============================================================
# MODEL WITH GRAD-CAM HOOKS
# ============================================================

class BirdClassifierWithGradCAM(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=54):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool=''
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, 128, 216)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # For Grad-CAM
        self.gradients = None
        self.activations = None
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x, return_cam=False):
        # Extract features
        features = self.backbone(x)
        
        if return_cam:
            # Register hook for gradients
            features.register_hook(self.activations_hook)
            self.activations = features
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.flatten(1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits
    
    def get_gradcam(self, class_idx):
        """Generate Grad-CAM heatmap for specified class"""
        
        # Get gradients and activations
        gradients = self.gradients  # (B, C, H, W)
        activations = self.activations  # (B, C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1)  # (B, H, W)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam

# ============================================================
# VISUALIZATION
# ============================================================

def create_gradcam_visualization(spectrogram, cam, true_species, pred_species, 
                                  confidence, chunk_file, save_path):
    """Create and save Grad-CAM visualization"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Original spectrogram
    axes[0].imshow(spectrogram, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title('Original Spectrogram', fontsize=12)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Frequency (Mel)')
    
    # 2. Grad-CAM heatmap
    cam_resized = np.array(
        plt.cm.jet(cam)[:, :, :3]
    )
    axes[1].imshow(cam, aspect='auto', origin='lower', cmap='jet')
    axes[1].set_title('Grad-CAM Attention', fontsize=12)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Frequency (Mel)')
    
    # 3. Overlay
    # Resize CAM to match spectrogram
    import cv2
    cam_resized = cv2.resize(cam, (spectrogram.shape[1], spectrogram.shape[0]))
    
    # Create overlay
    heatmap = cm.jet(cam_resized)[:, :, :3]
    spec_normalized = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-8)
    spec_rgb = cm.magma(spec_normalized)[:, :, :3]
    
    overlay = 0.6 * spec_rgb + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    axes[2].imshow(overlay, aspect='auto', origin='lower')
    axes[2].set_title('Overlay (Spectrogram + Attention)', fontsize=12)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Frequency (Mel)')
    
    # Overall title
    correct = "✓" if true_species == pred_species else "✗"
    color = "green" if true_species == pred_species else "red"
    
    fig.suptitle(
        f'{correct} True: {true_species} | Predicted: {pred_species} | Confidence: {confidence:.1%}',
        fontsize=14,
        color=color,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🔍 GRAD-CAM EXPLAINABILITY GENERATION")
    print("=" * 70)
    
    # Create directories
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Load label mapping
    with open(LABEL_MAPPING, 'r') as f:
        label_map = json.load(f)
    
    num_classes = label_map['num_classes']
    id_to_species = {int(k): v for k, v in label_map['id_to_species'].items()}
    
    # Load model
    print(f"\n🧠 Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    model = BirdClassifierWithGradCAM(
        model_name='efficientnet_b0',
        num_classes=num_classes
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print(f"\n📂 Loading test dataset...")
    dataset = BirdSpectrogramDataset(TEST_CSV, TEST_SPEC_DIR)
    print(f"   Total samples: {len(dataset)}")
    
    # Select samples - mix of correct and incorrect predictions
    print(f"\n🎯 Selecting {NUM_SAMPLES} diverse samples...")
    
    # Get predictions for all samples first
    predictions = []
    
    with torch.no_grad():
        for idx in tqdm(range(min(500, len(dataset))), desc="Quick scan"):
            spec_tensor, label, chunk_file, species, spec_orig = dataset[idx]
            spec_tensor = spec_tensor.unsqueeze(0).to(device)
            
            outputs = model(spec_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = outputs.argmax(1).item()
            confidence = probs[0, pred_idx].item()
            
            predictions.append({
                'idx': idx,
                'true_label': label,
                'pred_label': pred_idx,
                'correct': label == pred_idx,
                'confidence': confidence,
                'species': species
            })
    
    pred_df = pd.DataFrame(predictions)
    
    # Select diverse samples
    selected_indices = []
    
    # Get some correct high-confidence
    correct_high = pred_df[(pred_df['correct']) & (pred_df['confidence'] > 0.8)]
    selected_indices.extend(correct_high.head(10)['idx'].tolist())
    
    # Get some correct low-confidence
    correct_low = pred_df[(pred_df['correct']) & (pred_df['confidence'] < 0.6)]
    selected_indices.extend(correct_low.head(5)['idx'].tolist())
    
    # Get some incorrect predictions
    incorrect = pred_df[~pred_df['correct']]
    selected_indices.extend(incorrect.head(10)['idx'].tolist())
    
    # Fill remaining with random
    remaining = NUM_SAMPLES - len(selected_indices)
    if remaining > 0:
        available = [i for i in range(len(dataset)) if i not in selected_indices]
        np.random.seed(42)
        selected_indices.extend(np.random.choice(available, min(remaining, len(available)), replace=False).tolist())
    
    selected_indices = selected_indices[:NUM_SAMPLES]
    
    print(f"   Selected {len(selected_indices)} samples")
    
    # Generate Grad-CAM for selected samples
    print(f"\n🎨 Generating Grad-CAM visualizations...")
    
    for i, idx in enumerate(tqdm(selected_indices, desc="Generating")):
        spec_tensor, label, chunk_file, species, spec_orig = dataset[idx]
        spec_tensor = spec_tensor.unsqueeze(0).to(device)
        spec_tensor.requires_grad_(True)
        
        # Forward pass with CAM
        outputs = model(spec_tensor, return_cam=True)
        
        # Get prediction
        probs = torch.softmax(outputs, dim=1)
        pred_idx = outputs.argmax(1).item()
        confidence = probs[0, pred_idx].item()
        
        # Backward pass for Grad-CAM
        model.zero_grad()
        outputs[0, pred_idx].backward()
        
        # Get Grad-CAM
        cam = model.get_gradcam(pred_idx)
        cam = cam[0].detach().cpu().numpy()
        
        # Get species names
        true_species = id_to_species[label]
        pred_species = id_to_species[pred_idx]
        
        # Create visualization
        correct_str = "correct" if label == pred_idx else "incorrect"
        save_name = f"{i+1:02d}_{correct_str}_{chunk_file.replace('.wav', '')}.png"
        save_path = SAMPLES_DIR / save_name
        
        create_gradcam_visualization(
            spectrogram=spec_orig,
            cam=cam,
            true_species=true_species,
            pred_species=pred_species,
            confidence=confidence,
            chunk_file=chunk_file,
            save_path=save_path
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 GRAD-CAM GENERATION COMPLETE!")
    print("=" * 70)
    print(f"""
    📁 Visualizations saved to:
    └── {SAMPLES_DIR}
    
    📊 Generated {len(selected_indices)} visualizations:
    ├── Correct predictions (high confidence)
    ├── Correct predictions (low confidence)
    └── Incorrect predictions
    
    🔍 Each visualization shows:
    ├── Original spectrogram
    ├── Grad-CAM attention heatmap
    └── Overlay showing where model focuses
    
    🎯 Next: Build Streamlit demo!
    """)

if __name__ == "__main__":
    main()