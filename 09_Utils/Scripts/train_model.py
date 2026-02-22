"""
train_model.py
Bird Species Classification Training

Optimized for RTX 3050 (4GB VRAM)
- EfficientNet-B0 backbone
- Mixed precision training (FP16)
- Online augmentation
- Early stopping
- TensorBoard logging

Author: Bird Detection Project
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import timm
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")

# Paths
TRAIN_CSV = BASE_DIR / "04_Labels" / "Train_Val_Test_Split" / "train.csv"
VAL_CSV = BASE_DIR / "04_Labels" / "Train_Val_Test_Split" / "val.csv"
TRAIN_SPEC_DIR = BASE_DIR / "03_Features" / "Spectrograms" / "train"
VAL_SPEC_DIR = BASE_DIR / "03_Features" / "Spectrograms" / "val"
LABEL_MAPPING = BASE_DIR / "04_Labels" / "Processed_Labels" / "label_mapping.json"
CLASS_WEIGHTS_FILE = BASE_DIR / "04_Labels" / "Processed_Labels" / "class_weights.json"

# Output paths
MODEL_DIR = BASE_DIR / "05_Model"
CHECKPOINTS_DIR = MODEL_DIR / "Checkpoints"
SAVED_MODELS_DIR = MODEL_DIR / "Saved_Models"
LOGS_DIR = MODEL_DIR / "Training_Logs"
CONFIGS_DIR = MODEL_DIR / "Configs"

# Training settings (Optimized for RTX 3050 4GB)
CONFIG = {
    # Model
    'model_name': 'efficientnet_b0',
    'pretrained': True,
    'num_classes': 54,
    
    # Training
    'batch_size': 16,           # Safe for 4GB VRAM
    'epochs': 30,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'label_smoothing': 0.1,
    
    # Early stopping
    'patience': 7,
    'min_delta': 0.001,
    
    # Mixed precision
    'use_amp': True,            # Automatic Mixed Precision
    
    # Data
    'num_workers': 4,
    'pin_memory': True,
    
    # Augmentation
    'use_augmentation': True,
    'mixup_alpha': 0.2,
    'spec_augment': True,
    
    # Input shape
    'n_mels': 128,
    'time_steps': 216,          # Approximate
}

# ============================================================
# DATASET
# ============================================================

class BirdSpectrogramDataset(Dataset):
    """Dataset for loading spectrograms"""
    
    def __init__(self, csv_path, spec_dir, augment=False):
        self.df = pd.read_csv(csv_path)
        self.spec_dir = Path(spec_dir)
        self.augment = augment
        
        # Verify files exist
        valid_rows = []
        for idx, row in self.df.iterrows():
            spec_file = row['chunk_file'].replace('.wav', '.npy')
            if (self.spec_dir / spec_file).exists():
                valid_rows.append(idx)
        
        self.df = self.df.loc[valid_rows].reset_index(drop=True)
        print(f"Dataset: {len(self.df)} valid samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load spectrogram
        spec_file = row['chunk_file'].replace('.wav', '.npy')
        spec_path = self.spec_dir / spec_file
        
        spec = np.load(spec_path)  # Shape: (n_mels, time)
        
        # Apply augmentation
        if self.augment:
            spec = self._augment(spec)
        
        # Ensure consistent shape
        spec = self._pad_or_crop(spec, target_time=CONFIG['time_steps'])
        
        # Convert to tensor (add channel dimension)
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0)  # Shape: (1, n_mels, time)
        
        # Repeat to 3 channels for pretrained model
        spec_tensor = spec_tensor.repeat(3, 1, 1)  # Shape: (3, n_mels, time)
        
        # Get label
        label = row['label_id']
        
        return spec_tensor, label
    
    def _pad_or_crop(self, spec, target_time):
        """Ensure spectrogram has consistent time dimension"""
        current_time = spec.shape[1]
        
        if current_time < target_time:
            # Pad with zeros
            padding = target_time - current_time
            spec = np.pad(spec, ((0, 0), (0, padding)), mode='constant')
        elif current_time > target_time:
            # Crop
            spec = spec[:, :target_time]
        
        return spec
    
    def _augment(self, spec):
        """Apply spectrogram augmentation"""
        # Time masking
        if np.random.random() < 0.5:
            spec = self._time_mask(spec)
        
        # Frequency masking
        if np.random.random() < 0.5:
            spec = self._freq_mask(spec)
        
        # Add noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, spec.shape)
            spec = spec + noise
            spec = np.clip(spec, 0, 1)
        
        return spec
    
    def _time_mask(self, spec, max_mask=30):
        """Mask random time steps"""
        time_steps = spec.shape[1]
        mask_length = np.random.randint(1, min(max_mask, time_steps // 4))
        mask_start = np.random.randint(0, time_steps - mask_length)
        spec[:, mask_start:mask_start + mask_length] = 0
        return spec
    
    def _freq_mask(self, spec, max_mask=20):
        """Mask random frequency bands"""
        n_mels = spec.shape[0]
        mask_length = np.random.randint(1, min(max_mask, n_mels // 4))
        mask_start = np.random.randint(0, n_mels - mask_length)
        spec[mask_start:mask_start + mask_length, :] = 0
        return spec


# ============================================================
# MODEL
# ============================================================

class BirdClassifier(nn.Module):
    """EfficientNet-based bird classifier"""
    
    def __init__(self, model_name='efficientnet_b0', num_classes=54, pretrained=True):
        super().__init__()
        
        # Load pretrained EfficientNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool=''  # Remove global pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 128, 216)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.flatten(1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits
    
    def get_features(self, x):
        """Get feature maps for Grad-CAM"""
        features = self.backbone(x)
        return features


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def create_directories():
    """Create output directories"""
    for dir_path in [CHECKPOINTS_DIR, SAVED_MODELS_DIR, LOGS_DIR, CONFIGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

def get_class_weights(device):
    """Load class weights for imbalanced data"""
    if CLASS_WEIGHTS_FILE.exists():
        with open(CLASS_WEIGHTS_FILE, 'r') as f:
            weights_dict = json.load(f)
        
        # Load label mapping to get correct order
        with open(LABEL_MAPPING, 'r') as f:
            label_map = json.load(f)
        
        # Create weight tensor in correct order
        num_classes = label_map['num_classes']
        weights = torch.ones(num_classes)
        
        for species, weight in weights_dict.items():
            if species in label_map['species_to_id']:
                idx = label_map['species_to_id'][species]
                weights[idx] = weight
        
        # Normalize weights
        weights = weights / weights.sum() * num_classes
        
        return weights.to(device)
    
    return None

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for specs, labels in pbar:
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast(enabled=CONFIG['use_amp']):
            outputs = model(specs)
            loss = criterion(outputs, labels)
        
        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item() * specs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for specs, labels in tqdm(loader, desc="Validating", leave=False):
            specs = specs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast(enabled=CONFIG['use_amp']):
                outputs = model(specs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * specs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


# ============================================================
# MAIN TRAINING
# ============================================================

def main():
    print("=" * 70)
    print("🐦 BIRD SPECIES CLASSIFIER - TRAINING")
    print("=" * 70)
    
    # Create directories
    create_directories()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load label mapping
    with open(LABEL_MAPPING, 'r') as f:
        label_map = json.load(f)
    
    CONFIG['num_classes'] = label_map['num_classes']
    print(f"\n📊 Classes: {CONFIG['num_classes']}")
    
    # Create datasets
    print("\n📂 Loading datasets...")
    train_dataset = BirdSpectrogramDataset(
        TRAIN_CSV, TRAIN_SPEC_DIR, augment=CONFIG['use_augmentation']
    )
    val_dataset = BirdSpectrogramDataset(
        VAL_CSV, VAL_SPEC_DIR, augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Create model
    print(f"\n🧠 Creating model: {CONFIG['model_name']}")
    model = BirdClassifier(
        model_name=CONFIG['model_name'],
        num_classes=CONFIG['num_classes'],
        pretrained=CONFIG['pretrained']
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss function with class weights
    class_weights = get_class_weights(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=CONFIG['label_smoothing']
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=CONFIG['use_amp'])
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(LOGS_DIR / f"run_{timestamp}")
    
    # Save config
    config_path = CONFIGS_DIR / f"config_{timestamp}.json"
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # Training loop
    print(f"\n🚀 Starting training for {CONFIG['epochs']} epochs...")
    print("-" * 70)
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log to TensorBoard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch results
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': CONFIG,
                'label_mapping': label_map
            }
            
            best_model_path = SAVED_MODELS_DIR / "best_model.pth"
            torch.save(checkpoint, best_model_path)
            print(f"   ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{CONFIG['patience']})")
        
        # Early stopping
        if patience_counter >= CONFIG['patience']:
            print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = CHECKPOINTS_DIR / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, checkpoint_path)
    
    # Close TensorBoard
    writer.close()
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"""
    📊 Best Results:
    ├── Best Val Accuracy: {best_val_acc:.2f}%
    ├── Best Val Loss: {best_val_loss:.4f}
    │
    📁 Saved Files:
    ├── Best Model: {SAVED_MODELS_DIR / 'best_model.pth'}
    ├── Config: {config_path}
    ├── TensorBoard Logs: {LOGS_DIR / f'run_{timestamp}'}
    │
    🎯 Next Steps:
    ├── 1. Run evaluation on test set
    ├── 2. Generate Grad-CAM visualizations
    └── 3. Build Streamlit demo
    """)
    
    # View TensorBoard
    print(f"\n📈 To view training progress, run:")
    print(f"   tensorboard --logdir=\"{LOGS_DIR}\"")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()