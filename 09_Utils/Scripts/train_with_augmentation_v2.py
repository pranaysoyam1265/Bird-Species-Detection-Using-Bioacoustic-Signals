"""
Script: train_with_augmentation_v2.py
Purpose: Train with FIXED data split (no leakage) + augmentation
Location: Save to 09_Utils/Scripts/train_with_augmentation_v2.py
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import librosa
import timm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION - USING FIXED PATHS
# ============================================================
class Config:
    # Paths - USING FIXED SPLIT
    BASE_DIR = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"
    
    # ⚠️ CHANGED: Using fixed split directory
    TRAIN_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split_Fixed", "train.csv")
    VAL_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split_Fixed", "val.csv")
    TEST_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split_Fixed", "test.csv")
    
    AUDIO_DIR = os.path.join(BASE_DIR, "02_Preprocessed", "Audio_Chunks")
    
    # ⚠️ CHANGED: Using fixed label mapping (flat structure)
    LABEL_MAPPING = os.path.join(BASE_DIR, "04_Labels", "Processed_Labels", "label_mapping_fixed.json")
    
    SAVE_DIR = os.path.join(BASE_DIR, "05_Model", "Saved_Models")
    LOG_DIR = os.path.join(BASE_DIR, "05_Model", "Training_Logs", f"run_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Audio settings
    SAMPLE_RATE = 22050
    DURATION = 5  # seconds
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    FMIN = 150
    FMAX = 15000
    
    # Model settings
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = 54
    PRETRAINED = True
    
    # Training settings - TUNED FOR BETTER GENERALIZATION
    EPOCHS = 50
    BATCH_SIZE = 32  # Increased for better gradient estimates
    LEARNING_RATE = 0.0003  # Lower LR
    WEIGHT_DECAY = 0.05  # Higher regularization
    PATIENCE = 15
    
    # Augmentation
    AUG_PROB = 0.7
    MIXUP_ALPHA = 0.4
    USE_MIXUP = True
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# ============================================================
# AUDIO AUGMENTATION
# ============================================================
class AudioAugmentor:
    def __init__(self, sr=22050, p=0.7):
        self.sr = sr
        self.p = p
    
    def __call__(self, audio):
        if random.random() > self.p:
            return audio
        
        # Apply 1-2 random augmentations
        augmentations = [
            self.time_shift,
            self.pitch_shift,
            self.add_noise,
            self.volume_change,
        ]
        
        n_augs = random.randint(1, 2)
        selected = random.sample(augmentations, n_augs)
        
        for aug in selected:
            try:
                audio = aug(audio)
            except:
                pass
        
        return audio
    
    def time_shift(self, audio):
        shift = int(len(audio) * random.uniform(-0.15, 0.15))
        return np.roll(audio, shift)
    
    def pitch_shift(self, audio):
        n_steps = random.uniform(-2, 2)
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
    
    def add_noise(self, audio):
        noise = np.random.randn(len(audio)) * 0.005 * random.uniform(0.5, 2)
        return audio + noise
    
    def volume_change(self, audio):
        return audio * random.uniform(0.7, 1.3)


# ============================================================
# SPECAUGMENT
# ============================================================
class SpecAugment:
    def __init__(self, freq_mask=15, time_mask=25, n_masks=2, p=0.5):
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_masks = n_masks
        self.p = p
    
    def __call__(self, spec):
        if random.random() > self.p:
            return spec
        
        spec = spec.clone()
        _, n_mels, n_frames = spec.shape
        
        for _ in range(self.n_masks):
            # Frequency mask
            f = random.randint(0, min(self.freq_mask, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            spec[:, f0:f0+f, :] = 0
            
            # Time mask
            t = random.randint(0, min(self.time_mask, n_frames - 1))
            t0 = random.randint(0, n_frames - t)
            spec[:, :, t0:t0+t] = 0
        
        return spec


# ============================================================
# DATASET
# ============================================================
class BirdDataset(Dataset):
    def __init__(self, csv_path, audio_dir, label_mapping_path, config, augment=False):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.config = config
        self.augment = augment
        
        # Load FLAT label mapping
        with open(label_mapping_path, 'r') as f:
            self.label_map = json.load(f)
        
        # Verify it's flat (species -> int)
        sample_key = list(self.label_map.keys())[0]
        sample_val = self.label_map[sample_key]
        if isinstance(sample_val, dict):
            # It's nested, extract species_to_id
            self.label_map = self.label_map.get('species_to_id', self.label_map)
        
        if augment:
            self.audio_aug = AudioAugmentor(sr=config.SAMPLE_RATE, p=config.AUG_PROB)
            self.spec_aug = SpecAugment(p=0.5)
        else:
            self.audio_aug = None
            self.spec_aug = None
        
        print(f"   Loaded {len(self.df)} samples, Augment={augment}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['chunk_file']
        species = row['species']
        
        # Load audio
        filepath = os.path.join(self.audio_dir, filename)
        try:
            audio, sr = librosa.load(filepath, sr=self.config.SAMPLE_RATE, mono=True)
        except:
            audio = np.zeros(self.config.SAMPLE_RATE * self.config.DURATION)
        
        # Ensure correct length
        target_len = self.config.SAMPLE_RATE * self.config.DURATION
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        
        # Audio augmentation
        if self.audio_aug:
            audio = self.audio_aug(audio)
        
        # Generate spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.config.SAMPLE_RATE,
            n_mels=self.config.N_MELS, n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            fmin=self.config.FMIN, fmax=self.config.FMAX
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        
        # To 3-channel tensor
        spec = torch.FloatTensor(np.stack([mel_norm]*3))
        
        # Spec augmentation
        if self.spec_aug:
            spec = self.spec_aug(spec)
        
        # Label
        label = self.label_map.get(species, 0)
        
        return spec, label


# ============================================================
# MODEL
# ============================================================
class BirdClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.backbone = timm.create_model(
            config.MODEL_NAME,
            pretrained=config.PRETRAINED,
            num_classes=0,
            drop_rate=0.3,
            drop_path_rate=0.2,
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 216)
            feat_dim = self.backbone(dummy).shape[1]
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.NUM_CLASSES)
        )
    
    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)


# ============================================================
# MIXUP
# ============================================================
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# TRAINING
# ============================================================
def train_epoch(model, loader, criterion, optimizer, scaler, config):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Train")
    for data, target in pbar:
        data, target = data.to(config.DEVICE), target.to(config.DEVICE)
        
        optimizer.zero_grad()
        
        # Mixup
        if config.USE_MIXUP and random.random() < 0.5:
            data, y_a, y_b, lam = mixup_data(data, target, config.MIXUP_ALPHA)
            with autocast():
                output = model(data)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            _, pred = output.max(1)
            correct += (lam * pred.eq(y_a).sum().item() + (1-lam) * pred.eq(y_b).sum().item())
        else:
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total += target.size(0)
        pbar.set_postfix({'loss': f'{total_loss/total:.4f}', 'acc': f'{100*correct/total:.1f}%'})
    
    return total_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion, config):
    model.eval()
    total_loss = 0
    correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Val"):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            _, top5 = output.topk(5, dim=1)
            for i, t in enumerate(target):
                if t in top5[i]:
                    top5_correct += 1
    
    return total_loss / len(loader), 100 * correct / total, 100 * top5_correct / total


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("🐦 TRAINING WITH FIXED DATA SPLIT (NO LEAKAGE)")
    print("=" * 70)
    
    config = Config()
    set_seed(config.SEED)
    
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    print(f"\n📁 Using FIXED splits:")
    print(f"   Train: {config.TRAIN_CSV}")
    print(f"   Val:   {config.VAL_CSV}")
    print(f"   Labels: {config.LABEL_MAPPING}")
    print(f"   Device: {config.DEVICE}")
    
    # Verify files exist
    for path in [config.TRAIN_CSV, config.VAL_CSV, config.LABEL_MAPPING]:
        if not os.path.exists(path):
            print(f"\n❌ File not found: {path}")
            print("   Run fix_data_split.py first!")
            return
    
    # Load data
    print("\n📂 Loading datasets...")
    train_ds = BirdDataset(config.TRAIN_CSV, config.AUDIO_DIR, config.LABEL_MAPPING, config, augment=True)
    val_ds = BirdDataset(config.VAL_CSV, config.AUDIO_DIR, config.LABEL_MAPPING, config, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    print("\n🏗️  Building model...")
    model = BirdClassifier(config).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()
    
    # Training loop
    print("\n" + "=" * 70)
    print("🚀 TRAINING")
    print("=" * 70)
    
    best_val_acc = 0
    best_top5 = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_top5': []}
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n{'─'*70}")
        print(f"Epoch {epoch}/{config.EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, config)
        val_loss, val_acc, val_top5 = validate(model, val_loader, criterion, config)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_top5'].append(val_top5)
        
        gap = train_acc - val_acc
        print(f"\n📊 Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Top5: {val_top5:.2f}% | Gap: {gap:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_top5 = val_top5
            patience_counter = 0
            save_path = os.path.join(config.SAVE_DIR, "best_model_v2.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_top5': val_top5,
            }, save_path)
            print(f"   ✅ NEW BEST! Saved to {save_path}")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{config.PATIENCE})")
        
        if patience_counter >= config.PATIENCE:
            print(f"\n⚠️ Early stopping at epoch {epoch}")
            break
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n📊 Best Results:")
    print(f"   Val Accuracy: {best_val_acc:.2f}%")
    print(f"   Top-5 Accuracy: {best_top5:.2f}%")
    
    # Save history
    with open(os.path.join(config.LOG_DIR, "history.json"), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()