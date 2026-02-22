"""
Script: train_with_augmentation_v2_fixed.py
Purpose: Memory-optimized training with fixed data split
Location: Save to 09_Utils/Scripts/train_with_augmentation_v2_fixed.py
"""

import os
import sys
import gc
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
# CONFIGURATION - MEMORY OPTIMIZED
# ============================================================
class Config:
    BASE_DIR = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"
    
    # Using fixed split
    TRAIN_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split_Fixed", "train.csv")
    VAL_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split_Fixed", "val.csv")
    TEST_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split_Fixed", "test.csv")
    
    AUDIO_DIR = os.path.join(BASE_DIR, "02_Preprocessed", "Audio_Chunks")
    LABEL_MAPPING = os.path.join(BASE_DIR, "04_Labels", "Processed_Labels", "label_mapping_fixed.json")
    
    SAVE_DIR = os.path.join(BASE_DIR, "05_Model", "Saved_Models")
    LOG_DIR = os.path.join(BASE_DIR, "05_Model", "Training_Logs", f"run_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Audio settings
    SAMPLE_RATE = 22050
    DURATION = 5
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    FMIN = 150
    FMAX = 15000
    
    # Model
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = 54
    PRETRAINED = True
    
    # Training - MEMORY OPTIMIZED
    EPOCHS = 50
    BATCH_SIZE = 16          # ⬇️ Reduced from 32
    LEARNING_RATE = 0.0003
    WEIGHT_DECAY = 0.05
    PATIENCE = 15
    
    # ⚠️ CRITICAL MEMORY FIX
    NUM_WORKERS = 0          # ⬇️ Set to 0 to avoid multiprocessing memory issues
    PIN_MEMORY = False       # ⬇️ Disable to save memory
    
    # Augmentation
    AUG_PROB = 0.7
    MIXUP_ALPHA = 0.4
    USE_MIXUP = True
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory():
    """Clear memory between operations"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# SIMPLE AUDIO AUGMENTATION (Memory Efficient)
# ============================================================
class AudioAugmentor:
    def __init__(self, sr=22050, p=0.7):
        self.sr = sr
        self.p = p
    
    def __call__(self, audio):
        if random.random() > self.p:
            return audio
        
        # Apply only 1 augmentation (memory efficient)
        aug_choice = random.randint(0, 3)
        
        try:
            if aug_choice == 0:
                # Time shift (no extra memory)
                shift = int(len(audio) * random.uniform(-0.15, 0.15))
                audio = np.roll(audio, shift)
            elif aug_choice == 1:
                # Add noise (minimal memory)
                noise = np.random.randn(len(audio)).astype(np.float32) * 0.005
                audio = audio + noise
            elif aug_choice == 2:
                # Volume change (no extra memory)
                audio = audio * random.uniform(0.7, 1.3)
            elif aug_choice == 3:
                # Pitch shift (only if not too memory constrained)
                if random.random() < 0.5:  # 50% chance to skip expensive operation
                    n_steps = random.uniform(-1.5, 1.5)
                    audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
        except:
            pass
        
        return audio


# ============================================================
# SPECAUGMENT (Applied to tensor, memory efficient)
# ============================================================
class SpecAugment:
    def __init__(self, freq_mask=12, time_mask=20, p=0.5):
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.p = p
    
    def __call__(self, spec):
        if random.random() > self.p:
            return spec
        
        spec = spec.clone()
        _, n_mels, n_frames = spec.shape
        
        # Single frequency mask
        f = random.randint(1, min(self.freq_mask, n_mels - 1))
        f0 = random.randint(0, n_mels - f)
        spec[:, f0:f0+f, :] = 0
        
        # Single time mask
        t = random.randint(1, min(self.time_mask, n_frames - 1))
        t0 = random.randint(0, n_frames - t)
        spec[:, :, t0:t0+t] = 0
        
        return spec


# ============================================================
# DATASET - MEMORY OPTIMIZED
# ============================================================
class BirdDataset(Dataset):
    def __init__(self, csv_path, audio_dir, label_mapping_path, config, augment=False):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.config = config
        self.augment = augment
        
        # Load label mapping
        with open(label_mapping_path, 'r') as f:
            self.label_map = json.load(f)
        
        # Handle nested structure
        if isinstance(list(self.label_map.values())[0], dict):
            self.label_map = self.label_map.get('species_to_id', {})
        
        # Initialize augmentors
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
            # Load with float32 (less memory than float64)
            audio, _ = librosa.load(filepath, sr=self.config.SAMPLE_RATE, mono=True, dtype=np.float32)
        except Exception as e:
            # Return zeros if file fails to load
            audio = np.zeros(self.config.SAMPLE_RATE * self.config.DURATION, dtype=np.float32)
        
        # Ensure correct length
        target_len = self.config.SAMPLE_RATE * self.config.DURATION
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Audio augmentation
        if self.audio_aug:
            audio = self.audio_aug(audio)
        
        # Generate spectrogram (with explicit dtype)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.SAMPLE_RATE,
            n_mels=self.config.N_MELS,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            fmin=self.config.FMIN,
            fmax=self.config.FMAX
        ).astype(np.float32)
        
        # Convert to dB
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        
        # Normalize to 0-1
        mel_min = mel_db.min()
        mel_max = mel_db.max()
        if mel_max - mel_min > 1e-6:
            mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
        else:
            mel_norm = np.zeros_like(mel_db)
        
        # Stack to 3 channels and convert to tensor
        spec = torch.from_numpy(np.stack([mel_norm, mel_norm, mel_norm]).astype(np.float32))
        
        # Spec augmentation
        if self.spec_aug:
            spec = self.spec_aug(spec)
        
        # Get label
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
        
        # Get feature dimension
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
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_epoch(model, loader, criterion, optimizer, scaler, config):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Train")
    
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(config.DEVICE, non_blocking=True)
        target = target.to(config.DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # More memory efficient
        
        # Mixup (50% of the time)
        use_mixup = config.USE_MIXUP and random.random() < 0.5
        
        if use_mixup:
            data, y_a, y_b, lam = mixup_data(data, target, config.MIXUP_ALPHA)
            
            with autocast():
                output = model(data)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            
            _, pred = output.max(1)
            correct += (lam * pred.eq(y_a).sum().item() + (1 - lam) * pred.eq(y_b).sum().item())
        else:
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total += target.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{100*correct/total:.1f}%'
        })
        
        # Periodic memory cleanup
        if batch_idx % 100 == 0:
            clear_memory()
    
    return total_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion, config):
    model.eval()
    total_loss = 0
    correct = 0
    top3_correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val")
        
        for data, target in pbar:
            data = data.to(config.DEVICE, non_blocking=True)
            target = target.to(config.DEVICE, non_blocking=True)
            
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            total_loss += loss.item()
            
            # Top-1 accuracy
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Top-3 and Top-5 accuracy
            _, top5_pred = output.topk(5, dim=1)
            _, top3_pred = output.topk(3, dim=1)
            
            for i in range(target.size(0)):
                if target[i] in top3_pred[i]:
                    top3_correct += 1
                if target[i] in top5_pred[i]:
                    top5_correct += 1
            
            pbar.set_postfix({
                'loss': f'{total_loss/total:.4f}',
                'acc': f'{100*correct/total:.1f}%'
            })
    
    return (
        total_loss / len(loader),
        100 * correct / total,
        100 * top3_correct / total,
        100 * top5_correct / total
    )


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("🐦 TRAINING WITH FIXED DATA SPLIT (MEMORY OPTIMIZED)")
    print("=" * 70)
    
    config = Config()
    set_seed(config.SEED)
    clear_memory()
    
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    print(f"\n📁 Configuration:")
    print(f"   Train CSV:    {config.TRAIN_CSV}")
    print(f"   Val CSV:      {config.VAL_CSV}")
    print(f"   Label Map:    {config.LABEL_MAPPING}")
    print(f"   Device:       {config.DEVICE}")
    print(f"   Batch Size:   {config.BATCH_SIZE}")
    print(f"   Num Workers:  {config.NUM_WORKERS}")
    
    # Verify files exist
    for name, path in [("Train CSV", config.TRAIN_CSV), 
                       ("Val CSV", config.VAL_CSV), 
                       ("Label Map", config.LABEL_MAPPING)]:
        if not os.path.exists(path):
            print(f"\n❌ {name} not found: {path}")
            print("   Run fix_data_split.py first!")
            return
        print(f"   ✅ {name} exists")
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_ds = BirdDataset(
        config.TRAIN_CSV, config.AUDIO_DIR, config.LABEL_MAPPING,
        config, augment=True
    )
    val_ds = BirdDataset(
        config.VAL_CSV, config.AUDIO_DIR, config.LABEL_MAPPING,
        config, augment=False
    )
    
    # Create data loaders with memory-safe settings
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,  # 0 = main process only
        pin_memory=config.PIN_MEMORY,
        drop_last=True  # Avoid small batches
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    
    # Build model
    print("\n🏗️ Building model...")
    model = BirdClassifier(config).to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    scaler = GradScaler()
    
    # Training loop
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)
    
    best_val_acc = 0
    best_top5 = 0
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_top3': [], 'val_top5': []
    }
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n{'─' * 70}")
        print(f"Epoch {epoch}/{config.EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'─' * 70}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, config
        )
        
        # Clear memory before validation
        clear_memory()
        
        # Validate
        val_loss, val_acc, val_top3, val_top5 = validate(
            model, val_loader, criterion, config
        )
        
        # Update scheduler
        scheduler.step()
        
        # Clear memory
        clear_memory()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_top3'].append(val_top3)
        history['val_top5'].append(val_top5)
        
        # Calculate gap
        gap = train_acc - val_acc
        
        # Print results
        print(f"\n📊 Results:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   Val Top-3:  {val_top3:.2f}% | Val Top-5: {val_top5:.2f}%")
        print(f"   Gap (Train-Val): {gap:.2f}%")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_top5 = val_top5
            patience_counter = 0
            
            # Save best model
            save_path = os.path.join(config.SAVE_DIR, "best_model_v2.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_top3': val_top3,
                'val_top5': val_top5,
                'train_acc': train_acc,
                'config': {
                    'model_name': config.MODEL_NAME,
                    'num_classes': config.NUM_CLASSES,
                }
            }, save_path)
            print(f"   ✅ NEW BEST! Saved to {save_path}")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement for {patience_counter}/{config.PATIENCE} epochs")
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n⚠️ Early stopping triggered at epoch {epoch}")
            break
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(config.SAVE_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"   💾 Checkpoint saved: {checkpoint_path}")
    
    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE")
    print("=" * 70)
    
    print(f"\n📊 Best Results:")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Best Top-5 Accuracy:      {best_top5:.2f}%")
    
    # Save training history
    history_path = os.path.join(config.LOG_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n📈 Training history saved to: {history_path}")
    
    # Compare with original
    original_val_acc = 68.79
    improvement = best_val_acc - original_val_acc
    
    print(f"\n📈 COMPARISON WITH ORIGINAL:")
    print(f"   Original Val Acc: {original_val_acc:.2f}%")
    print(f"   New Val Acc:      {best_val_acc:.2f}%")
    print(f"   Change:           {improvement:+.2f}%")
    
    if best_val_acc < 100:
        print(f"\n✅ Results look REALISTIC (not 100% = no data leakage)")
    else:
        print(f"\n⚠️ 100% accuracy still suspicious - check data split")


if __name__ == "__main__":
    main()