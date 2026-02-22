"""
Script: train_with_augmentation.py
Purpose: Retrain model with comprehensive audio augmentation
Location: Save to 09_Utils/Scripts/train_with_augmentation.py

EXPECTED IMPROVEMENT: +8-12% validation accuracy
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
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # Paths
    BASE_DIR = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"
    TRAIN_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split", "train.csv")
    VAL_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split", "val.csv")
    AUDIO_DIR = os.path.join(BASE_DIR, "02_Preprocessed", "Audio_Chunks")
    LABEL_MAPPING = os.path.join(BASE_DIR, "04_Labels", "Processed_Labels", "label_mapping.json")
    SAVE_DIR = os.path.join(BASE_DIR, "05_Model", "Saved_Models")
    LOG_DIR = os.path.join(BASE_DIR, "05_Model", "Training_Logs", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Audio settings
    SAMPLE_RATE = 22050
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    FMIN = 150
    FMAX = 15000
    
    # Model settings
    MODEL_NAME = "efficientnet_b0"  # Can change to "efficientnet_b2" for larger model
    NUM_CLASSES = 54
    PRETRAINED = True
    
    # Training settings
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0005  # Reduced from 0.001
    WEIGHT_DECAY = 0.01     # Increased from 0.0001
    PATIENCE = 10           # Early stopping patience
    
    # Augmentation probabilities
    AUG_PROB = 0.8          # Probability of applying augmentation
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Random seed
    SEED = 42


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# AUDIO AUGMENTATION CLASS
# ============================================================
class AudioAugmentor:
    """
    Comprehensive audio augmentation for bird sound classification.
    Simulates real-world recording variations.
    """
    
    def __init__(self, sample_rate=22050, p=0.8):
        self.sr = sample_rate
        self.p = p  # Overall probability of augmentation
    
    def __call__(self, audio):
        """Apply random augmentations to audio"""
        if random.random() > self.p:
            return audio
        
        # Randomly select 1-3 augmentations
        augmentations = [
            self.time_shift,
            self.pitch_shift,
            self.speed_change,
            self.add_noise,
            self.volume_change,
            self.time_mask,
        ]
        
        n_augs = random.randint(1, 3)
        selected_augs = random.sample(augmentations, n_augs)
        
        for aug in selected_augs:
            audio = aug(audio)
        
        return audio
    
    def time_shift(self, audio, max_shift=0.2):
        """Shift audio in time (simulates different recording start points)"""
        shift = int(len(audio) * random.uniform(-max_shift, max_shift))
        return np.roll(audio, shift)
    
    def pitch_shift(self, audio, max_semitones=2):
        """Shift pitch (simulates individual bird variation)"""
        n_steps = random.uniform(-max_semitones, max_semitones)
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
    
    def speed_change(self, audio, min_rate=0.9, max_rate=1.1):
        """Change speed without changing pitch (simulates tempo variation)"""
        rate = random.uniform(min_rate, max_rate)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def add_noise(self, audio, min_snr=10, max_snr=30):
        """Add Gaussian noise (simulates environmental noise)"""
        snr = random.uniform(min_snr, max_snr)
        audio_power = np.mean(audio ** 2)
        noise_power = audio_power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise
    
    def volume_change(self, audio, min_gain=0.5, max_gain=1.5):
        """Change volume (simulates distance from microphone)"""
        gain = random.uniform(min_gain, max_gain)
        return audio * gain
    
    def time_mask(self, audio, max_mask_ratio=0.1):
        """Mask random time segment (simulates brief interruptions)"""
        mask_length = int(len(audio) * random.uniform(0, max_mask_ratio))
        mask_start = random.randint(0, len(audio) - mask_length)
        audio_copy = audio.copy()
        audio_copy[mask_start:mask_start + mask_length] = 0
        return audio_copy


# ============================================================
# SPECAUGMENT (Applied to Spectrogram)
# ============================================================
class SpecAugment:
    """
    SpecAugment: Augmentation directly on spectrograms.
    Paper: https://arxiv.org/abs/1904.08779
    """
    
    def __init__(self, freq_mask_param=20, time_mask_param=30, n_freq_masks=2, n_time_masks=2, p=0.5):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p
    
    def __call__(self, spec):
        """Apply SpecAugment to spectrogram (C, H, W) tensor"""
        if random.random() > self.p:
            return spec
        
        spec = spec.clone()
        _, n_mels, n_frames = spec.shape
        
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            spec[:, f0:f0 + f, :] = 0
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = random.randint(0, min(self.time_mask_param, n_frames - 1))
            t0 = random.randint(0, n_frames - t)
            spec[:, :, t0:t0 + t] = 0
        
        return spec


# ============================================================
# MIXUP AUGMENTATION
# ============================================================
def mixup_data(x, y, alpha=0.4):
    """
    Mixup: Mix two samples and their labels.
    Paper: https://arxiv.org/abs/1710.09412
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# DATASET CLASS WITH AUGMENTATION
# ============================================================
class BirdAudioDataset(Dataset):
    """
    Dataset with on-the-fly audio loading and augmentation.
    """
    
    def __init__(self, csv_path, audio_dir, label_mapping, config, augment=False):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.config = config
        self.augment = augment
        
        # Load label mapping
        with open(label_mapping, 'r') as f:
            self.label_map = json.load(f)
        
        # Detect column names
        self.file_col = self._detect_column(['filename', 'file', 'filepath', 'path', 'chunk_file'])
        self.species_col = self._detect_column(['species', 'label', 'class', 'scientific_name'])
        
        # Initialize augmentors
        if augment:
            self.audio_aug = AudioAugmentor(sample_rate=config.SAMPLE_RATE, p=config.AUG_PROB)
            self.spec_aug = SpecAugment(p=0.5)
        else:
            self.audio_aug = None
            self.spec_aug = None
        
        print(f"   Dataset loaded: {len(self.df)} samples, Augment: {augment}")
    
    def _detect_column(self, candidates):
        for col in candidates:
            if col in self.df.columns:
                return col
        raise ValueError(f"Could not find column from {candidates}")
    
    def __len__(self):
        return len(self.df)
    
    def _find_audio_file(self, filename):
        """Search for audio file in various locations"""
        # Direct path
        path = os.path.join(self.audio_dir, filename)
        if os.path.exists(path):
            return path
        
        # Search in subdirectories
        for root, dirs, files in os.walk(self.audio_dir):
            if filename in files:
                return os.path.join(root, filename)
        
        # Try with different extensions
        base = os.path.splitext(filename)[0]
        for ext in ['.wav', '.mp3', '.ogg', '.flac']:
            path = os.path.join(self.audio_dir, base + ext)
            if os.path.exists(path):
                return path
        
        return None
    
    def _load_audio(self, filepath):
        """Load audio file and ensure consistent length"""
        try:
            audio, sr = librosa.load(filepath, sr=self.config.SAMPLE_RATE, mono=True)
            
            # Ensure 5 seconds
            target_length = self.config.SAMPLE_RATE * 5
            
            if len(audio) < target_length:
                # Pad with zeros
                audio = np.pad(audio, (0, target_length - len(audio)))
            elif len(audio) > target_length:
                # Trim
                audio = audio[:target_length]
            
            return audio
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return np.zeros(self.config.SAMPLE_RATE * 5)
    
    def _audio_to_spectrogram(self, audio):
        """Convert audio to mel spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.SAMPLE_RATE,
            n_mels=self.config.N_MELS,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            fmin=self.config.FMIN,
            fmax=self.config.FMAX,
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to 0-1
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Ensure consistent time dimension (pad or crop to 216 frames)
        target_time_steps = 216
        current_time_steps = mel_spec_norm.shape[1]
        
        if current_time_steps < target_time_steps:
            # Pad with the last frame
            pad_amount = target_time_steps - current_time_steps
            mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_amount)), mode='edge')
        elif current_time_steps > target_time_steps:
            # Crop to target size
            mel_spec_norm = mel_spec_norm[:, :target_time_steps]
        
        # Convert to 3-channel image (for pretrained model)
        spec_3ch = np.stack([mel_spec_norm, mel_spec_norm, mel_spec_norm], axis=0)
        
        return torch.FloatTensor(spec_3ch)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row[self.file_col]
        species = row[self.species_col]
        
        # Find and load audio
        filepath = self._find_audio_file(filename)
        if filepath is None:
            # Return zeros if file not found
            audio = np.zeros(self.config.SAMPLE_RATE * 5)
        else:
            audio = self._load_audio(filepath)
        
        # Apply audio augmentation
        if self.audio_aug is not None:
            audio = self.audio_aug(audio)
        
        # Convert to spectrogram
        spectrogram = self._audio_to_spectrogram(audio)
        
        # Apply SpecAugment
        if self.spec_aug is not None:
            spectrogram = self.spec_aug(spectrogram)
        
        # Get label
        label = self.label_map.get(species, 0)
        
        return spectrogram, label


# ============================================================
# MODEL WITH STRONGER REGULARIZATION
# ============================================================
class BirdClassifier(nn.Module):
    """
    EfficientNet-based classifier with improved regularization.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Load pretrained model
        self.backbone = timm.create_model(
            config.MODEL_NAME,
            pretrained=config.PRETRAINED,
            num_classes=0,  # Remove classifier
            drop_rate=0.3,  # Dropout in backbone
            drop_path_rate=0.2,  # Stochastic depth
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 216)
            features = self.backbone(dummy)
            feature_dim = features.shape[1]
        
        # Custom classifier with stronger regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Increased from 0.3
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, config.NUM_CLASSES)
        )
        
        print(f"   Model: {config.MODEL_NAME}")
        print(f"   Feature dim: {feature_dim}")
        print(f"   Classes: {config.NUM_CLASSES}")
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ============================================================
# FOCAL LOSS (Handles Class Imbalance)
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Paper: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================
# LABEL SMOOTHING CROSS ENTROPY
# ============================================================
class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(1)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * nn.functional.log_softmax(pred, dim=1), dim=1))


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_epoch(model, train_loader, criterion, optimizer, scaler, config, use_mixup=True):
    """Train for one epoch with optional mixup"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(config.DEVICE), target.to(config.DEVICE)
        
        optimizer.zero_grad()
        
        # Apply mixup
        if use_mixup and random.random() < 0.5:
            data, target_a, target_b, lam = mixup_data(data, target, alpha=0.4)
            
            with autocast():
                output = model(data)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            
            # For accuracy, use original target
            _, predicted = output.max(1)
            correct += (lam * predicted.eq(target_a).sum().item() + 
                       (1 - lam) * predicted.eq(target_b).sum().item())
        else:
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total += target.size(0)
        
        pbar.set_postfix({
            'loss': f'{total_loss / (batch_idx + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, config):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    top3_correct = 0
    top5_correct = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validating"):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)
            
            # Top-3 and Top-5 accuracy
            _, top5_pred = output.topk(5, dim=1)
            _, top3_pred = output.topk(3, dim=1)
            
            for i, t in enumerate(target):
                if t in top3_pred[i]:
                    top3_correct += 1
                if t in top5_pred[i]:
                    top5_correct += 1
    
    return (
        total_loss / len(val_loader),
        100. * correct / total,
        100. * top3_correct / total,
        100. * top5_correct / total
    )


# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def main():
    print("=" * 70)
    print("🐦 BIRD DETECTION - TRAINING WITH AUGMENTATION")
    print("=" * 70)
    
    config = Config()
    set_seed(config.SEED)
    
    # Create directories
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    print(f"\n📁 Base directory: {config.BASE_DIR}")
    print(f"💾 Save directory: {config.SAVE_DIR}")
    print(f"📊 Log directory:  {config.LOG_DIR}")
    print(f"🖥️  Device: {config.DEVICE}")
    
    # ---------------------------------------------------------
    # Load datasets
    # ---------------------------------------------------------
    print("\n📂 Loading datasets...")
    
    train_dataset = BirdAudioDataset(
        config.TRAIN_CSV, config.AUDIO_DIR, config.LABEL_MAPPING, 
        config, augment=True  # ← AUGMENTATION ENABLED
    )
    
    val_dataset = BirdAudioDataset(
        config.VAL_CSV, config.AUDIO_DIR, config.LABEL_MAPPING,
        config, augment=False  # ← NO augmentation for validation
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    
    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------
    print("\n🏗️  Building model...")
    model = BirdClassifier(config).to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # ---------------------------------------------------------
    # Loss function and optimizer
    # ---------------------------------------------------------
    print("\n⚙️  Setting up training...")
    
    # Option 1: Focal Loss (good for class imbalance)
    # criterion = FocalLoss(gamma=2)
    
    # Option 2: Label Smoothing (good for regularization)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Cosine annealing scheduler with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Gradient scaler for mixed precision
    scaler = GradScaler()
    
    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)
    
    best_val_acc = 0
    best_val_top5 = 0
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
            model, train_loader, criterion, optimizer, scaler, config, use_mixup=True
        )
        
        # Validate
        val_loss, val_acc, val_top3, val_top5 = validate(
            model, val_loader, criterion, config
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_top3'].append(val_top3)
        history['val_top5'].append(val_top5)
        
        # Print results
        print(f"\n📊 Results:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   Val Top-3:  {val_top3:.2f}% | Val Top-5: {val_top5:.2f}%")
        print(f"   Gap (Train-Val): {train_acc - val_acc:.2f}%")
        
        # Check for improvement
        improved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_top5 = val_top5
            improved = True
            patience_counter = 0
            
            # Save best model
            save_path = os.path.join(config.SAVE_DIR, "best_model_augmented.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_top5': val_top5,
                'config': {
                    'model_name': config.MODEL_NAME,
                    'num_classes': config.NUM_CLASSES,
                }
            }, save_path)
            print(f"   ✅ NEW BEST! Model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement for {patience_counter}/{config.PATIENCE} epochs")
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n⚠️ Early stopping triggered after {epoch} epochs")
            break
    
    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n📊 Best Results:")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Best Top-5 Accuracy:      {best_val_top5:.2f}%")
    
    # Save training history
    history_path = os.path.join(config.LOG_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n📈 Training history saved to: {history_path}")
    
    # Calculate improvement
    original_acc = 68.79
    improvement = best_val_acc - original_acc
    print(f"\n📈 IMPROVEMENT OVER ORIGINAL:")
    print(f"   Original Val Acc:  {original_acc:.2f}%")
    print(f"   New Val Acc:       {best_val_acc:.2f}%")
    print(f"   Improvement:       {improvement:+.2f}%")
    
    return best_val_acc


if __name__ == "__main__":
    main()