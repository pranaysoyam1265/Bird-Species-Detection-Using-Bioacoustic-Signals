"""
╔══════════════════════════════════════════════════════════════════════╗
║  OPTIMIZED TRAINING SCRIPT v3                                       ║
║  Target: 85%+ accuracy on 87 bird species                          ║
║  Dataset: 127,263 chunks (180K spectrograms)                        ║
║                                                                      ║
║  Features:                                                           ║
║  ├── EfficientNet-B2 (upgradable to B3)                             ║
║  ├── Focal Loss (handles class imbalance)                           ║
║  ├── Weighted Random Sampling                                        ║
║  ├── SpecAugment (frequency + time masking)                         ║
║  ├── Mixup Augmentation                                              ║
║  ├── Cosine Annealing with Warmup                                   ║
║  ├── Mixed Precision (FP16)                                          ║
║  ├── Gradient Accumulation                                           ║
║  └── Early Stopping + Best Model Saving                             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import timm

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    # Paths
    PROJECT_ROOT = Path(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")
    SPEC_DIR = PROJECT_ROOT / "03_Features" / "Spectrograms"
    SPLIT_DIR = PROJECT_ROOT / "04_Labels" / "Train_Val_Test_Split"
    LABEL_MAP_PATH = PROJECT_ROOT / "04_Labels" / "Processed_Labels" / "label_mapping_v3.json"
    MODEL_DIR = PROJECT_ROOT / "05_Model" / "Saved_Models"
    CHECKPOINT_DIR = PROJECT_ROOT / "05_Model" / "Checkpoints"
    LOG_DIR = PROJECT_ROOT / "05_Model" / "Training_Logs"
    
    # Model
    MODEL_NAME = 'tf_efficientnet_b2_ns'  # Noisy Student pretrained
    NUM_CLASSES = 87  # Will be updated from label mapping
    
    # Training
    EPOCHS = 50
    BATCH_SIZE = 16  # Safe for 4GB VRAM with FP16
    ACCUMULATION_STEPS = 2  # Effective batch size = 16 × 2 = 32
    LEARNING_RATE = 1e-3
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 3
    
    # Augmentation
    SPEC_AUGMENT = True
    FREQ_MASK_PARAM = 20  # Max frequency bands to mask
    TIME_MASK_PARAM = 40  # Max time steps to mask
    NUM_FREQ_MASKS = 2
    NUM_TIME_MASKS = 2
    MIXUP_ALPHA = 0.3  # Mixup strength (0 = disabled)
    
    # Loss
    FOCAL_LOSS_GAMMA = 2.0  # Focus on hard examples
    LABEL_SMOOTHING = 0.1
    
    # Early stopping
    PATIENCE = 10
    MIN_DELTA = 0.001
    
    # Hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4
    PIN_MEMORY = True
    USE_AMP = True  # Mixed precision
    
    # Seed
    SEED = 42

# ============================================================
# SET SEED FOR REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# DATASET
# ============================================================

class BirdSpectrogramDataset(Dataset):
    """Dataset for loading mel-spectrograms with augmentation"""
    
    def __init__(self, csv_path, spec_dir, augment=False, config=None):
        self.df = pd.read_csv(csv_path)
        self.spec_dir = Path(spec_dir)
        self.augment = augment
        self.config = config or Config()
        
        # Verify spectrogram paths exist
        valid_mask = []
        for _, row in self.df.iterrows():
            spec_path = Path(row['spectrogram_path'])
            if not spec_path.exists():
                # Try relative path from spec_dir
                alt_path = self.spec_dir / f"{row['chunk_filename']}.npy"
                if alt_path.exists():
                    valid_mask.append(True)
                else:
                    valid_mask.append(False)
            else:
                valid_mask.append(True)
        
        initial_len = len(self.df)
        self.df = self.df[valid_mask].reset_index(drop=True)
        if len(self.df) < initial_len:
            print(f"  ⚠️ Removed {initial_len - len(self.df)} missing spectrograms")
        
        print(f"  📊 Dataset loaded: {len(self.df)} samples")
    
    def __len__(self):
        return len(self.df)
    
    def _load_spectrogram(self, idx):
        """Load spectrogram from .npy file"""
        row = self.df.iloc[idx]
        
        # Try absolute path first
        spec_path = Path(row['spectrogram_path'])
        if not spec_path.exists():
            spec_path = self.spec_dir / f"{row['chunk_filename']}.npy"
        
        spec = np.load(str(spec_path))  # Shape: (128, 216)
        return spec
    
    def _normalize(self, spec):
        """Normalize spectrogram to [0, 1]"""
        spec_min = spec.min()
        spec_max = spec.max()
        if spec_max - spec_min > 0:
            spec = (spec - spec_min) / (spec_max - spec_min)
        else:
            spec = np.zeros_like(spec)
        return spec
    
    def _spec_augment(self, spec):
        """Apply SpecAugment: frequency and time masking"""
        spec = spec.copy()
        n_mels, n_time = spec.shape
        
        # Frequency masking
        for _ in range(self.config.NUM_FREQ_MASKS):
            f = np.random.randint(0, self.config.FREQ_MASK_PARAM)
            f0 = np.random.randint(0, max(1, n_mels - f))
            spec[f0:f0 + f, :] = 0
        
        # Time masking
        for _ in range(self.config.NUM_TIME_MASKS):
            t = np.random.randint(0, self.config.TIME_MASK_PARAM)
            t0 = np.random.randint(0, max(1, n_time - t))
            spec[:, t0:t0 + t] = 0
        
        return spec
    
    def _random_augment(self, spec):
        """Additional random augmentations"""
        # Random volume/gain adjustment
        if np.random.random() > 0.5:
            gain = np.random.uniform(0.8, 1.2)
            spec = spec * gain
        
        # Add slight Gaussian noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.01, spec.shape)
            spec = spec + noise
        
        # Random time shift
        if np.random.random() > 0.5:
            shift = np.random.randint(-10, 10)
            spec = np.roll(spec, shift, axis=1)
        
        return spec
    
    def __getitem__(self, idx):
        # Load spectrogram
        spec = self._load_spectrogram(idx)  # (128, 216)
        
        # Normalize
        spec = self._normalize(spec)
        
        # Apply augmentation during training
        if self.augment:
            if self.config.SPEC_AUGMENT:
                spec = self._spec_augment(spec)
            spec = self._random_augment(spec)
        
        # Clip values
        spec = np.clip(spec, 0, 1)
        
        # Convert to 3-channel tensor (for pretrained ImageNet models)
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0)  # (1, 128, 216)
        spec_tensor = spec_tensor.repeat(3, 1, 1)  # (3, 128, 216)
        
        # Get label
        label = int(self.df.iloc[idx]['label'])
        
        return spec_tensor, label

# ============================================================
# FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: focuses on hard-to-classify examples
    Down-weights easy examples, up-weights hard ones
    """
    def __init__(self, gamma=2.0, label_smoothing=0.1, weight=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight
    
    def forward(self, inputs, targets):
        # Apply label smoothing
        n_classes = inputs.size(1)
        
        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        # Compute log softmax
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Focal weight
        focal_weight = (1 - probs) ** self.gamma
        
        # Compute loss
        loss = -focal_weight * smooth_targets * log_probs
        
        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight.to(inputs.device)
            loss = loss * weight.unsqueeze(0)
        
        return loss.sum(dim=1).mean()

# ============================================================
# MIXUP AUGMENTATION
# ============================================================

def mixup_data(x, y, alpha=0.3):
    """Apply mixup augmentation"""
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
    """Compute mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================
# MODEL
# ============================================================

class BirdClassifier(nn.Module):
    """EfficientNet-based bird species classifier"""
    
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        
        # Load pretrained EfficientNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove original classifier
            global_pool='avg'
        )
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Custom classifier with dropout
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        print(f"  🏗️ Model: {model_name}")
        print(f"  📐 Feature dim: {feature_dim}")
        print(f"  🎯 Classes: {num_classes}")
        print(f"  📊 Total params: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

# ============================================================
# LEARNING RATE SCHEDULER WITH WARMUP
# ============================================================

class WarmupCosineScheduler:
    """Cosine annealing with linear warmup"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = max(self.min_lr, base_lr * factor)
        
        return self.optimizer.param_groups[0]['lr']

# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    """Stop training when validation loss stops improving"""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop

# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, config, epoch):
    """Train for one epoch with mixup and gradient accumulation"""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc=f"  Train Epoch {epoch+1}", leave=False)
    
    for batch_idx, (specs, labels) in enumerate(pbar):
        specs = specs.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)
        
        # Apply mixup
        use_mixup = config.MIXUP_ALPHA > 0 and np.random.random() > 0.5
        
        if use_mixup:
            specs, labels_a, labels_b, lam = mixup_data(specs, labels, config.MIXUP_ALPHA)
        
        # Forward pass with mixed precision
        with autocast(enabled=config.USE_AMP):
            outputs = model(specs)
            
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            
            loss = loss / config.ACCUMULATION_STEPS
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Track metrics
        running_loss += loss.item() * config.ACCUMULATION_STEPS
        
        if not use_mixup:
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{running_loss/(batch_idx+1):.4f}",
            'acc': f"{100.*correct/max(total,1):.1f}%"
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / max(total, 1)
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, config):
    """Validate model"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    correct_top5 = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="  Validation", leave=False)
        
        for specs, labels in pbar:
            specs = specs.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)
            
            with autocast(enabled=config.USE_AMP):
                outputs = model(specs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, True, True)
            correct_top5 += top5_pred.eq(labels.unsqueeze(1).expand_as(top5_pred)).any(1).sum().item()
            
            # Store for per-class analysis
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f"{running_loss/(len(all_labels)//config.BATCH_SIZE + 1):.4f}",
                'acc': f"{100.*correct/total:.1f}%"
            })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    epoch_top5 = 100. * correct_top5 / total
    
    return epoch_loss, epoch_acc, epoch_top5, np.array(all_preds), np.array(all_labels)

# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main():
    config = Config()
    
    # Create directories
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set seed
    set_seed(config.SEED)
    
    print("=" * 70)
    print("🐦 BIRD SPECIES CLASSIFIER — OPTIMIZED TRAINING v3")
    print("=" * 70)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️ Device: {config.DEVICE}")
    if config.DEVICE == 'cuda':
        print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
        print(f"🖥️ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 70)
    
    # ── Load Label Mapping ──────────────────────────────────
    print("\n📋 Loading label mapping...")
    with open(config.LABEL_MAP_PATH, 'r') as f:
        label_mapping = json.load(f)
    
    config.NUM_CLASSES = len(label_mapping)
    print(f"  🏷️ Number of classes: {config.NUM_CLASSES}")
    
    # Create index to name mapping
    idx_to_species = {}
    for species, info in label_mapping.items():
        idx = info['index']
        english = info['english_name']
        idx_to_species[idx] = english
    
    # ── Load Datasets ───────────────────────────────────────
    print("\n📂 Loading datasets...")
    
    train_dataset = BirdSpectrogramDataset(
        csv_path=config.SPLIT_DIR / "train_v3.csv",
        spec_dir=config.SPEC_DIR,
        augment=True,
        config=config
    )
    
    val_dataset = BirdSpectrogramDataset(
        csv_path=config.SPLIT_DIR / "val_v3.csv",
        spec_dir=config.SPEC_DIR,
        augment=False,
        config=config
    )
    
    test_dataset = BirdSpectrogramDataset(
        csv_path=config.SPLIT_DIR / "test_v3.csv",
        spec_dir=config.SPEC_DIR,
        augment=False,
        config=config
    )
    
    # ── Class-Weighted Sampling ─────────────────────────────
    print("\n⚖️ Creating weighted sampler for class balance...")
    
    train_labels = train_dataset.df['label'].values
    class_counts = np.bincount(train_labels, minlength=config.NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1)  # +1 to avoid division by zero
    sample_weights = class_weights[train_labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_dataset),
        replacement=True
    )
    
    print(f"  📊 Max class size: {class_counts.max()}")
    print(f"  📊 Min class size: {class_counts[class_counts > 0].min()}")
    print(f"  📊 Imbalance ratio: {class_counts.max() / class_counts[class_counts > 0].min():.1f}x")
    
    # ── Data Loaders ────────────────────────────────────────
    print("\n📦 Creating data loaders...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"  🚂 Train batches: {len(train_loader)}")
    print(f"  ✅ Val batches: {len(val_loader)}")
    print(f"  🧪 Test batches: {len(test_loader)}")
    
    # ── Create Model ────────────────────────────────────────
    print(f"\n🏗️ Creating model...")
    model = BirdClassifier(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=True
    )
    model = model.to(config.DEVICE)
    
    # ── Loss Function ───────────────────────────────────────
    print("\n⚙️ Setting up training components...")
    
    # Class weights for focal loss
    class_weights_tensor = torch.FloatTensor(class_weights / class_weights.sum())
    
    criterion = FocalLoss(
        gamma=config.FOCAL_LOSS_GAMMA,
        label_smoothing=config.LABEL_SMOOTHING,
        weight=class_weights_tensor
    )
    
    print(f"  📉 Loss: Focal Loss (γ={config.FOCAL_LOSS_GAMMA}, "
          f"smoothing={config.LABEL_SMOOTHING})")
    
    # ── Optimizer ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    print(f"  🔧 Optimizer: AdamW (lr={config.LEARNING_RATE}, "
          f"wd={config.WEIGHT_DECAY})")
    
    # ── Scheduler ───────────────────────────────────────────
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        total_epochs=config.EPOCHS,
        min_lr=config.MIN_LR
    )
    print(f"  📈 Scheduler: Cosine Annealing with {config.WARMUP_EPOCHS} warmup epochs")
    
    # ── Gradient Scaler (FP16) ──────────────────────────────
    scaler = GradScaler(enabled=config.USE_AMP)
    print(f"  ⚡ Mixed Precision: {'ON' if config.USE_AMP else 'OFF'}")
    print(f"  📦 Effective Batch Size: {config.BATCH_SIZE * config.ACCUMULATION_STEPS}")
    
    # ── Early Stopping ──────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        min_delta=config.MIN_DELTA
    )
    
    # ── Training Log ────────────────────────────────────────
    training_log = []
    best_val_acc = 0.0
    best_epoch = 0
    
    # ════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)
    print(f"{'Epoch':>6} │ {'Train Loss':>10} │ {'Train Acc':>9} │ "
          f"{'Val Loss':>9} │ {'Val Acc':>8} │ {'Top-5':>6} │ "
          f"{'LR':>10} │ {'Status':>8}")
    print("─" * 90)
    
    total_start_time = time.time()
    
    # Resume from checkpoint
    RESUME_FROM = "checkpoint_epoch_10.pth"  # Change epoch number (10 available, others: 5, 15)
    
    if RESUME_FROM:
        try:
            checkpoint_path = config.CHECKPOINT_DIR / RESUME_FROM
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path)
                # Use strict=False to skip incompatible keys due to architecture changes
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                # Optimizer might also have incompatibilities, wrap in try-except
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    print("⚠️ Could not load optimizer state (different architecture), using fresh optimizer")
                start_epoch = checkpoint.get('epoch', 0)
                print(f"✅ Resumed from epoch {start_epoch}")
            else:
                print(f"⚠️ Checkpoint not found: {checkpoint_path}")
                start_epoch = 0
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            start_epoch = 0
    else:
        start_epoch = 0
    
    for epoch in range(start_epoch, config.EPOCHS):
        epoch_start = time.time()
        
        # Update learning rate
        current_lr = scheduler.step(epoch)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, config, epoch
        )
        
        # Validate
        val_loss, val_acc, val_top5, val_preds, val_labels = validate(
            model, val_loader, criterion, config
        )
        
        epoch_time = time.time() - epoch_start
        
        # Check for best model
        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            status = "⭐ BEST"
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_top5': val_top5,
                'val_loss': val_loss,
                'config': {
                    'model_name': config.MODEL_NAME,
                    'num_classes': config.NUM_CLASSES,
                    'batch_size': config.BATCH_SIZE,
                }
            }, config.MODEL_DIR / "best_model_v3.pth")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch+1}.pth")
        
        # Log
        training_log.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_top5': val_top5,
            'lr': current_lr,
            'time': epoch_time
        })
        
        # Print epoch summary
        print(f"{epoch+1:>6} │ {train_loss:>10.4f} │ {train_acc:>8.2f}% │ "
              f"{val_loss:>9.4f} │ {val_acc:>7.2f}% │ {val_top5:>5.1f}% │ "
              f"{current_lr:>10.6f} │ {status:>8}")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}")
            print(f"   Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
            break
    
    total_time = time.time() - total_start_time
    
    # ════════════════════════════════════════════════════════
    # SAVE TRAINING LOG
    # ════════════════════════════════════════════════════════
    
    log_df = pd.DataFrame(training_log)
    log_path = config.LOG_DIR / f"training_log_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_df.to_csv(log_path, index=False)
    
    # ════════════════════════════════════════════════════════
    # TEST EVALUATION
    # ════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("🧪 EVALUATING ON TEST SET")
    print("=" * 70)
    
    # Load best model
    checkpoint = torch.load(config.MODEL_DIR / "best_model_v3.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_top5, test_preds, test_labels = validate(
        model, test_loader, criterion, config
    )
    
    # Per-class accuracy
    print(f"\n📊 Test Results:")
    print(f"  Top-1 Accuracy: {test_acc:.2f}%")
    print(f"  Top-5 Accuracy: {test_top5:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}")
    
    # Per-class breakdown
    print(f"\n📊 Per-Class Accuracy (Top 10 Best):")
    class_correct = {}
    class_total = {}
    
    for pred, label in zip(test_preds, test_labels):
        species = idx_to_species.get(label, f"Class {label}")
        class_total[species] = class_total.get(species, 0) + 1
        if pred == label:
            class_correct[species] = class_correct.get(species, 0) + 1
    
    class_accuracies = {}
    for species in class_total:
        acc = 100.0 * class_correct.get(species, 0) / class_total[species]
        class_accuracies[species] = acc
    
    # Sort by accuracy
    sorted_acc = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    print("  ┌─────────────────────────────────────┬──────────┬─────────┐")
    print("  │ Species                             │ Accuracy │ Samples │")
    print("  ├─────────────────────────────────────┼──────────┼─────────┤")
    for species, acc in sorted_acc[:10]:
        print(f"  │ {species:<35} │ {acc:>6.1f}%  │ {class_total[species]:>7} │")
    print("  └─────────────────────────────────────┴──────────┴─────────┘")
    
    print(f"\n📊 Per-Class Accuracy (Bottom 10 - Need Improvement):")
    print("  ┌─────────────────────────────────────┬──────────┬─────────┐")
    print("  │ Species                             │ Accuracy │ Samples │")
    print("  ├─────────────────────────────────────┼──────────┼─────────┤")
    for species, acc in sorted_acc[-10:]:
        print(f"  │ {species:<35} │ {acc:>6.1f}%  │ {class_total[species]:>7} │")
    print("  └─────────────────────────────────────┴──────────┴─────────┘")
    
    # Species below 85%
    below_85 = [(s, a) for s, a in sorted_acc if a < 85]
    print(f"\n⚠️ Species below 85% accuracy: {len(below_85)}/{len(sorted_acc)}")
    
    # ════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"""
📊 FINAL RESULTS
{'─' * 50}
Model:              {config.MODEL_NAME}
Classes:            {config.NUM_CLASSES}
Best Epoch:         {best_epoch}
Training Time:      {total_time/3600:.1f} hours

ACCURACY
{'─' * 50}
Train Accuracy:     {training_log[best_epoch-1]['train_acc']:.2f}%
Val Accuracy:       {best_val_acc:.2f}%
Test Top-1:         {test_acc:.2f}%
Test Top-5:         {test_top5:.2f}%
Overfitting Gap:    {training_log[best_epoch-1]['train_acc'] - test_acc:.2f}%

FILES SAVED
{'─' * 50}
Best Model:    {config.MODEL_DIR / 'best_model_v3.pth'}
Training Log:  {log_path}

{'─' * 50}
Species ≥85%:  {len(sorted_acc) - len(below_85)}/{len(sorted_acc)}
Species <85%:  {len(below_85)}/{len(sorted_acc)}
""")
    
    # Save test results
    results = {
        'test_accuracy': test_acc,
        'test_top5': test_top5,
        'test_loss': test_loss,
        'best_epoch': best_epoch,
        'num_classes': config.NUM_CLASSES,
        'model_name': config.MODEL_NAME,
        'per_class_accuracy': class_accuracies,
        'training_time_hours': total_time / 3600
    }
    
    results_path = config.LOG_DIR / "test_results_v3.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📊 Test results saved: {results_path}")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()