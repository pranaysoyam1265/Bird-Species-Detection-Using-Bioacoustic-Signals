"""
Script: train_balanced.py
Purpose: Training with class balancing + focal loss to fix poor-performing species
Location: 09_Utils/Scripts/train_balanced.py

KEY IMPROVEMENTS:
1. Class-weighted loss (helps rare species)
2. Focal Loss (focuses on hard examples)
3. Weighted random sampler (balanced batches)
4. Stronger augmentation for rare species
"""

import os
import gc
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

import timm
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    BASE_DIR = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"
    
    # Data paths
    SPEC_DIR = os.path.join(BASE_DIR, "03_Features", "Spectrograms_Precomputed")
    TRAIN_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split_Fixed", "train.csv")
    VAL_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split_Fixed", "val.csv")
    TEST_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split_Fixed", "test.csv")
    LABEL_MAPPING = os.path.join(BASE_DIR, "04_Labels", "Processed_Labels", "label_mapping_fixed.json")
    
    # Output paths
    SAVE_DIR = os.path.join(BASE_DIR, "05_Model", "Saved_Models")
    LOG_DIR = os.path.join(BASE_DIR, "05_Model", "Training_Logs", f"balanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Model
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = 54
    PRETRAINED = True
    
    # Training
    EPOCHS = 60
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0002      # Slightly lower
    WEIGHT_DECAY = 0.05
    LABEL_SMOOTHING = 0.1
    PATIENCE = 20               # More patience
    
    # Class balancing options
    USE_FOCAL_LOSS = True       # Focus on hard examples
    FOCAL_GAMMA = 2.0           # Focal loss focusing parameter
    USE_CLASS_WEIGHTS = True    # Weight rare classes higher
    USE_WEIGHTED_SAMPLER = True # Balance batches
    
    # Data loading
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Augmentation - STRONGER for rare species
    SPEC_AUG_PROB = 0.6
    FREQ_MASK_PARAM = 25
    TIME_MASK_PARAM = 35
    N_FREQ_MASKS = 2
    N_TIME_MASKS = 2
    
    # Mixup
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4
    MIXUP_PROB = 0.5
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ============================================================
# FOCAL LOSS - Focuses on hard examples
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Down-weights easy examples, focuses on hard ones.
    
    Paper: https://arxiv.org/abs/1708.02002
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    gamma = 0: equivalent to CrossEntropy
    gamma > 0: reduces loss for well-classified examples
    """
    
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Apply label smoothing
        n_classes = inputs.size(1)
        
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get probability of true class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                   label_smoothing=self.label_smoothing)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================
# SPECAUGMENT
# ============================================================
class SpecAugment(nn.Module):
    def __init__(self, config, is_rare_species=False):
        super().__init__()
        self.freq_mask_param = config.FREQ_MASK_PARAM
        self.time_mask_param = config.TIME_MASK_PARAM
        self.n_freq_masks = config.N_FREQ_MASKS
        self.n_time_masks = config.N_TIME_MASKS
        self.p = config.SPEC_AUG_PROB
        
        # Stronger augmentation for rare species
        if is_rare_species:
            self.freq_mask_param = int(self.freq_mask_param * 1.3)
            self.time_mask_param = int(self.time_mask_param * 1.3)
            self.p = min(0.8, self.p + 0.2)
    
    def forward(self, spec):
        if not self.training or random.random() > self.p:
            return spec
        
        spec = spec.clone()
        _, n_mels, n_frames = spec.shape
        
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = random.randint(1, min(self.freq_mask_param, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            spec[:, f0:f0 + f, :] = 0
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = random.randint(1, min(self.time_mask_param, n_frames - 1))
            t0 = random.randint(0, n_frames - t)
            spec[:, :, t0:t0 + t] = 0
        
        return spec


class SpectrogramTransform:
    def __init__(self, p=0.3):
        self.p = p
    
    def __call__(self, spec):
        if random.random() > self.p:
            return spec
        
        # Random augmentation
        aug_type = random.randint(0, 2)
        
        if aug_type == 0:
            # Random gain
            spec = torch.clamp(spec * random.uniform(0.8, 1.2), 0, 1)
        elif aug_type == 1:
            # Add noise
            noise = torch.randn_like(spec) * 0.015
            spec = torch.clamp(spec + noise, 0, 1)
        else:
            # Time shift
            shift = random.randint(-10, 10)
            spec = torch.roll(spec, shifts=shift, dims=2)
        
        return spec


# ============================================================
# DATASET WITH RARE SPECIES AWARENESS
# ============================================================
class BalancedBirdDataset(Dataset):
    def __init__(self, csv_path, spec_dir, split_name, label_mapping_path, 
                 config, augment=False, rare_species=None):
        self.df = pd.read_csv(csv_path)
        self.spec_dir = os.path.join(spec_dir, split_name)
        self.config = config
        self.augment = augment
        self.rare_species = rare_species or set()
        
        # Load label mapping
        with open(label_mapping_path, 'r') as f:
            label_data = json.load(f)
        
        if isinstance(list(label_data.values())[0], dict):
            self.label_map = label_data.get('species_to_id', {})
        else:
            self.label_map = label_data
        
        # Create species to index mapping for the dataframe
        self.df['label'] = self.df['species'].map(self.label_map)
        
        # Augmentation modules
        if augment:
            self.spec_aug_normal = SpecAugment(config, is_rare_species=False)
            self.spec_aug_rare = SpecAugment(config, is_rare_species=True)
            self.spec_transform = SpectrogramTransform(p=0.3)
        else:
            self.spec_aug_normal = None
            self.spec_aug_rare = None
            self.spec_transform = None
        
        print(f"   📂 {split_name}: {len(self.df)} samples, Augment={augment}")
        if rare_species:
            print(f"      Rare species (stronger aug): {len(rare_species)}")
    
    def __len__(self):
        return len(self.df)
    
    def get_labels(self):
        """Return all labels for weighted sampler"""
        return self.df['label'].tolist()
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['chunk_file']
        species = row['species']
        label = row['label']
        
        # Load spectrogram
        spec_filename = filename.replace('.wav', '.npy').replace('.mp3', '.npy')
        spec_path = os.path.join(self.spec_dir, spec_filename)
        
        try:
            spec = np.load(spec_path)
        except:
            spec = np.zeros((128, 216), dtype=np.float32)
        
        # To 3-channel tensor
        spec_3ch = np.stack([spec, spec, spec], axis=0).astype(np.float32)
        spec_tensor = torch.from_numpy(spec_3ch)
        
        # Apply augmentation
        if self.augment:
            # Stronger augmentation for rare species
            if species in self.rare_species:
                self.spec_aug_rare.training = True
                spec_tensor = self.spec_aug_rare(spec_tensor)
                # Extra transforms for rare species
                if random.random() < 0.5:
                    spec_tensor = self.spec_transform(spec_tensor)
            else:
                self.spec_aug_normal.training = True
                spec_tensor = self.spec_aug_normal(spec_tensor)
            
            # General transforms
            if self.spec_transform and random.random() < 0.3:
                spec_tensor = self.spec_transform(spec_tensor)
        
        return spec_tensor, label


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
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, config.NUM_CLASSES)
        )
    
    def forward(self, x):
        return self.classifier(self.backbone(x))


# ============================================================
# COMPUTE CLASS WEIGHTS
# ============================================================
def compute_class_weights(train_csv, label_mapping_path, num_classes=54):
    """
    Compute class weights based on inverse frequency.
    Rare classes get higher weights.
    """
    df = pd.read_csv(train_csv)
    class_counts = df['species'].value_counts()
    
    with open(label_mapping_path, 'r') as f:
        label_data = json.load(f)
    
    if isinstance(list(label_data.values())[0], dict):
        species_to_id = label_data.get('species_to_id', {})
    else:
        species_to_id = label_data
    
    # Compute weights
    weights = torch.zeros(num_classes)
    total_samples = len(df)
    
    for species, idx in species_to_id.items():
        count = class_counts.get(species, 1)
        # Inverse frequency with sqrt smoothing
        weights[idx] = np.sqrt(total_samples / (count + 1))
    
    # Normalize
    weights = weights / weights.mean()
    
    return weights


def get_rare_species(train_csv, threshold=200):
    """
    Identify species with fewer than threshold samples.
    These get stronger augmentation.
    """
    df = pd.read_csv(train_csv)
    class_counts = df['species'].value_counts()
    
    rare = set(class_counts[class_counts < threshold].index.tolist())
    
    # Also include species known to have poor performance
    problem_species = {
        'Bucephala albeola',      # 0% F1
        'Hirundo rustica',        # 0% F1
        'Haliaeetus leucocephalus',  # 11% F1
        'Passerina caerulea',     # 14% F1
        'Recurvirostra americana',# 17% F1
        'Riparia riparia',        # 19% F1
        'Turdus migratorius',     # 24% F1
    }
    
    rare = rare.union(problem_species)
    return rare


def create_weighted_sampler(dataset, num_samples=None):
    """
    Create weighted random sampler for balanced batches.
    """
    labels = dataset.get_labels()
    class_counts = Counter(labels)
    
    # Weight = inverse of class frequency
    weights = [1.0 / class_counts[label] for label in labels]
    weights = torch.DoubleTensor(weights)
    
    if num_samples is None:
        num_samples = len(dataset)
    
    sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
    return sampler


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
def train_epoch(model, loader, criterion, optimizer, scaler, config, epoch):
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Train E{epoch}", ncols=100)
    
    for data, target in pbar:
        data = data.to(config.DEVICE, non_blocking=True)
        target = target.to(config.DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixup
        use_mixup = config.USE_MIXUP and random.random() < config.MIXUP_PROB
        
        if use_mixup:
            data, y_a, y_b, lam = mixup_data(data, target, config.MIXUP_ALPHA)
            
            with autocast():
                output = model(data)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            
            _, pred = output.max(1)
            correct += (lam * pred.eq(y_a).sum().float() + 
                       (1 - lam) * pred.eq(y_b).sum().float()).item()
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
        
        pbar.set_postfix({
            'loss': f'{total_loss/(pbar.n+1):.3f}',
            'acc': f'{100*correct/total:.1f}%'
        })
    
    return total_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion, config, epoch):
    model.eval()
    
    total_loss = 0
    correct = 0
    top5_correct = 0
    total = 0
    
    # Per-class tracking
    class_correct = Counter()
    class_total = Counter()
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc=f"Val E{epoch}", ncols=100):
            data = data.to(config.DEVICE, non_blocking=True)
            target = target.to(config.DEVICE, non_blocking=True)
            
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            total_loss += loss.item()
            
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Top-5
            _, top5 = output.topk(5, dim=1)
            for i in range(target.size(0)):
                if target[i] in top5[i]:
                    top5_correct += 1
                
                # Per-class tracking
                class_total[target[i].item()] += 1
                if pred[i] == target[i]:
                    class_correct[target[i].item()] += 1
    
    # Calculate per-class accuracy
    per_class_acc = {}
    for cls in class_total:
        per_class_acc[cls] = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
    
    # Find worst performing classes
    worst_classes = sorted(per_class_acc.items(), key=lambda x: x[1])[:5]
    
    return {
        'loss': total_loss / len(loader),
        'acc': 100 * correct / total,
        'top5_acc': 100 * top5_correct / total,
        'per_class_acc': per_class_acc,
        'worst_classes': worst_classes
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("⚖️ BALANCED TRAINING WITH FOCAL LOSS")
    print("=" * 70)
    
    config = Config()
    set_seed(config.SEED)
    
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    print(f"\n📋 Configuration:")
    print(f"   Model:              {config.MODEL_NAME}")
    print(f"   Focal Loss:         {config.USE_FOCAL_LOSS} (gamma={config.FOCAL_GAMMA})")
    print(f"   Class Weights:      {config.USE_CLASS_WEIGHTS}")
    print(f"   Weighted Sampler:   {config.USE_WEIGHTED_SAMPLER}")
    print(f"   Device:             {config.DEVICE}")
    
    # Identify rare/problem species
    print(f"\n🔍 Identifying rare species...")
    rare_species = get_rare_species(config.TRAIN_CSV, threshold=200)
    print(f"   Rare/problem species: {len(rare_species)}")
    for s in list(rare_species)[:5]:
        print(f"      - {s}")
    if len(rare_species) > 5:
        print(f"      ... and {len(rare_species) - 5} more")
    
    # Compute class weights
    print(f"\n⚖️ Computing class weights...")
    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(config.TRAIN_CSV, config.LABEL_MAPPING)
        class_weights = class_weights.to(config.DEVICE)
        
        print(f"   Weight range: [{class_weights.min():.2f}, {class_weights.max():.2f}]")
        print(f"   Mean weight:  {class_weights.mean():.2f}")
    
    # Load datasets
    print(f"\n📂 Loading datasets...")
    
    train_ds = BalancedBirdDataset(
        config.TRAIN_CSV, config.SPEC_DIR, "train",
        config.LABEL_MAPPING, config, augment=True,
        rare_species=rare_species
    )
    
    val_ds = BalancedBirdDataset(
        config.VAL_CSV, config.SPEC_DIR, "val",
        config.LABEL_MAPPING, config, augment=False
    )
    
    # Create data loaders
    if config.USE_WEIGHTED_SAMPLER:
        print(f"   Using weighted random sampler for balanced batches")
        sampler = create_weighted_sampler(train_ds)
        train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            sampler=sampler,  # Use sampler instead of shuffle
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=True,
            drop_last=True
        )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=True
    )
    
    print(f"   Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"   Val:   {len(val_ds)} samples, {len(val_loader)} batches")
    
    # Build model
    print(f"\n🏗️ Building model...")
    model = BirdClassifier(config).to(config.DEVICE)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    print(f"\n📉 Setting up loss function...")
    if config.USE_FOCAL_LOSS:
        criterion = FocalLoss(
            gamma=config.FOCAL_GAMMA,
            alpha=class_weights,
            label_smoothing=config.LABEL_SMOOTHING
        )
        print(f"   Using Focal Loss (gamma={config.FOCAL_GAMMA})")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.LABEL_SMOOTHING
        )
        print(f"   Using CrossEntropy with class weights")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )
    
    scaler = GradScaler()
    
    # Training loop
    print("\n" + "=" * 70)
    print("🚀 TRAINING STARTED")
    print("=" * 70)
    
    best_val_acc = 0
    best_top5_acc = 0
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_top5': []
    }
    
    total_start = time.time()
    
    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()
        
        print(f"\n{'─' * 70}")
        print(f"Epoch {epoch}/{config.EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, config, epoch
        )
        
        # Validate
        val_results = validate(model, val_loader, criterion, config, epoch)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['acc'])
        history['val_top5'].append(val_results['top5_acc'])
        
        # Print results
        gap = train_acc - val_results['acc']
        print(f"\n📊 Train: {train_acc:.1f}% | Val: {val_results['acc']:.1f}% | "
              f"Top5: {val_results['top5_acc']:.1f}% | Gap: {gap:.1f}% | Time: {epoch_time:.0f}s")
        
        # Show worst classes progress
        if epoch % 5 == 0:
            print(f"   Worst classes this epoch:")
            for cls_id, acc in val_results['worst_classes'][:3]:
                print(f"      Class {cls_id}: {acc*100:.1f}%")
        
        # Check for best model
        if val_results['acc'] > best_val_acc:
            best_val_acc = val_results['acc']
            best_top5_acc = val_results['top5_acc']
            patience_counter = 0
            
            save_path = os.path.join(config.SAVE_DIR, "best_model_balanced.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_results['acc'],
                'val_top5_acc': val_results['top5_acc'],
            }, save_path)
            print(f"   ✅ NEW BEST! Saved.")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{config.PATIENCE})")
        
        if patience_counter >= config.PATIENCE:
            print(f"\n⚠️ Early stopping at epoch {epoch}")
            break
    
    # Summary
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n📊 Best Results:")
    print(f"   Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Top-5 Accuracy:      {best_top5_acc:.2f}%")
    print(f"   Training Time:       {total_time/60:.1f} minutes")
    
    # Save history
    with open(os.path.join(config.LOG_DIR, "history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Compare
    print(f"\n📈 Comparison:")
    print(f"   Previous Best Val:  71.32%")
    print(f"   New Val Accuracy:   {best_val_acc:.2f}%")
    print(f"   Change:             {best_val_acc - 71.32:+.2f}%")


if __name__ == "__main__":
    main()