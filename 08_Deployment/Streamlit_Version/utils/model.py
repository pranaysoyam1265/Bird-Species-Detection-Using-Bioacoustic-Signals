# Save as: 08_Deployment/utils/model.py

"""
Model loading and inference utilities
"""

import torch
import torch.nn as nn
import timm
import numpy as np
from pathlib import Path
import json
import streamlit as st

# ============================================================
# MODEL DEFINITION
# ============================================================

class BirdClassifier(nn.Module):
    """EfficientNet-based bird species classifier"""
    
    def __init__(self, num_classes=87, dropout_rate=0.4):
        super().__init__()
        
        # Backbone
        self.backbone = timm.create_model(
            'tf_efficientnet_b2_ns',
            pretrained=False,
            num_classes=0,
            global_pool='avg'
        )
        
        num_features = self.backbone.num_features  # 1408 for B2
        
        # Classifier
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
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_features(self, x):
        """Get intermediate features for visualization"""
        return self.backbone(x)

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model(model_path, device='cuda'):
    """Load trained model (cached)"""
    
    if not Path(model_path).exists():
        st.error(f"Model not found: {model_path}")
        return None, None
    
    # Determine device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    device = torch.device(device)
    
    # Create model
    model = BirdClassifier(num_classes=87)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    # Get metadata
    metadata = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'val_acc': checkpoint.get('val_acc', 0),
        'val_top5': checkpoint.get('val_top5', 0),
    }
    
    return model, metadata

@st.cache_data
def load_label_mapping(mapping_path):
    """Load label mapping (cached)"""
    
    if not Path(mapping_path).exists():
        st.error(f"Label mapping not found: {mapping_path}")
        return {}, {}
    
    with open(mapping_path, 'r') as f:
        raw_mapping = json.load(f)
    
    # Extract indices and create both mappings
    idx_to_species = {}
    idx_to_english = {}
    species_to_idx = {}
    
    for scientific_name, info in raw_mapping.items():
        idx = info['index']
        english_name = info.get('english_name', scientific_name)
        
        idx_to_species[idx] = scientific_name
        idx_to_english[idx] = english_name
        species_to_idx[scientific_name] = idx
    
    return {
        'idx_to_species': idx_to_species,
        'idx_to_english': idx_to_english,
        'species_to_idx': species_to_idx,
        'raw': raw_mapping,
    }

# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_spectrogram(spec):
    """
    Preprocess spectrogram for model input.
    Matches training preprocessing exactly.
    """
    # Normalize to [0, 1]
    spec_min = spec.min()
    spec_max = spec.max()
    
    if spec_max - spec_min > 0:
        spec = (spec - spec_min) / (spec_max - spec_min)
    else:
        spec = np.zeros_like(spec)
    
    # Clip values
    spec = np.clip(spec, 0, 1)
    
    # Convert to 3-channel tensor
    spec_tensor = torch.FloatTensor(spec).unsqueeze(0)  # (1, H, W)
    spec_tensor = spec_tensor.repeat(3, 1, 1)  # (3, H, W)
    
    return spec_tensor

# ============================================================
# INFERENCE
# ============================================================

def predict_single(model, spectrogram, device='cuda'):
    """
    Run inference on a single spectrogram.
    
    Args:
        model: Loaded PyTorch model
        spectrogram: numpy array (128, 216) or preprocessed tensor
        device: 'cuda' or 'cpu'
    
    Returns:
        dict with predictions, probabilities, features
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    device = torch.device(device)
    
    # Preprocess if needed
    if isinstance(spectrogram, np.ndarray):
        input_tensor = preprocess_spectrogram(spectrogram)
    else:
        input_tensor = spectrogram
    
    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        
        # Get features for visualization
        features = model.get_features(input_tensor)
    
    # Convert to numpy
    probs_np = probs[0].cpu().numpy()
    features_np = features[0].cpu().numpy()
    
    # Get top-k predictions
    top_k = 10
    top_indices = np.argsort(probs_np)[::-1][:top_k]
    
    return {
        'probabilities': probs_np,
        'top_indices': top_indices,
        'top_probs': probs_np[top_indices],
        'features': features_np,
        'predicted_class': top_indices[0],
        'confidence': probs_np[top_indices[0]],
    }

def predict_batch(model, spectrograms, device='cuda', batch_size=16):
    """
    Run inference on multiple spectrograms.
    
    Args:
        model: Loaded PyTorch model
        spectrograms: list of numpy arrays
        device: 'cuda' or 'cpu'
        batch_size: batch size for inference
    
    Returns:
        list of prediction dicts
    """
    results = []
    
    for i in range(0, len(spectrograms), batch_size):
        batch = spectrograms[i:i+batch_size]
        
        # Preprocess batch
        tensors = [preprocess_spectrogram(s) for s in batch]
        batch_tensor = torch.stack(tensors).to(device)
        
        # Run inference
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
        
        # Process each result
        for j, prob in enumerate(probs):
            prob_np = prob.cpu().numpy()
            top_indices = np.argsort(prob_np)[::-1][:10]
            
            results.append({
                'probabilities': prob_np,
                'top_indices': top_indices,
                'top_probs': prob_np[top_indices],
                'predicted_class': top_indices[0],
                'confidence': prob_np[top_indices[0]],
            })
    
    return results

# ============================================================
# UNCERTAINTY QUANTIFICATION
# ============================================================

def calculate_uncertainty(probs):
    """
    Calculate uncertainty metrics for predictions.
    
    Args:
        probs: numpy array of probabilities
    
    Returns:
        dict with uncertainty metrics
    """
    # Entropy (higher = more uncertain)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(probs))  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy
    
    # Margin (difference between top 2)
    sorted_probs = np.sort(probs)[::-1]
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
    
    # Top probability
    top_prob = sorted_probs[0]
    
    # Confidence level
    if top_prob >= 0.8:
        confidence_level = "High"
    elif top_prob >= 0.5:
        confidence_level = "Medium"
    elif top_prob >= 0.2:
        confidence_level = "Low"
    else:
        confidence_level = "Very Low"
    
    return {
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'margin': margin,
        'top_probability': top_prob,
        'confidence_level': confidence_level,
        'is_confident': top_prob >= 0.5 and margin >= 0.2,
    }