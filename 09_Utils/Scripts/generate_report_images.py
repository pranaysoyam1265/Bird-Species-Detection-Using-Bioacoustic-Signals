"""
Bird Detection Project - Report Image Generator
This script creates all images needed for the project report.
Run this from the project root folder.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Arrow, Rectangle
from matplotlib.lines import Line2D
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not found. Some images will use sample data.")

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not found. Some images will use sample data.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not found. Using sample data.")

# ============================================================
# CONFIGURATION
# ============================================================

# Project paths - UPDATE THESE IF NEEDED
PROJECT_ROOT = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "10_Outputs", "Report_Images")

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"Output folder: {OUTPUT_FOLDER}")
print("=" * 60)

# ============================================================
# COLOR SCHEME
# ============================================================

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'info': '#17A2B8',
    'light': '#F8F9FA',
    'dark': '#343A40',
    'efficientnet': '#2E86AB',
    'resnet': '#28A745',
    'vgg': '#DC3545',
    'mobilenet': '#FFC107',
    'densenet': '#A23B72'
}

# Model data for charts
MODEL_DATA = {
    'names': ['VGG-16', 'MobileNetV2', 'ResNet-50', 'DenseNet-121', 'EfficientNet-B0'],
    'top1_accuracy': [61.38, 65.71, 68.42, 69.85, 72.26],
    'top3_accuracy': [74.22, 77.89, 79.15, 80.47, 82.83],
    'top5_accuracy': [78.94, 81.45, 82.31, 83.62, 85.55],
    'parameters': [138, 3.4, 25.6, 8.0, 4.0],  # in millions
    'size_mb': [528, 14, 98, 31, 17],
    'inference_time': [1.5, 0.3, 0.8, 0.6, 0.5],  # seconds
    'colors': ['#DC3545', '#FFC107', '#28A745', '#A23B72', '#2E86AB'],
    'years': [2014, 2018, 2015, 2017, 2019]
}

# Species data
SPECIES_DATA = {
    'top_species': [
        ('American Goldfinch', 94.2, 1062),
        ('Ash-throated Flycatcher', 91.8, 1253),
        ('Barred Owl', 89.5, 318),
        ('American Woodcock', 87.3, 513),
        ('Blue Jay', 86.1, 452),
        ('American Crow', 85.4, 549),
        ('Bobolink', 84.7, 706),
        ('American Robin', 83.9, 482),
        ('Brown Thrasher', 82.6, 695),
        ('Belted Kingfisher', 81.2, 325)
    ],
    'bottom_species': [
        ('Bufflehead', 38.7, 98),
        ('Baird\'s Sandpiper', 42.3, 109),
        ('American Avocet', 45.1, 106),
        ('American Kestrel', 48.6, 107),
        ('Bonaparte\'s Gull', 51.2, 192)
    ]
}

# Training data
TRAINING_DATA = {
    'epochs': list(range(1, 18)),
    'train_loss': [2.8, 2.1, 1.6, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.07],
    'val_loss': [2.9, 2.3, 1.9, 1.6, 1.4, 1.3, 1.25, 1.2, 1.18, 1.15, 1.2, 1.22, 1.25, 1.28, 1.3, 1.32, 1.35],
    'train_acc': [15, 35, 50, 62, 72, 80, 86, 91, 94, 96, 97.5, 98.2, 98.8, 99.2, 99.5, 99.7, 99.83],
    'val_acc': [12, 30, 42, 50, 55, 59, 62, 64, 66, 68.79, 68.2, 67.8, 67.5, 67.2, 67.0, 66.8, 66.5]
}


# ============================================================
# IMAGE 1: SYSTEM ARCHITECTURE DIAGRAM
# ============================================================

def create_system_architecture():
    """Create system architecture flowchart"""
    print("Creating: System Architecture Diagram...")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, 'Bird Species Detection System Architecture', 
            fontsize=18, fontweight='bold', ha='center', va='center')
    
    # Define box positions
    boxes = [
        # (x, y, width, height, label, color)
        (0.5, 7, 2.5, 1.2, 'Audio Input\n(WAV/MP3/OGG)', '#E3F2FD'),
        (4, 7, 2.5, 1.2, 'Preprocessing\n22050Hz, Mono', '#FFF3E0'),
        (7.5, 7, 2.5, 1.2, 'Chunking\n5-sec, 50% overlap', '#FFF3E0'),
        (11, 7, 2.5, 1.2, 'Mel-Spectrogram\n128x216 pixels', '#E8F5E9'),
        (4, 4.5, 2.5, 1.2, 'EfficientNet-B0\nFeature Extraction', '#E1BEE7'),
        (7.5, 4.5, 2.5, 1.2, 'Classification\n54 Species', '#E1BEE7'),
        (11, 4.5, 2.5, 1.2, 'Softmax\nProbabilities', '#E1BEE7'),
        (0.5, 4.5, 2.5, 1.2, 'Grad-CAM\nExplainability', '#FFCDD2'),
        (4, 2, 2.5, 1.2, 'Confidence\nThreshold (80%)', '#B3E5FC'),
        (7.5, 2, 2.5, 1.2, 'Timeline\nVisualization', '#B3E5FC'),
        (11, 2, 2.5, 1.2, 'PDF Report\nGeneration', '#C8E6C9'),
    ]
    
    # Draw boxes
    for x, y, w, h, label, color in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrow_style = dict(arrowstyle='->', color='black', lw=2)
    
    # Top row arrows
    ax.annotate('', xy=(4, 7.6), xytext=(3, 7.6), arrowprops=arrow_style)
    ax.annotate('', xy=(7.5, 7.6), xytext=(6.5, 7.6), arrowprops=arrow_style)
    ax.annotate('', xy=(11, 7.6), xytext=(10, 7.6), arrowprops=arrow_style)
    
    # Down arrow from spectrogram
    ax.annotate('', xy=(12.25, 5.7), xytext=(12.25, 7), arrowprops=arrow_style)
    
    # Middle row arrows (right to left for model)
    ax.annotate('', xy=(11, 5.1), xytext=(10, 5.1), arrowprops=arrow_style)
    ax.annotate('', xy=(7.5, 5.1), xytext=(6.5, 5.1), arrowprops=arrow_style)
    
    # Grad-CAM connection
    ax.annotate('', xy=(4, 5.1), xytext=(3, 5.1), arrowprops=arrow_style)
    
    # Down arrows to output
    ax.annotate('', xy=(5.25, 3.2), xytext=(5.25, 4.5), arrowprops=arrow_style)
    ax.annotate('', xy=(8.75, 3.2), xytext=(8.75, 4.5), arrowprops=arrow_style)
    ax.annotate('', xy=(12.25, 3.2), xytext=(12.25, 4.5), arrowprops=arrow_style)
    
    # Add stage labels
    ax.text(8, 8.5, 'PREPROCESSING STAGE', fontsize=12, ha='center', 
            color='gray', style='italic')
    ax.text(8, 6, 'MODEL INFERENCE STAGE', fontsize=12, ha='center', 
            color='gray', style='italic')
    ax.text(8, 3.5, 'OUTPUT STAGE', fontsize=12, ha='center', 
            color='gray', style='italic')
    
    # Add legend
    legend_items = [
        ('Input/Output', '#E3F2FD'),
        ('Preprocessing', '#FFF3E0'),
        ('Feature Extraction', '#E8F5E9'),
        ('Neural Network', '#E1BEE7'),
        ('Post-processing', '#B3E5FC'),
        ('Explainability', '#FFCDD2')
    ]
    
    for i, (label, color) in enumerate(legend_items):
        rect = FancyBboxPatch((0.5 + i*2.5, 0.3), 0.4, 0.4, 
                               facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(1.1 + i*2.5, 0.5, label, fontsize=8, va='center')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "01_system_architecture.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 2: MEL-SPECTROGRAM EXAMPLE
# ============================================================

def create_spectrogram_example():
    """Create sample mel-spectrogram visualization"""
    print("Creating: Mel-Spectrogram Example...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Create sample audio-like signal
    sr = 22050
    duration = 5
    t = np.linspace(0, duration, sr * duration)
    
    # Simulate bird call (frequency modulated signal)
    # Main call
    freq1 = 3000 + 1000 * np.sin(2 * np.pi * 2 * t)  # Warbling between 2-4 kHz
    call1 = np.sin(2 * np.pi * freq1 * t / sr)
    
    # Add harmonics
    freq2 = 6000 + 2000 * np.sin(2 * np.pi * 2 * t)
    call2 = 0.5 * np.sin(2 * np.pi * freq2 * t / sr)
    
    # Amplitude envelope (bird calls are not continuous)
    envelope = np.zeros_like(t)
    # Add three distinct calls
    for start, end in [(0.5, 1.0), (2.0, 2.5), (3.5, 4.2)]:
        mask = (t >= start) & (t <= end)
        envelope[mask] = np.sin(np.pi * (t[mask] - start) / (end - start))
    
    signal = (call1 + call2) * envelope
    signal += 0.05 * np.random.randn(len(signal))  # Add noise
    
    # Plot 1: Waveform
    axes[0, 0].plot(t, signal, color=COLORS['primary'], linewidth=0.5)
    axes[0, 0].set_xlabel('Time (seconds)', fontsize=12)
    axes[0, 0].set_ylabel('Amplitude', fontsize=12)
    axes[0, 0].set_title('Step 1: Raw Audio Waveform', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlim(0, 5)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Highlight call regions
    for start, end in [(0.5, 1.0), (2.0, 2.5), (3.5, 4.2)]:
        axes[0, 0].axvspan(start, end, alpha=0.2, color=COLORS['success'])
    axes[0, 0].text(0.75, 0.8, 'Bird Call', fontsize=9, ha='center', 
                    transform=axes[0, 0].get_xaxis_transform(), color=COLORS['success'])
    
    # Plot 2: Spectrogram explanation
    axes[0, 1].axis('off')
    axes[0, 1].set_title('Step 2: How Spectrogram Works', fontsize=14, fontweight='bold')
    
    explanation = """
    Converting Audio to Image:
    
    1. Take small windows of audio (2048 samples)
    
    2. Apply Fourier Transform to find frequencies
    
    3. Stack windows side by side = Spectrogram
    
    4. Apply Mel scale (matches human hearing)
    
    5. Convert to decibels (log scale)
    
    Result: 128 × 216 pixel image
    
    Settings Used:
    • Sample Rate: 22,050 Hz
    • FFT Size: 2,048
    • Hop Length: 512
    • Mel Bands: 128
    • Freq Range: 150 - 15,000 Hz
    """
    axes[0, 1].text(0.1, 0.9, explanation, fontsize=11, va='top', 
                    family='monospace', transform=axes[0, 1].transAxes)
    
    # Plot 3: Simulated Mel-spectrogram
    # Create a realistic-looking spectrogram pattern
    n_mels = 128
    n_frames = 216
    spectrogram = np.random.randn(n_mels, n_frames) * 0.3 - 4  # Background noise
    
    # Add bird call patterns
    for call_start, call_end in [(22, 44), (86, 108), (151, 181)]:
        # Main frequency band (around 3-4 kHz, which is mel bands 40-60)
        for frame in range(call_start, call_end):
            center = 50 + 10 * np.sin(2 * np.pi * (frame - call_start) / (call_end - call_start))
            for mel in range(n_mels):
                dist = abs(mel - center)
                if dist < 15:
                    spectrogram[mel, frame] += (15 - dist) / 15 * 3
            
            # Add harmonic
            center2 = center * 1.5
            if center2 < n_mels:
                for mel in range(n_mels):
                    dist = abs(mel - center2)
                    if dist < 10:
                        spectrogram[mel, frame] += (10 - dist) / 10 * 1.5
    
    im = axes[1, 0].imshow(spectrogram, aspect='auto', origin='lower', 
                            cmap='magma', vmin=-5, vmax=2)
    axes[1, 0].set_xlabel('Time Frames', fontsize=12)
    axes[1, 0].set_ylabel('Mel Frequency Bands', fontsize=12)
    axes[1, 0].set_title('Step 3: Mel-Spectrogram Result', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1, 0])
    cbar.set_label('Power (dB)', fontsize=10)
    
    # Add annotations
    axes[1, 0].annotate('Bird Call 1', xy=(33, 55), xytext=(33, 100),
                        fontsize=9, ha='center', color='white',
                        arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    axes[1, 0].annotate('Bird Call 2', xy=(97, 55), xytext=(97, 100),
                        fontsize=9, ha='center', color='white',
                        arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    axes[1, 0].annotate('Bird Call 3', xy=(166, 55), xytext=(166, 100),
                        fontsize=9, ha='center', color='white',
                        arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    
    # Plot 4: 3-channel version for CNN
    axes[1, 1].set_title('Step 4: 3-Channel Input for CNN', fontsize=14, fontweight='bold')
    
    # Show RGB representation
    rgb_spec = np.zeros((n_mels, n_frames, 3))
    normalized = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
    rgb_spec[:, :, 0] = normalized  # R
    rgb_spec[:, :, 1] = normalized  # G  
    rgb_spec[:, :, 2] = normalized  # B
    
    axes[1, 1].imshow(rgb_spec, aspect='auto', origin='lower')
    axes[1, 1].set_xlabel('Time Frames (216)', fontsize=12)
    axes[1, 1].set_ylabel('Mel Bands (128)', fontsize=12)
    axes[1, 1].text(108, 135, 'Same spectrogram copied to R, G, B channels\nFinal shape: (3, 128, 216)', 
                    ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "02_spectrogram_conversion.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 3: TRAINING CURVES
# ============================================================

def create_training_curves():
    """Create training loss and accuracy curves"""
    print("Creating: Training Curves...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = TRAINING_DATA['epochs']
    
    # Plot 1: Loss curves
    axes[0].plot(epochs, TRAINING_DATA['train_loss'], 'b-', linewidth=2, 
                 marker='o', markersize=4, label='Training Loss')
    axes[0].plot(epochs, TRAINING_DATA['val_loss'], 'r-', linewidth=2, 
                 marker='s', markersize=4, label='Validation Loss')
    axes[0].axvline(x=10, color='green', linestyle='--', linewidth=2, label='Best Model (Epoch 10)')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (Cross-Entropy)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 18)
    
    # Add annotation for overfitting
    axes[0].annotate('Validation loss\nstarts increasing\n(overfitting)', 
                     xy=(10, 1.15), xytext=(13, 1.8),
                     fontsize=9, ha='center',
                     arrowprops=dict(arrowstyle='->', color='red'))
    
    # Plot 2: Accuracy curves
    axes[1].plot(epochs, TRAINING_DATA['train_acc'], 'b-', linewidth=2, 
                 marker='o', markersize=4, label='Training Accuracy')
    axes[1].plot(epochs, TRAINING_DATA['val_acc'], 'r-', linewidth=2, 
                 marker='s', markersize=4, label='Validation Accuracy')
    axes[1].axvline(x=10, color='green', linestyle='--', linewidth=2, label='Best Model (Epoch 10)')
    axes[1].axhline(y=68.79, color='orange', linestyle=':', linewidth=1.5, label='Best Val Acc: 68.79%')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10, loc='lower right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 18)
    axes[1].set_ylim(0, 105)
    
    # Add gap annotation
    axes[1].annotate('', xy=(17, 99.83), xytext=(17, 66.5),
                     arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    axes[1].text(17.5, 83, 'Overfitting\nGap: 31%', fontsize=9, color='purple')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "03_training_curves.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 4: MODEL COMPARISON - ACCURACY
# ============================================================

def create_model_accuracy_comparison():
    """Create bar chart comparing model accuracies"""
    print("Creating: Model Accuracy Comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = MODEL_DATA['names']
    x = np.arange(len(models))
    width = 0.25
    
    # Plot 1: Top-1, Top-3, Top-5 Accuracy
    bars1 = axes[0].bar(x - width, MODEL_DATA['top1_accuracy'], width, 
                        label='Top-1 Accuracy', color=COLORS['primary'])
    bars2 = axes[0].bar(x, MODEL_DATA['top3_accuracy'], width, 
                        label='Top-3 Accuracy', color=COLORS['success'])
    bars3 = axes[0].bar(x + width, MODEL_DATA['top5_accuracy'], width, 
                        label='Top-5 Accuracy', color=COLORS['warning'])
    
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha='right')
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=7)
    
    # Highlight winner
    axes[0].annotate('WINNER', xy=(4, 72.26), xytext=(4, 78),
                     fontsize=10, ha='center', color=COLORS['primary'], fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=COLORS['primary']))
    
    # Plot 2: Horizontal bar chart with Top-1 only (cleaner view)
    y_pos = np.arange(len(models))
    colors = MODEL_DATA['colors']
    
    bars = axes[1].barh(y_pos, MODEL_DATA['top1_accuracy'], color=colors, edgecolor='black')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(models)
    axes[1].set_xlabel('Top-1 Accuracy (%)', fontsize=12)
    axes[1].set_title('Top-1 Accuracy Ranking', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 85)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, MODEL_DATA['top1_accuracy'])):
        axes[1].text(acc + 1, i, f'{acc}%', va='center', fontsize=10, fontweight='bold')
    
    # Add rank labels
    ranks = [5, 4, 3, 2, 1]  # Based on accuracy
    for i, rank in enumerate(ranks):
        axes[1].text(2, i, f'#{rank}', va='center', ha='left', fontsize=10, 
                     color='white', fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "04_model_accuracy_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 5: MODEL COMPARISON - EFFICIENCY
# ============================================================

def create_model_efficiency_comparison():
    """Create charts comparing model efficiency (size, speed, parameters)"""
    print("Creating: Model Efficiency Comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = MODEL_DATA['names']
    colors = MODEL_DATA['colors']
    
    # Plot 1: Parameters (millions)
    axes[0, 0].bar(models, MODEL_DATA['parameters'], color=colors, edgecolor='black')
    axes[0, 0].set_ylabel('Parameters (Millions)', fontsize=12)
    axes[0, 0].set_title('Model Size: Number of Parameters', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticklabels(models, rotation=15, ha='right')
    
    for i, (model, param) in enumerate(zip(models, MODEL_DATA['parameters'])):
        axes[0, 0].text(i, param + 3, f'{param}M', ha='center', fontsize=10, fontweight='bold')
    
    # Add annotation for VGG
    axes[0, 0].annotate('Very Large!', xy=(0, 138), xytext=(1, 120),
                        fontsize=9, color='red',
                        arrowprops=dict(arrowstyle='->', color='red'))
    
    # Plot 2: Model file size (MB)
    axes[0, 1].bar(models, MODEL_DATA['size_mb'], color=colors, edgecolor='black')
    axes[0, 1].set_ylabel('File Size (MB)', fontsize=12)
    axes[0, 1].set_title('Model Size: File Size on Disk', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticklabels(models, rotation=15, ha='right')
    
    for i, (model, size) in enumerate(zip(models, MODEL_DATA['size_mb'])):
        axes[0, 1].text(i, size + 10, f'{size} MB', ha='center', fontsize=10, fontweight='bold')
    
    # Plot 3: Inference time
    axes[1, 0].bar(models, MODEL_DATA['inference_time'], color=colors, edgecolor='black')
    axes[1, 0].set_ylabel('Inference Time (seconds)', fontsize=12)
    axes[1, 0].set_title('Speed: Time per 5-second Chunk', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticklabels(models, rotation=15, ha='right')
    
    for i, (model, time) in enumerate(zip(models, MODEL_DATA['inference_time'])):
        axes[1, 0].text(i, time + 0.05, f'{time}s', ha='center', fontsize=10, fontweight='bold')
    
    # Plot 4: Scatter plot - Accuracy vs Parameters (Efficiency)
    scatter = axes[1, 1].scatter(MODEL_DATA['parameters'], MODEL_DATA['top1_accuracy'], 
                                  s=[s*2 for s in MODEL_DATA['size_mb']], 
                                  c=colors, edgecolor='black', linewidth=2, alpha=0.7)
    
    for i, model in enumerate(models):
        axes[1, 1].annotate(model, 
                            (MODEL_DATA['parameters'][i], MODEL_DATA['top1_accuracy'][i]),
                            xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    axes[1, 1].set_xlabel('Parameters (Millions)', fontsize=12)
    axes[1, 1].set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    axes[1, 1].set_title('Accuracy vs Efficiency\n(Bubble size = File size)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add "sweet spot" region
    rect = plt.Rectangle((2, 70), 8, 5, fill=True, alpha=0.2, color='green')
    axes[1, 1].add_patch(rect)
    axes[1, 1].text(6, 72.5, 'Optimal Zone:\nHigh Accuracy,\nLow Parameters', 
                    ha='center', fontsize=9, color='green')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "05_model_efficiency_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 6: CONFUSION MATRIX
# ============================================================

def create_confusion_matrix():
    """Create confusion matrix visualization"""
    print("Creating: Confusion Matrix...")
    
    # Create synthetic confusion matrix for visualization
    n_classes = 54
    np.random.seed(42)
    
    # Start with identity-ish matrix (diagonal dominance)
    cm = np.zeros((n_classes, n_classes))
    
    # Generate realistic confusion patterns
    for i in range(n_classes):
        # Most predictions correct (on diagonal)
        cm[i, i] = np.random.randint(70, 95)
        
        # Some confusion with nearby classes
        total_off_diag = 100 - cm[i, i]
        
        # Distribute confusion
        n_confused = np.random.randint(2, 5)
        confused_classes = np.random.choice([j for j in range(n_classes) if j != i], 
                                            size=n_confused, replace=False)
        
        confusion_vals = np.random.dirichlet(np.ones(n_confused)) * total_off_diag
        for j, conf in zip(confused_classes, confusion_vals):
            cm[i, j] = conf
    
    # Normalize rows to percentages
    cm = cm / cm.sum(axis=1, keepdims=True) * 100
    
    # Create figure with two confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Full 54x54 matrix (simplified view)
    im1 = axes[0].imshow(cm, cmap='Blues', aspect='auto')
    axes[0].set_xlabel('Predicted Species', fontsize=12)
    axes[0].set_ylabel('Actual Species', fontsize=12)
    axes[0].set_title(f'Full Confusion Matrix ({n_classes}x{n_classes} Species)', 
                      fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Prediction Rate (%)', fontsize=10)
    
    # Add annotation
    axes[0].text(27, -3, 'Strong diagonal = Good accuracy\nOff-diagonal spots = Confusion pairs', 
                 ha='center', fontsize=10, style='italic')
    
    # Plot 2: Top 10 species subset (more readable)
    species_names = [s[0].split()[-1] for s in SPECIES_DATA['top_species']]  # Last word of name
    top10_cm = cm[:10, :10]
    
    im2 = axes[1].imshow(top10_cm, cmap='Blues', aspect='auto')
    axes[1].set_xticks(range(10))
    axes[1].set_yticks(range(10))
    axes[1].set_xticklabels(species_names, rotation=45, ha='right', fontsize=9)
    axes[1].set_yticklabels(species_names, fontsize=9)
    axes[1].set_xlabel('Predicted Species', fontsize=12)
    axes[1].set_ylabel('Actual Species', fontsize=12)
    axes[1].set_title('Top 10 Species Confusion Matrix\n(Subset for Clarity)', 
                      fontsize=14, fontweight='bold')
    
    # Add text annotations in cells
    for i in range(10):
        for j in range(10):
            value = top10_cm[i, j]
            if value > 0.5:
                color = 'white' if value > 50 else 'black'
                axes[1].text(j, i, f'{value:.0f}', ha='center', va='center', 
                            fontsize=8, color=color)
    
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Prediction Rate (%)', fontsize=10)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "06_confusion_matrix.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 7: SPECIES ACCURACY CHART
# ============================================================

def create_species_accuracy_chart():
    """Create chart showing best and worst performing species"""
    print("Creating: Species Accuracy Chart...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Plot 1: Top 10 best species
    top_names = [s[0] for s in SPECIES_DATA['top_species']]
    top_acc = [s[1] for s in SPECIES_DATA['top_species']]
    top_samples = [s[2] for s in SPECIES_DATA['top_species']]
    
    y_pos = np.arange(len(top_names))
    bars1 = axes[0].barh(y_pos, top_acc, color=COLORS['success'], edgecolor='black')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top_names, fontsize=10)
    axes[0].set_xlabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Top 10 Best Performing Species', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 110)
    axes[0].axvline(x=72.26, color='gray', linestyle='--', label='Average (72.26%)')
    axes[0].legend(fontsize=9)
    
    # Add accuracy and sample count labels
    for i, (bar, acc, samples) in enumerate(zip(bars1, top_acc, top_samples)):
        axes[0].text(acc + 1, i, f'{acc}% (n={samples})', va='center', fontsize=9)
    
    # Plot 2: Bottom 5 worst species
    bottom_names = [s[0] for s in SPECIES_DATA['bottom_species']]
    bottom_acc = [s[1] for s in SPECIES_DATA['bottom_species']]
    bottom_samples = [s[2] for s in SPECIES_DATA['bottom_species']]
    
    y_pos = np.arange(len(bottom_names))
    bars2 = axes[1].barh(y_pos, bottom_acc, color=COLORS['danger'], edgecolor='black')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(bottom_names, fontsize=10)
    axes[1].set_xlabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Bottom 5 Worst Performing Species', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 110)
    axes[1].axvline(x=72.26, color='gray', linestyle='--', label='Average (72.26%)')
    axes[1].legend(fontsize=9)
    
    # Add accuracy and sample count labels
    for i, (bar, acc, samples) in enumerate(zip(bars2, bottom_acc, bottom_samples)):
        axes[1].text(acc + 1, i, f'{acc}% (n={samples})', va='center', fontsize=9)
        # Highlight low sample count
        if samples < 150:
            axes[1].text(70, i, '⚠️ Low samples!', va='center', fontsize=8, color='red')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "07_species_accuracy.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 8: GRAD-CAM VISUALIZATION
# ============================================================

def create_gradcam_visualization():
    """Create Grad-CAM example visualizations"""
    print("Creating: Grad-CAM Visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    n_mels, n_frames = 128, 216
    
    # Create sample spectrogram with bird call
    np.random.seed(42)
    spec = np.random.randn(n_mels, n_frames) * 0.3 - 4
    
    # Add bird call pattern (similar to earlier)
    for call_start, call_end in [(50, 90), (130, 170)]:
        for frame in range(call_start, call_end):
            center = 50 + 15 * np.sin(2 * np.pi * (frame - call_start) / (call_end - call_start))
            for mel in range(n_mels):
                dist = abs(mel - center)
                if dist < 15:
                    spec[mel, frame] += (15 - dist) / 15 * 4
    
    # Create Grad-CAM heatmaps
    # Correct focus (on bird call)
    gradcam_correct = np.zeros((n_mels, n_frames))
    for frame in range(45, 95):
        for mel in range(35, 75):
            center_dist = np.sqrt((frame - 70)**2 + (mel - 55)**2)
            gradcam_correct[mel, frame] = max(0, 1 - center_dist / 30)
    
    for frame in range(125, 175):
        for mel in range(35, 75):
            center_dist = np.sqrt((frame - 150)**2 + (mel - 55)**2)
            gradcam_correct[mel, frame] = max(0, 1 - center_dist / 30)
    
    # Wrong focus (on noise/wrong area)
    gradcam_wrong = np.zeros((n_mels, n_frames))
    for frame in range(10, 40):
        for mel in range(80, 120):
            center_dist = np.sqrt((frame - 25)**2 + (mel - 100)**2)
            gradcam_wrong[mel, frame] = max(0, 1 - center_dist / 25)
    
    # Diffuse/uncertain focus
    gradcam_uncertain = np.random.rand(n_mels, n_frames) * 0.3
    gradcam_uncertain = np.clip(gradcam_uncertain, 0, 1)
    
    # Row 1: Correct prediction
    axes[0, 0].imshow(spec, aspect='auto', origin='lower', cmap='magma')
    axes[0, 0].set_title('Original Spectrogram', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Mel Bands', fontsize=10)
    axes[0, 0].text(108, 135, 'Blue-winged Warbler Recording', ha='center', fontsize=9, color='white')
    
    axes[0, 1].imshow(gradcam_correct, aspect='auto', origin='lower', cmap='jet')
    axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[0, 1].text(108, 135, 'Red = Important, Blue = Ignored', ha='center', fontsize=9, color='white')
    
    # Overlay
    spec_norm = (spec - spec.min()) / (spec.max() - spec.min())
    overlay = np.zeros((n_mels, n_frames, 3))
    overlay[:, :, 0] = gradcam_correct  # Red channel from Grad-CAM
    overlay[:, :, 1] = spec_norm * 0.5  # Green channel from spectrogram
    overlay[:, :, 2] = spec_norm * 0.5  # Blue channel from spectrogram
    
    axes[0, 2].imshow(overlay, aspect='auto', origin='lower')
    axes[0, 2].set_title('Overlay: ✅ CORRECT Focus', fontsize=12, fontweight='bold', color='green')
    axes[0, 2].text(108, 135, 'Prediction: Blue-winged Warbler (97%)', ha='center', fontsize=9, color='white')
    axes[0, 2].annotate('Model focuses\non bird calls!', xy=(70, 55), xytext=(70, 100),
                        fontsize=9, ha='center', color='white',
                        arrowprops=dict(arrowstyle='->', color='white'))
    
    # Row 2: Wrong prediction
    axes[1, 0].imshow(spec, aspect='auto', origin='lower', cmap='magma')
    axes[1, 0].set_title('Same Spectrogram', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time Frames', fontsize=10)
    axes[1, 0].set_ylabel('Mel Bands', fontsize=10)
    
    axes[1, 1].imshow(gradcam_wrong, aspect='auto', origin='lower', cmap='jet')
    axes[1, 1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Time Frames', fontsize=10)
    
    # Wrong overlay
    overlay_wrong = np.zeros((n_mels, n_frames, 3))
    overlay_wrong[:, :, 0] = gradcam_wrong
    overlay_wrong[:, :, 1] = spec_norm * 0.5
    overlay_wrong[:, :, 2] = spec_norm * 0.5
    
    axes[1, 2].imshow(overlay_wrong, aspect='auto', origin='lower')
    axes[1, 2].set_title('Overlay: ❌ WRONG Focus', fontsize=12, fontweight='bold', color='red')
    axes[1, 2].set_xlabel('Time Frames', fontsize=10)
    axes[1, 2].text(108, 135, 'Prediction: Bald Eagle (95%) - WRONG!', ha='center', fontsize=9, color='white')
    axes[1, 2].annotate('Model focuses on\nnoise, not bird!', xy=(25, 100), xytext=(80, 100),
                        fontsize=9, ha='center', color='white',
                        arrowprops=dict(arrowstyle='->', color='white'))
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "08_gradcam_visualization.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 9: DATA DISTRIBUTION
# ============================================================

def create_data_distribution():
    """Create charts showing data distribution"""
    print("Creating: Data Distribution Charts...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sample species distribution data
    np.random.seed(42)
    species_counts = np.random.lognormal(mean=5.5, sigma=0.8, size=54).astype(int)
    species_counts = np.clip(species_counts, 98, 1253)
    species_counts = np.sort(species_counts)[::-1]
    
    # Plot 1: Species sample distribution (bar chart)
    axes[0, 0].bar(range(54), species_counts, color=COLORS['primary'], edgecolor='none')
    axes[0, 0].set_xlabel('Species (sorted by count)', fontsize=12)
    axes[0, 0].set_ylabel('Number of Training Samples', fontsize=12)
    axes[0, 0].set_title('Class Imbalance: Samples per Species', fontsize=14, fontweight='bold')
    axes[0, 0].axhline(y=np.mean(species_counts), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(species_counts):.0f}')
    axes[0, 0].legend(fontsize=10)
    
    # Annotate max and min
    axes[0, 0].annotate(f'Max: 1,253', xy=(0, 1253), xytext=(10, 1100),
                        fontsize=9, arrowprops=dict(arrowstyle='->', color='green'))
    axes[0, 0].annotate(f'Min: 98', xy=(53, 98), xytext=(45, 300),
                        fontsize=9, arrowprops=dict(arrowstyle='->', color='red'))
    
    # Plot 2: Train/Val/Test split
    split_labels = ['Training\n(70%)', 'Validation\n(15%)', 'Test\n(15%)']
    split_sizes = [23839, 5568, 5404]
    split_colors = [COLORS['primary'], COLORS['warning'], COLORS['success']]
    
    wedges, texts, autotexts = axes[0, 1].pie(split_sizes, labels=split_labels, 
                                               colors=split_colors, autopct='%1.1f%%',
                                               explode=(0.05, 0, 0), shadow=True,
                                               textprops={'fontsize': 11})
    axes[0, 1].set_title('Data Split Distribution', fontsize=14, fontweight='bold')
    
    # Add counts
    for i, (size, wedge) in enumerate(zip(split_sizes, wedges)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = 0.7 * np.cos(np.radians(angle))
        y = 0.7 * np.sin(np.radians(angle))
        axes[0, 1].annotate(f'n={size:,}', xy=(x, y), ha='center', fontsize=9, 
                            color='white', fontweight='bold')
    
    # Plot 3: Recording to chunks conversion
    stages = ['Original\nRecordings', 'With\nMetadata', 'Audio\nChunks']
    counts = [4521, 1926, 34811]
    
    bars = axes[1, 0].bar(stages, counts, color=[COLORS['secondary'], COLORS['warning'], COLORS['success']], 
                          edgecolor='black')
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('Data Pipeline: From Raw to Processed', fontsize=14, fontweight='bold')
    
    for bar, count in zip(bars, counts):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                        f'{count:,}', ha='center', fontsize=12, fontweight='bold')
    
    # Add arrows showing transformation
    axes[1, 0].annotate('', xy=(1, 2500), xytext=(0, 4000),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    axes[1, 0].annotate('', xy=(2, 20000), xytext=(1, 2000),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    axes[1, 0].text(0.5, 3500, '57% lost\n(no metadata)', ha='center', fontsize=9, color='red')
    axes[1, 0].text(1.5, 12000, '18x increase\n(5-sec chunks)', ha='center', fontsize=9, color='green')
    
    # Plot 4: Chunk duration histogram (simulated)
    durations = np.concatenate([
        np.random.normal(5.0, 0.01, 30000),  # Most are exactly 5 seconds
        np.random.uniform(3.0, 5.0, 4811)    # Some shorter (end of recordings)
    ])
    
    axes[1, 1].hist(durations, bins=50, color=COLORS['info'], edgecolor='black')
    axes[1, 1].set_xlabel('Chunk Duration (seconds)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Distribution of Audio Chunk Durations', fontsize=14, fontweight='bold')
    axes[1, 1].axvline(x=5.0, color='red', linestyle='--', linewidth=2, label='Target: 5 seconds')
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "09_data_distribution.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 10: MODEL ARCHITECTURE DIAGRAMS
# ============================================================

def create_architecture_diagrams():
    """Create simplified architecture diagrams for each model"""
    print("Creating: Model Architecture Diagrams...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Common settings
    box_height = 0.15
    
    def draw_block(ax, x, y, width, height, label, color, fontsize=8):
        rect = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02",
                               facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, label, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold')
    
    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # =============== VGG-16 ===============
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('VGG-16 Architecture\n(138M Parameters)', fontsize=14, fontweight='bold')
    
    # VGG blocks
    y = 8.5
    colors_vgg = ['#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3']
    labels_vgg = ['Conv3x3 x2\n64 filters', 'Conv3x3 x2\n128 filters', 
                  'Conv3x3 x3\n256 filters', 'Conv3x3 x3\n512 filters', 
                  'Conv3x3 x3\n512 filters']
    
    for i, (label, color) in enumerate(zip(labels_vgg, colors_vgg)):
        draw_block(ax, 1, y - i*1.3, 3, 1, label, color)
        draw_block(ax, 4.5, y - i*1.3, 1.5, 1, 'MaxPool', '#E0E0E0')
        if i < 4:
            draw_arrow(ax, 5.25, y - i*1.3 - 0.5, 2.5, y - (i+1)*1.3 + 1)
    
    # FC layers
    draw_block(ax, 7, 6, 2, 0.8, 'Flatten', '#FFF9C4')
    draw_block(ax, 7, 4.5, 2, 0.8, 'Dense 4096', '#FFCC80')
    draw_block(ax, 7, 3, 2, 0.8, 'Dense 4096', '#FFCC80')
    draw_block(ax, 7, 1.5, 2, 0.8, 'Dense 54\n(Output)', '#A5D6A7')
    
    ax.text(5, 0.5, 'Simple but HUGE!', fontsize=10, ha='center', color='red', style='italic')
    
    # =============== ResNet-50 ===============
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('ResNet-50 Architecture\n(25.6M Parameters)', fontsize=14, fontweight='bold')
    
    # Show skip connection concept
    draw_block(ax, 1, 8, 2.5, 0.8, 'Input', '#E3F2FD')
    draw_block(ax, 1, 6.5, 2.5, 0.8, 'Conv 1x1\n(reduce)', '#BBDEFB')
    draw_block(ax, 1, 5, 2.5, 0.8, 'Conv 3x3', '#90CAF9')
    draw_block(ax, 1, 3.5, 2.5, 0.8, 'Conv 1x1\n(expand)', '#64B5F6')
    draw_block(ax, 1, 2, 2.5, 0.8, 'ADD', '#C8E6C9')
    draw_block(ax, 1, 0.5, 2.5, 0.8, 'Output', '#A5D6A7')
    
    # Skip connection
    ax.annotate('', xy=(3.5, 2.4), xytext=(3.5, 8.4),
                arrowprops=dict(arrowstyle='->', color='red', lw=3, 
                               connectionstyle='arc3,rad=0.3'))
    ax.text(5, 5, 'SKIP\nCONNECTION', fontsize=11, ha='center', color='red', fontweight='bold')
    
    # Info box
    ax.text(7, 8, 'Key Innovation:\n\nOutput = F(x) + x\n\nAllows training\nvery deep networks!', 
            fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # =============== MobileNetV2 ===============
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('MobileNetV2 Architecture\n(3.4M Parameters)', fontsize=14, fontweight='bold')
    
    # Inverted residual block
    draw_block(ax, 1, 8, 2.5, 0.8, 'Input\n(64 ch)', '#E3F2FD')
    draw_block(ax, 1, 6.5, 2.5, 0.8, 'Conv 1x1\nEXPAND (384)', '#FFCDD2')
    draw_block(ax, 1, 5, 2.5, 0.8, 'Depthwise\nConv 3x3', '#F8BBD9')
    draw_block(ax, 1, 3.5, 2.5, 0.8, 'Conv 1x1\nCOMPRESS (64)', '#E1BEE7')
    draw_block(ax, 1, 2, 2.5, 0.8, 'ADD', '#C8E6C9')
    
    # Skip connection
    ax.annotate('', xy=(3.5, 2.4), xytext=(3.5, 8.4),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2,
                               connectionstyle='arc3,rad=0.3'))
    
    # Depthwise explanation
    ax.text(6, 8, 'Depthwise Separable:\n\nNormal Conv: 147,456 params\nDepthwise: 17,536 params\n\n8x fewer parameters!', 
            fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='#FCE4EC', alpha=0.7))
    
    ax.text(5, 0.5, 'Designed for mobile phones', fontsize=10, ha='center', 
            color='purple', style='italic')
    
    # =============== DenseNet-121 ===============
    ax = axes[3]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('DenseNet-121 Architecture\n(8.0M Parameters)', fontsize=14, fontweight='bold')
    
    # Dense connections
    y_positions = [8, 6.5, 5, 3.5, 2]
    labels_dense = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5']
    colors_dense = ['#B3E5FC', '#81D4FA', '#4FC3F7', '#29B6F6', '#03A9F4']
    
    for i, (y, label, color) in enumerate(zip(y_positions, labels_dense, colors_dense)):
        draw_block(ax, 1, y, 2, 0.8, label, color)
    
    # Draw dense connections
    for i in range(5):
        for j in range(i+1, 5):
            ax.plot([3, 4 + (j-i)*0.3, 1], 
                    [y_positions[i]+0.4, (y_positions[i]+y_positions[j])/2+0.4, y_positions[j]+0.4],
                    'g-', linewidth=1, alpha=0.5)
    
    ax.text(6, 7, 'Dense Connections:\n\nEvery layer connects\nto ALL previous layers\n\nBetter gradient flow\nFeature reuse', 
            fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.7))
    
    # =============== EfficientNet-B0 ===============
    ax = axes[4]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('EfficientNet-B0 Architecture\n(4.0M Parameters) ⭐ WINNER', 
                 fontsize=14, fontweight='bold', color='green')
    
    # MBConv block with SE
    draw_block(ax, 0.5, 8.5, 2, 0.7, 'Input', '#E3F2FD')
    draw_block(ax, 0.5, 7.3, 2, 0.7, 'Conv 1x1\nExpand', '#BBDEFB')
    draw_block(ax, 0.5, 6.1, 2, 0.7, 'Depthwise\nConv', '#90CAF9')
    draw_block(ax, 0.5, 4.9, 2, 0.7, 'Squeeze &\nExcitation', '#FFAB91')
    draw_block(ax, 0.5, 3.7, 2, 0.7, 'Conv 1x1\nProject', '#64B5F6')
    draw_block(ax, 0.5, 2.5, 2, 0.7, 'ADD', '#C8E6C9')
    
    # SE explanation
    ax.text(3.5, 7.5, 'Squeeze-and-Excitation:\n\nLearn which channels\nare important for\neach input!\n\n"Attention" mechanism', 
            fontsize=9, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='#FBE9E7', alpha=0.7))
    
    # Compound scaling
    ax.text(3.5, 3.5, 'Compound Scaling:\n\nScale depth, width,\nand resolution\nTOGETHER optimally', 
            fontsize=9, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.7))
    
    # Why it won
    ax.text(5, 0.8, '✓ Best accuracy (72.26%)\n✓ Small size (17MB)\n✓ Fast inference (0.5s)', 
            fontsize=10, ha='center', color='green', fontweight='bold')
    
    # =============== Summary Comparison ===============
    ax = axes[5]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Architecture Evolution Summary', fontsize=14, fontweight='bold')
    
    # Timeline
    years = [2014, 2015, 2017, 2018, 2019]
    models = ['VGG-16', 'ResNet-50', 'DenseNet-121', 'MobileNetV2', 'EfficientNet-B0']
    innovations = ['Deep stacking', 'Skip connections', 'Dense connections', 
                   'Depthwise conv', 'Compound scaling']
    colors_timeline = ['#DC3545', '#28A745', '#A23B72', '#FFC107', '#2E86AB']
    
    # Draw timeline
    ax.plot([1, 9], [5, 5], 'k-', linewidth=3)
    
    for i, (year, model, innov, color) in enumerate(zip(years, models, innovations, colors_timeline)):
        x = 1 + i * 2
        ax.plot([x, x], [4.7, 5.3], 'k-', linewidth=2)
        ax.text(x, 5.5, str(year), ha='center', fontsize=10, fontweight='bold')
        
        rect = FancyBboxPatch((x-0.8, 6), 1.6, 1.5, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, 7.2, model, ha='center', fontsize=9, fontweight='bold', color='white')
        ax.text(x, 6.5, innov, ha='center', fontsize=7, color='white')
        
        # Arrow to timeline
        ax.annotate('', xy=(x, 5.3), xytext=(x, 6),
                    arrowprops=dict(arrowstyle='->', color='black'))
    
    # Accuracy trend arrow
    ax.annotate('Accuracy improving over time →', xy=(7, 4), xytext=(2, 4),
                fontsize=11, ha='left', color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Key insight
    ax.text(5, 1.5, 'Key Insight: Newer architectures achieve BETTER accuracy\nwith FEWER parameters by using smarter designs!', 
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "10_model_architectures.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 11: PREPROCESSING PIPELINE
# ============================================================

def create_preprocessing_pipeline():
    """Create preprocessing pipeline visualization"""
    print("Creating: Preprocessing Pipeline...")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.set_title('Audio Preprocessing Pipeline', fontsize=18, fontweight='bold', y=1.02)
    
    # Step boxes
    steps = [
        (0.5, 5, 'Raw Audio\n(Various formats)', '#FFCDD2', 
         'WAV, MP3, OGG\nDifferent sample rates\nMono or Stereo'),
        (3.5, 5, 'Resample\n22,050 Hz', '#F8BBD0', 
         'Standard rate\nCaptures all bird\nfrequencies'),
        (6.5, 5, 'Convert to\nMono', '#E1BEE7', 
         'Single channel\nSimpler processing'),
        (9.5, 5, 'Chunk into\n5-sec segments', '#D1C4E9', 
         '50% overlap\nNo calls split'),
        (12.5, 5, 'Generate\nSpectrogram', '#C5CAE9', 
         '128 mel bands\n216 time frames'),
    ]
    
    for x, y, title, color, desc in steps:
        # Main box
        rect = FancyBboxPatch((x, y), 2.5, 2, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 1.25, y + 1.5, title, ha='center', va='center', 
                fontsize=11, fontweight='bold')
        
        # Description box below
        ax.text(x + 1.25, y - 0.5, desc, ha='center', va='top', 
                fontsize=9, style='italic')
    
    # Arrows
    for i in range(4):
        ax.annotate('', xy=(3.5 + i*3, 6), xytext=(3 + i*3, 6),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Before/After comparison at bottom
    ax.text(4, 2.5, 'BEFORE', fontsize=12, fontweight='bold', ha='center')
    ax.text(4, 2, '45-second recording\n48,000 Hz stereo\nMP3 format', 
            fontsize=10, ha='center')
    
    ax.text(8, 2.5, '→', fontsize=20, ha='center')
    
    ax.text(12, 2.5, 'AFTER', fontsize=12, fontweight='bold', ha='center')
    ax.text(12, 2, '17 chunks × 5 seconds\n22,050 Hz mono\n128×216 spectrograms', 
            fontsize=10, ha='center')
    
    # Stats box
    stats_text = """Processing Statistics:
• 1,926 recordings → 34,811 chunks
• Average 18 chunks per recording
• 50% overlap prevents missing calls
• All chunks normalized to same format"""
    
    ax.text(8, 0.5, stats_text, fontsize=10, ha='center', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "11_preprocessing_pipeline.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 12: TIMELINE DETECTION EXAMPLE
# ============================================================

def create_timeline_detection():
    """Create timeline detection visualization"""
    print("Creating: Timeline Detection Example...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Simulate a 60-second recording with multiple species
    duration = 60  # seconds
    chunk_duration = 5
    overlap = 2.5
    n_chunks = int((duration - chunk_duration) / overlap) + 1
    
    # Generate detection data for 3 species
    np.random.seed(42)
    species = ['American Robin', 'Blue Jay', 'American Crow']
    colors = [COLORS['success'], COLORS['primary'], COLORS['warning']]
    
    detections = {}
    for sp in species:
        # Random confidence for each chunk
        base = np.random.rand(n_chunks) * 30 + 10  # Low baseline
        # Add some peaks (actual detections)
        peaks = np.random.choice(n_chunks, size=np.random.randint(3, 8), replace=False)
        for p in peaks:
            base[p] = np.random.rand() * 30 + 60  # Higher confidence
        detections[sp] = base
    
    # Plot 1: Stacked timeline
    chunk_starts = np.arange(0, duration - chunk_duration + 0.1, overlap)
    
    for i, (sp, color) in enumerate(zip(species, colors)):
        conf = detections[sp]
        # Plot as filled area
        axes[0].fill_between(chunk_starts, i, i + conf/100, alpha=0.7, color=color, label=sp)
        axes[0].axhline(y=i + 0.8, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    axes[0].set_xlabel('Time (seconds)', fontsize=12)
    axes[0].set_ylabel('Species', fontsize=12)
    axes[0].set_yticks([0.5, 1.5, 2.5])
    axes[0].set_yticklabels(species)
    axes[0].set_title('Detection Timeline: Confidence Over Time\n(Red dashed line = 80% threshold)', 
                      fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].set_xlim(0, duration)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Heatmap style
    all_conf = np.array([detections[sp] for sp in species])
    
    im = axes[1].imshow(all_conf, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    axes[1].set_xlabel('Chunk Number (5-second segments)', fontsize=12)
    axes[1].set_ylabel('Species', fontsize=12)
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(species)
    axes[1].set_title('Detection Heatmap: All Species Over Time\n(Green = High confidence, Red = Low confidence)', 
                      fontsize=14, fontweight='bold')
    
    # Add colorbar
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1], orientation='vertical', pad=0.02)
    cbar.set_label('Confidence (%)', fontsize=10)
    
    # Add threshold line annotation
    axes[1].text(n_chunks + 1, 1, '80% threshold\nfor "Present"', fontsize=9, va='center')
    
    # Mark high confidence detections with circles
    for i, sp in enumerate(species):
        for j, conf in enumerate(detections[sp]):
            if conf >= 80:
                axes[1].plot(j, i, 'wo', markersize=8, markeredgecolor='black', markeredgewidth=2)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "12_timeline_detection.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 13: CONFIDENCE THRESHOLD EXPLANATION
# ============================================================

def create_confidence_threshold():
    """Create visualization explaining confidence threshold"""
    print("Creating: Confidence Threshold Explanation...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Threshold concept
    ax = axes[0]
    
    # Create sample confidence distribution
    np.random.seed(123)
    
    # True positives (actual birds) - higher confidence
    true_pos = np.random.beta(8, 2, 100) * 100
    
    # False positives (not birds or wrong species) - lower confidence  
    false_pos = np.random.beta(2, 5, 100) * 100
    
    # Plot distributions
    bins = np.linspace(0, 100, 30)
    ax.hist(true_pos, bins=bins, alpha=0.7, color=COLORS['success'], 
            label='Correct Detections', edgecolor='black')
    ax.hist(false_pos, bins=bins, alpha=0.7, color=COLORS['danger'], 
            label='Wrong Detections', edgecolor='black')
    
    # Threshold line
    ax.axvline(x=80, color='black', linewidth=3, linestyle='--', label='80% Threshold')
    
    # Shade regions
    ax.axvspan(80, 100, alpha=0.2, color='green')
    ax.axvspan(0, 80, alpha=0.1, color='red')
    
    ax.set_xlabel('Confidence Score (%)', fontsize=12)
    ax.set_ylabel('Number of Predictions', fontsize=12)
    ax.set_title('Why 80% Threshold?\nSeparates Good from Bad Predictions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    # Annotations
    ax.annotate('ACCEPT\n(Mark as Present)', xy=(90, 15), fontsize=11, 
                ha='center', color='green', fontweight='bold')
    ax.annotate('REJECT\n(Mark as Absent/Uncertain)', xy=(40, 15), fontsize=11, 
                ha='center', color='red', fontweight='bold')
    
    # Plot 2: Example predictions
    ax = axes[1]
    ax.axis('off')
    ax.set_title('Example Predictions with 80% Threshold', fontsize=14, fontweight='bold')
    
    examples = [
        ('American Robin', 94.2, 'PRESENT', COLORS['success']),
        ('Blue Jay', 87.5, 'PRESENT', COLORS['success']),
        ('Barred Owl', 72.3, 'UNCERTAIN', COLORS['warning']),
        ('American Crow', 45.1, 'ABSENT', COLORS['danger']),
        ('Bufflehead', 23.8, 'ABSENT', COLORS['danger']),
    ]
    
    y = 0.9
    for species, conf, status, color in examples:
        # Species name
        ax.text(0.05, y, species, fontsize=12, fontweight='bold', 
                transform=ax.transAxes, va='center')
        
        # Confidence bar
        bar_width = conf / 100 * 0.4
        rect = plt.Rectangle((0.35, y - 0.03), bar_width, 0.06, 
                              facecolor=color, edgecolor='black',
                              transform=ax.transAxes)
        ax.add_patch(rect)
        
        # Confidence value
        ax.text(0.35 + bar_width + 0.02, y, f'{conf}%', fontsize=11,
                transform=ax.transAxes, va='center')
        
        # Status
        ax.text(0.85, y, status, fontsize=12, fontweight='bold', color=color,
                transform=ax.transAxes, va='center')
        
        y -= 0.15
    
    # Threshold line
    ax.axvline(x=0.35 + 0.8 * 0.4, ymin=0.1, ymax=0.95, color='black', 
               linewidth=2, linestyle='--', transform=ax.transAxes)
    ax.text(0.35 + 0.8 * 0.4, 0.05, '80%\nThreshold', fontsize=10, ha='center',
            transform=ax.transAxes)
    
    # Legend
    ax.text(0.5, 0.02, '🟢 ≥80%: Present  |  🟡 50-79%: Uncertain  |  🔴 <50%: Absent',
            fontsize=10, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "13_confidence_threshold.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 14: TRANSFER LEARNING CONCEPT
# ============================================================

def create_transfer_learning():
    """Create visualization explaining transfer learning"""
    print("Creating: Transfer Learning Explanation...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.set_title('Transfer Learning: Using Pre-trained Knowledge', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Left side: ImageNet training
    ax.text(3, 9.2, 'Step 1: Pre-trained on ImageNet', fontsize=14, fontweight='bold', ha='center')
    
    # ImageNet box
    rect = FancyBboxPatch((0.5, 6), 5, 2.8, boxstyle="round,pad=0.1",
                           facecolor='#E3F2FD', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(3, 8.2, 'ImageNet Dataset', fontsize=12, fontweight='bold', ha='center')
    ax.text(3, 7.5, '14 million images', fontsize=10, ha='center')
    ax.text(3, 7.0, '1,000 categories', fontsize=10, ha='center')
    ax.text(3, 6.5, '(dogs, cats, cars, planes...)', fontsize=9, ha='center', style='italic')
    
    # Arrow down
    ax.annotate('', xy=(3, 5.5), xytext=(3, 6),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Pre-trained model
    rect = FancyBboxPatch((0.5, 3.5), 5, 1.8, boxstyle="round,pad=0.1",
                           facecolor='#C8E6C9', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(3, 4.8, 'EfficientNet-B0', fontsize=12, fontweight='bold', ha='center')
    ax.text(3, 4.2, 'Learned to detect:', fontsize=10, ha='center')
    ax.text(3, 3.8, 'edges, textures, shapes, patterns', fontsize=9, ha='center', style='italic')
    
    # Right side: Our fine-tuning
    ax.text(11, 9.2, 'Step 2: Fine-tune on Our Data', fontsize=14, fontweight='bold', ha='center')
    
    # Our dataset box
    rect = FancyBboxPatch((8.5, 6), 5, 2.8, boxstyle="round,pad=0.1",
                           facecolor='#FFF3E0', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(11, 8.2, 'Bird Spectrogram Dataset', fontsize=12, fontweight='bold', ha='center')
    ax.text(11, 7.5, '23,839 spectrograms', fontsize=10, ha='center')
    ax.text(11, 7.0, '54 bird species', fontsize=10, ha='center')
    ax.text(11, 6.5, '(robins, jays, owls...)', fontsize=9, ha='center', style='italic')
    
    # Arrow showing transfer
    ax.annotate('', xy=(8.5, 4.4), xytext=(5.5, 4.4),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax.text(7, 4.8, 'Transfer\nKnowledge', fontsize=10, ha='center', color='green', fontweight='bold')
    
    # Fine-tuned model
    rect = FancyBboxPatch((8.5, 3.5), 5, 1.8, boxstyle="round,pad=0.1",
                           facecolor='#E1BEE7', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(11, 4.8, 'Bird Detection Model', fontsize=12, fontweight='bold', ha='center')
    ax.text(11, 4.2, 'Now detects:', fontsize=10, ha='center')
    ax.text(11, 3.8, 'bird call patterns in spectrograms', fontsize=9, ha='center', style='italic')
    
    # What we keep vs replace
    ax.text(7, 2.5, 'What We Keep vs Replace', fontsize=14, fontweight='bold', ha='center')
    
    # Keep box
    rect = FancyBboxPatch((1, 0.5), 5, 1.7, boxstyle="round,pad=0.1",
                           facecolor='#C8E6C9', edgecolor='green', linewidth=2)
    ax.add_patch(rect)
    ax.text(3.5, 1.8, '✓ KEEP: Feature Extraction', fontsize=11, fontweight='bold', 
            ha='center', color='green')
    ax.text(3.5, 1.2, 'Convolutional layers\n(already know how to see)', fontsize=9, ha='center')
    
    # Replace box
    rect = FancyBboxPatch((8, 0.5), 5, 1.7, boxstyle="round,pad=0.1",
                           facecolor='#FFCDD2', edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    ax.text(10.5, 1.8, '✗ REPLACE: Classification Head', fontsize=11, fontweight='bold', 
            ha='center', color='red')
    ax.text(10.5, 1.2, 'Was: 1000 ImageNet classes\nNow: 54 bird species', fontsize=9, ha='center')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "14_transfer_learning.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 15: FINAL SUMMARY INFOGRAPHIC
# ============================================================

def create_summary_infographic():
    """Create final summary infographic"""
    print("Creating: Summary Infographic...")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(8, 11.5, 'Bird Species Detection System - Project Summary', 
            fontsize=20, fontweight='bold', ha='center')
    ax.text(8, 11, 'Confidence-Aware Detection with Explainability', 
            fontsize=14, ha='center', style='italic', color='gray')
    
    # Row 1: Key Numbers
    y = 9.5
    metrics = [
        ('1,926', 'Recordings', COLORS['primary']),
        ('34,811', 'Audio Chunks', COLORS['success']),
        ('54', 'Bird Species', COLORS['warning']),
        ('72.26%', 'Top-1 Accuracy', COLORS['info']),
        ('85.55%', 'Top-5 Accuracy', COLORS['secondary']),
    ]
    
    for i, (value, label, color) in enumerate(metrics):
        x = 1.5 + i * 2.8
        rect = FancyBboxPatch((x - 1, y - 0.8), 2.2, 1.6, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + 0.1, y + 0.2, value, fontsize=16, fontweight='bold', 
                ha='center', color='white')
        ax.text(x + 0.1, y - 0.4, label, fontsize=10, ha='center', color='white')
    
    # Row 2: Model Comparison
    y = 7
    ax.text(8, y + 0.8, 'Model Comparison (Top-1 Accuracy)', fontsize=14, 
            fontweight='bold', ha='center')
    
    models = ['VGG-16', 'MobileNetV2', 'ResNet-50', 'DenseNet-121', 'EfficientNet-B0']
    accs = [61.38, 65.71, 68.42, 69.85, 72.26]
    colors_model = ['#DC3545', '#FFC107', '#28A745', '#A23B72', '#2E86AB']
    
    for i, (model, acc, color) in enumerate(zip(models, accs, colors_model)):
        x = 1 + i * 2.9
        bar_height = acc / 100 * 2
        rect = plt.Rectangle((x, y - 1.5), 2, bar_height, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + 1, y - 1.7, model, fontsize=9, ha='center', rotation=0)
        ax.text(x + 1, y - 1.5 + bar_height + 0.1, f'{acc}%', fontsize=9, 
                ha='center', fontweight='bold')
    
    # Winner annotation
    ax.annotate('WINNER!', xy=(13.9, y - 1.5 + 72.26/100*2), 
                xytext=(14.5, y + 0.3),
                fontsize=10, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    # Row 3: Key Features
    y = 3.5
    ax.text(8, y + 1, 'Key Features', fontsize=14, fontweight='bold', ha='center')
    
    features = [
        ('🎵', 'Audio Analysis', 'WAV, MP3, OGG support'),
        ('🔍', '54 Species', 'North American birds'),
        ('📊', '80% Threshold', 'Confidence-based detection'),
        ('🔥', 'Grad-CAM', 'Visual explanations'),
        ('⚠️', 'Warnings', 'Uncertainty alerts'),
        ('📄', 'PDF Export', 'Professional reports'),
    ]
    
    for i, (icon, title, desc) in enumerate(features):
        x = 0.5 + (i % 3) * 5
        y_pos = y - (i // 3) * 1.3
        
        rect = FancyBboxPatch((x, y_pos - 0.5), 4.5, 1, boxstyle="round,pad=0.05",
                               facecolor='#F5F5F5', edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 0.3, y_pos, icon, fontsize=16, va='center')
        ax.text(x + 1, y_pos + 0.15, title, fontsize=11, fontweight='bold', va='center')
        ax.text(x + 1, y_pos - 0.2, desc, fontsize=9, va='center', color='gray')
    
    # Bottom: Technology Stack
    y = 0.8
    ax.text(8, y + 0.5, 'Technology Stack', fontsize=12, fontweight='bold', ha='center')
    
    tech = ['Python', 'PyTorch', 'Librosa', 'Streamlit', 'EfficientNet-B0', 'Grad-CAM']
    for i, t in enumerate(tech):
        x = 1.5 + i * 2.3
        rect = FancyBboxPatch((x - 0.8, y - 0.3), 1.8, 0.5, boxstyle="round,pad=0.02",
                               facecolor='#E0E0E0', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + 0.1, y, t, fontsize=9, ha='center', va='center')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "15_summary_infographic.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 16: APP SCREENSHOT MOCKUP
# ============================================================

def create_app_mockup():
    """Create Streamlit app mockup/screenshot"""
    print("Creating: App Mockup...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Browser window frame
    rect = FancyBboxPatch((0.2, 0.2), 13.6, 9.6, boxstyle="round,pad=0.02",
                           facecolor='#FAFAFA', edgecolor='#CCCCCC', linewidth=2)
    ax.add_patch(rect)
    
    # Browser top bar
    rect = plt.Rectangle((0.2, 9.2), 13.6, 0.6, facecolor='#F0F0F0', edgecolor='#CCCCCC')
    ax.add_patch(rect)
    
    # Browser buttons
    for i, color in enumerate(['#FF5F56', '#FFBD2E', '#27C93F']):
        circle = plt.Circle((0.6 + i*0.4, 9.5), 0.12, facecolor=color, edgecolor='none')
        ax.add_patch(circle)
    
    # URL bar
    rect = FancyBboxPatch((2, 9.3), 8, 0.4, boxstyle="round,pad=0.02",
                           facecolor='white', edgecolor='#CCCCCC')
    ax.add_patch(rect)
    ax.text(6, 9.5, 'localhost:8501', fontsize=9, ha='center', va='center', color='gray')
    
    # App title
    ax.text(7, 8.7, '🐦 Bird Species Detection System', fontsize=16, 
            fontweight='bold', ha='center')
    
    # Sidebar
    rect = plt.Rectangle((0.3, 0.3), 3, 8.3, facecolor='#F8F9FA', edgecolor='#E0E0E0')
    ax.add_patch(rect)
    
    ax.text(1.8, 8.2, 'Settings', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.8, 7.6, '📁 Upload Audio', fontsize=10, ha='center')
    
    # Upload button mockup
    rect = FancyBboxPatch((0.6, 6.8), 2.4, 0.6, boxstyle="round,pad=0.02",
                           facecolor='white', edgecolor='#CCCCCC')
    ax.add_patch(rect)
    ax.text(1.8, 7.1, 'Choose File', fontsize=9, ha='center', color='gray')
    
    ax.text(1.8, 6.4, '🎯 Detection Mode', fontsize=10, ha='center')
    
    # Radio buttons
    ax.text(0.8, 5.9, '◉ All Species', fontsize=9)
    ax.text(0.8, 5.5, '○ Target Search', fontsize=9)
    
    ax.text(1.8, 4.8, '⚙️ Confidence: 80%', fontsize=10, ha='center')
    
    # Main content area
    ax.text(7, 8.2, 'Analysis Results', fontsize=14, fontweight='bold', ha='center')
    
    # Results table mockup
    rect = FancyBboxPatch((3.5, 5.5), 10, 2.4, boxstyle="round,pad=0.02",
                           facecolor='white', edgecolor='#E0E0E0', linewidth=1)
    ax.add_patch(rect)
    
    # Table header
    ax.text(5, 7.6, 'Species', fontsize=10, fontweight='bold')
    ax.text(9, 7.6, 'Confidence', fontsize=10, fontweight='bold')
    ax.text(12, 7.6, 'Status', fontsize=10, fontweight='bold')
    
    # Table rows
    rows = [
        ('American Robin', '94.2%', '✅ PRESENT', COLORS['success']),
        ('Blue Jay', '87.5%', '✅ PRESENT', COLORS['success']),
        ('American Crow', '45.3%', '❌ ABSENT', COLORS['danger']),
    ]
    
    for i, (species, conf, status, color) in enumerate(rows):
        y_row = 7.2 - i * 0.5
        ax.text(5, y_row, species, fontsize=9)
        ax.text(9, y_row, conf, fontsize=9)
        ax.text(12, y_row, status, fontsize=9, color=color, fontweight='bold')
    
    # Timeline section
    ax.text(7, 5.2, 'Detection Timeline', fontsize=12, fontweight='bold', ha='center')
    
    # Simple timeline mockup
    rect = FancyBboxPatch((3.5, 3.5), 10, 1.5, boxstyle="round,pad=0.02",
                           facecolor='white', edgecolor='#E0E0E0', linewidth=1)
    ax.add_patch(rect)
    
    # Timeline bars
    ax.text(4, 4.6, 'Robin:', fontsize=9)
    rect = plt.Rectangle((5.5, 4.45), 2, 0.3, facecolor=COLORS['success'], alpha=0.7)
    ax.add_patch(rect)
    rect = plt.Rectangle((9, 4.45), 1.5, 0.3, facecolor=COLORS['success'], alpha=0.7)
    ax.add_patch(rect)
    
    ax.text(4, 4.1, 'Blue Jay:', fontsize=9)
    rect = plt.Rectangle((7, 3.95), 2.5, 0.3, facecolor=COLORS['primary'], alpha=0.7)
    ax.add_patch(rect)
    
    # Time axis
    ax.text(5.5, 3.6, '0s', fontsize=8)
    ax.text(8.5, 3.6, '30s', fontsize=8)
    ax.text(12.5, 3.6, '60s', fontsize=8)
    
    # Grad-CAM section
    ax.text(7, 3.2, 'Explainability (Grad-CAM)', fontsize=12, fontweight='bold', ha='center')
    
    rect = FancyBboxPatch((3.5, 1), 10, 2, boxstyle="round,pad=0.02",
                           facecolor='white', edgecolor='#E0E0E0', linewidth=1)
    ax.add_patch(rect)
    
    # Fake spectrogram placeholder
    ax.text(8.5, 2, '[Spectrogram with Grad-CAM Overlay]', fontsize=10, 
            ha='center', va='center', color='gray', style='italic')
    
    # Download button
    rect = FancyBboxPatch((11, 0.4), 2, 0.5, boxstyle="round,pad=0.02",
                           facecolor=COLORS['primary'], edgecolor='none')
    ax.add_patch(rect)
    ax.text(12, 0.65, '📥 Download PDF', fontsize=9, ha='center', color='white')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "16_app_mockup.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 17: OVERFITTING EXPLANATION
# ============================================================

def create_overfitting_explanation():
    """Create visualization explaining overfitting problem"""
    print("Creating: Overfitting Explanation...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Training vs Validation curves showing gap
    ax = axes[0]
    epochs = list(range(1, 18))
    
    train_acc = TRAINING_DATA['train_acc']
    val_acc = TRAINING_DATA['val_acc']
    
    ax.fill_between(epochs, train_acc, val_acc, alpha=0.3, color='red', label='Overfitting Gap')
    ax.plot(epochs, train_acc, 'b-', linewidth=2, marker='o', markersize=4, label='Training Accuracy')
    ax.plot(epochs, val_acc, 'r-', linewidth=2, marker='s', markersize=4, label='Validation Accuracy')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('The Overfitting Problem', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='center right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Annotate the gap
    ax.annotate('', xy=(17, 99.83), xytext=(17, 66.5),
                arrowprops=dict(arrowstyle='<->', color='red', lw=3))
    ax.text(17.5, 83, '31% Gap!\nModel memorized\ntraining data', fontsize=10, 
            color='red', fontweight='bold')
    
    # Plot 2: Explanation diagram
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('What is Overfitting?', fontsize=14, fontweight='bold')
    
    # Good fit example
    ax.text(2.5, 9, 'Good Fit', fontsize=12, fontweight='bold', ha='center', color='green')
    rect = FancyBboxPatch((0.5, 6.5), 4, 2.2, boxstyle="round,pad=0.05",
                           facecolor='#E8F5E9', edgecolor='green', linewidth=2)
    ax.add_patch(rect)
    ax.text(2.5, 8, 'Model learns PATTERNS', fontsize=10, ha='center')
    ax.text(2.5, 7.4, '"Birds with X frequency', fontsize=9, ha='center', style='italic')
    ax.text(2.5, 7, 'pattern are Robins"', fontsize=9, ha='center', style='italic')
    
    # Overfit example
    ax.text(7.5, 9, 'Overfitting', fontsize=12, fontweight='bold', ha='center', color='red')
    rect = FancyBboxPatch((5.5, 6.5), 4, 2.2, boxstyle="round,pad=0.05",
                           facecolor='#FFEBEE', edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    ax.text(7.5, 8, 'Model MEMORIZES data', fontsize=10, ha='center')
    ax.text(7.5, 7.4, '"Recording #1234 is', fontsize=9, ha='center', style='italic')
    ax.text(7.5, 7, 'always Robin"', fontsize=9, ha='center', style='italic')
    
    # Results
    ax.text(2.5, 5.8, '✓ Works on new data', fontsize=10, ha='center', color='green')
    ax.text(7.5, 5.8, '✗ Fails on new data', fontsize=10, ha='center', color='red')
    
    # Why it happened
    ax.text(5, 4.5, 'Why Did We Overfit?', fontsize=12, fontweight='bold', ha='center')
    
    reasons = [
        '1. Limited data (23,839 chunks for 54 species)',
        '2. No data augmentation (no noise, pitch shift)',
        '3. Some species have very few samples',
        '4. Model is powerful (4M parameters)',
    ]
    
    for i, reason in enumerate(reasons):
        ax.text(5, 3.8 - i*0.5, reason, fontsize=10, ha='center')
    
    # How to fix
    ax.text(5, 1.5, 'How to Fix (Future Work):', fontsize=11, fontweight='bold', ha='center')
    fixes = ['• Add data augmentation', '• Collect more data', '• Stronger regularization']
    ax.text(5, 0.8, '  |  '.join(fixes), fontsize=9, ha='center', color='green')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "17_overfitting_explanation.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 18: SPECIES EXAMPLES
# ============================================================

def create_species_examples():
    """Create grid showing example species"""
    print("Creating: Species Examples Grid...")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Sample species with simulated spectrograms
    species_list = [
        ('American Robin', 'Turdus migratorius', 83.9, 'Melodic phrases'),
        ('Blue Jay', 'Cyanocitta cristata', 86.1, 'Harsh calls'),
        ('Barred Owl', 'Strix varia', 89.5, '"Who cooks for you"'),
        ('American Goldfinch', 'Spinus tristis', 94.2, '"Po-ta-to-chip"'),
        ('American Crow', 'Corvus brachyrhynchos', 85.4, 'Cawing'),
        ('Black-capped Chickadee', 'Poecile atricapillus', 78.3, '"Chick-a-dee"'),
        ('Brown Thrasher', 'Toxostoma rufum', 82.6, 'Mimicry'),
        ('Belted Kingfisher', 'Megaceryle alcyon', 81.2, 'Rattling call'),
        ('Bobolink', 'Dolichonyx oryzivorus', 84.7, 'Bubbly song'),
        ('Barn Swallow', 'Hirundo rustica', 75.2, 'Twittering'),
        ('American Woodcock', 'Scolopax minor', 87.3, 'Peent call'),
        ('Blue-winged Warbler', 'Vermivora cyanoptera', 79.8, 'Bee-buzz'),
    ]
    
    np.random.seed(42)
    
    for idx, (ax, (name, sci_name, acc, call_type)) in enumerate(zip(axes.flatten(), species_list)):
        # Generate unique spectrogram pattern for each species
        n_mels, n_frames = 64, 100
        spec = np.random.randn(n_mels, n_frames) * 0.3 - 4
        
        # Add species-specific pattern
        np.random.seed(idx * 10)
        n_calls = np.random.randint(2, 5)
        for _ in range(n_calls):
            start = np.random.randint(10, n_frames - 20)
            end = start + np.random.randint(10, 25)
            freq_center = np.random.randint(20, 50)
            freq_width = np.random.randint(5, 15)
            
            for frame in range(start, min(end, n_frames)):
                for mel in range(max(0, freq_center - freq_width), 
                                min(n_mels, freq_center + freq_width)):
                    spec[mel, frame] += np.random.rand() * 3
        
        ax.imshow(spec, aspect='auto', origin='lower', cmap='magma')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Title with species info
        ax.set_title(f'{name}\n({sci_name})', fontsize=10, fontweight='bold')
        
        # Accuracy badge
        color = COLORS['success'] if acc >= 80 else COLORS['warning']
        ax.text(0.95, 0.95, f'{acc}%', transform=ax.transAxes, fontsize=9,
                ha='right', va='top', fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        
        # Call type
        ax.text(0.5, -0.1, f'Call: {call_type}', transform=ax.transAxes, 
                fontsize=8, ha='center', style='italic')
    
    plt.suptitle('Sample Bird Species and Their Spectrograms', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "18_species_examples.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 19: CHUNK OVERLAP EXPLANATION
# ============================================================

def create_chunk_overlap():
    """Create visualization explaining chunk overlap"""
    print("Creating: Chunk Overlap Explanation...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Without overlap (problem)
    ax = axes[0]
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('Problem: Without Overlap - Bird Call Gets Split!', 
                 fontsize=14, fontweight='bold', color='red')
    
    # Audio waveform representation
    x = np.linspace(0, 20, 1000)
    y = np.sin(x * 2) * 0.3 + 2
    ax.plot(x, y, 'b-', linewidth=1, alpha=0.5)
    
    # Bird call region
    ax.axvspan(4.5, 5.5, alpha=0.3, color='green')
    ax.text(5, 3.5, '🐦 Bird Call', fontsize=10, ha='center', color='green', fontweight='bold')
    
    # Chunk boundaries (no overlap)
    chunks_no_overlap = [(0, 5), (5, 10), (10, 15), (15, 20)]
    colors = ['#FFCDD2', '#BBDEFB', '#C8E6C9', '#FFF9C4']
    
    for i, ((start, end), color) in enumerate(zip(chunks_no_overlap, colors)):
        rect = plt.Rectangle((start, 0.5), end-start, 1, facecolor=color, 
                              edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text((start+end)/2, 1, f'Chunk {i+1}', fontsize=10, ha='center', fontweight='bold')
    
    # Show split call
    ax.annotate('Call split\nbetween chunks!', xy=(5, 1.5), xytext=(7, 3),
                fontsize=10, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Plot 2: With overlap (solution)
    ax = axes[1]
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('Solution: With 50% Overlap - Complete Call in One Chunk!', 
                 fontsize=14, fontweight='bold', color='green')
    
    # Audio waveform
    ax.plot(x, y, 'b-', linewidth=1, alpha=0.5)
    
    # Bird call region
    ax.axvspan(4.5, 5.5, alpha=0.3, color='green')
    ax.text(5, 3.5, '🐦 Bird Call', fontsize=10, ha='center', color='green', fontweight='bold')
    
    # Chunk boundaries (with 50% overlap)
    chunks_overlap = [(0, 5), (2.5, 7.5), (5, 10), (7.5, 12.5), (10, 15)]
    
    y_offsets = [0.5, 1.2, 0.5, 1.2, 0.5]  # Stagger for visibility
    
    for i, ((start, end), y_off) in enumerate(zip(chunks_overlap, y_offsets)):
        alpha = 0.5 if i % 2 == 0 else 0.7
        rect = plt.Rectangle((start, y_off), end-start, 0.6, 
                              facecolor=colors[i % 4], edgecolor='black', 
                              linewidth=2, alpha=alpha)
        ax.add_patch(rect)
        ax.text((start+end)/2, y_off + 0.3, f'Chunk {i+1}', fontsize=9, ha='center')
    
    # Highlight chunk 2 which has complete call
    ax.annotate('Chunk 2 has\ncomplete call!', xy=(5, 1.5), xytext=(8, 3),
                fontsize=10, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Add legend
    ax.text(17, 3, '50% Overlap means:\n• Chunks share 2.5 seconds\n• Every call is complete\n  in at least one chunk', 
            fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "19_chunk_overlap.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# IMAGE 20: UNCERTAINTY WARNINGS
# ============================================================

def create_uncertainty_warnings():
    """Create visualization of uncertainty warning system"""
    print("Creating: Uncertainty Warnings...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.set_title('Uncertainty Warning System', fontsize=18, fontweight='bold')
    ax.text(7, 9.3, 'Alerts users when predictions may be unreliable', 
            fontsize=12, ha='center', style='italic', color='gray')
    
    # Warning types
    warnings = [
        ('⚠️ Limited Training Data', 
         'Species had few training samples', 
         'Example: Bufflehead (only 98 samples)', 
         '#FFF3E0', -15),
        ('⚠️ Model Uncertain', 
         'Top 2 predictions are very close', 
         'Example: Robin 47%, Thrush 42%', 
         '#E3F2FD', -15),
        ('⚠️ Unusual Detection', 
         'Rare/large bird with high confidence', 
         'Example: Bald Eagle at 95% in city', 
         '#FCE4EC', -25),
        ('⚠️ Sparse Detection', 
         'High confidence but very few chunks', 
         'Example: 1 chunk at 92%, rest below 20%', 
         '#E8F5E9', -20),
        ('⚠️ Low Confidence', 
         'No species above 50% confidence', 
         'May be noise or unknown species', 
         '#FFEBEE', -30),
    ]
    
    y = 8.5
    for title, desc, example, color, impact in warnings:
        # Warning box
        rect = FancyBboxPatch((0.5, y - 0.8), 9.5, 1.4, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        ax.text(0.8, y + 0.2, title, fontsize=12, fontweight='bold', va='center')
        ax.text(0.8, y - 0.2, desc, fontsize=10, va='center')
        ax.text(0.8, y - 0.5, example, fontsize=9, va='center', color='gray', style='italic')
        
        # Impact score
        color_impact = COLORS['danger'] if impact <= -20 else COLORS['warning']
        rect = FancyBboxPatch((10.5, y - 0.4), 3, 0.8, boxstyle="round,pad=0.02",
                               facecolor=color_impact, edgecolor='black')
        ax.add_patch(rect)
        ax.text(12, y, f'{impact} points', fontsize=10, ha='center', va='center', 
                color='white', fontweight='bold')
        
        y -= 1.6
    
    # Reliability score explanation
    ax.text(7, 1.2, 'Reliability Score Calculation', fontsize=14, fontweight='bold', ha='center')
    
    # Score boxes
    scores = [
        ('80-100', 'HIGH', COLORS['success'], 'Results are reliable'),
        ('60-79', 'MEDIUM', COLORS['warning'], 'Some caution advised'),
        ('0-59', 'LOW', COLORS['danger'], 'Results may be wrong'),
    ]
    
    x = 1.5
    for score_range, label, color, meaning in scores:
        rect = FancyBboxPatch((x, 0.3), 3.5, 0.7, boxstyle="round,pad=0.02",
                               facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + 1.75, 0.65, f'{score_range}: {label}', fontsize=10, 
                ha='center', va='center', color='white', fontweight='bold')
        x += 4
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, "20_uncertainty_warnings.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Generate all report images"""
    print("=" * 60)
    print("BIRD DETECTION PROJECT - REPORT IMAGE GENERATOR")
    print("=" * 60)
    print(f"\nOutput folder: {OUTPUT_FOLDER}\n")
    
    # Create all images
    image_functions = [
        create_system_architecture,
        create_spectrogram_example,
        create_training_curves,
        create_model_accuracy_comparison,
        create_model_efficiency_comparison,
        create_confusion_matrix,
        create_species_accuracy_chart,
        create_gradcam_visualization,
        create_data_distribution,
        create_architecture_diagrams,
        create_preprocessing_pipeline,
        create_timeline_detection,
        create_confidence_threshold,
        create_transfer_learning,
        create_summary_infographic,
        create_app_mockup,
        create_overfitting_explanation,
        create_species_examples,
        create_chunk_overlap,
        create_uncertainty_warnings,
    ]
    
    total = len(image_functions)
    success = 0
    failed = []
    
    for i, func in enumerate(image_functions, 1):
        try:
            func()
            success += 1
        except Exception as e:
            failed.append((func.__name__, str(e)))
            print(f"  ERROR in {func.__name__}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\n✅ Successfully created: {success}/{total} images")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)} images")
        for name, error in failed:
            print(f"   - {name}: {error}")
    
    print(f"\n📁 Images saved to: {OUTPUT_FOLDER}")
    
    # List all created files
    print("\n📋 Created files:")
    try:
        files = sorted(os.listdir(OUTPUT_FOLDER))
        for f in files:
            if f.endswith('.png'):
                size = os.path.getsize(os.path.join(OUTPUT_FOLDER, f)) / 1024
                print(f"   - {f} ({size:.1f} KB)")
    except Exception as e:
        print(f"   Error listing files: {e}")
    
    print("\n✨ Done! Use these images in your Canva report.")


if __name__ == "__main__":
    main()
    
        