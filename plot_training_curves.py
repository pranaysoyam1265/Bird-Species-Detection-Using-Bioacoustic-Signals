"""
Create advanced comparison plots showing training curves and detailed metrics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path

# Set style
sns.set_style("whitegrid")

# Load training histories from JSON files
def load_training_history(filepath):
    """Load training history from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None

# Load data
history_balanced = load_training_history(r'05_Model\Training_Logs\balanced_20260216_202533\history.json')
history_fast = load_training_history(r'05_Model\Training_Logs\fast_quality_20260214_225930\training_history.json')
history_run = load_training_history(r'05_Model\Training_Logs\run_20260210_212446\training_history.json')
results_v3 = load_training_history(r'05_Model\Training_Logs\test_results_v3_FINAL.json')

# Create figure for training curves
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Plot 1: Balanced Model Training Curves
if history_balanced:
    ax1 = fig.add_subplot(gs[0, 0])
    epochs_balanced = range(1, len(history_balanced['train_loss']) + 1)
    ax1.plot(epochs_balanced, history_balanced['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=4, color='#3498db')
    ax1.plot(epochs_balanced, history_balanced['val_loss'], 's-', label='Val Loss', linewidth=2, markersize=4, color='#e74c3c')
    ax1.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=10, fontweight='bold')
    ax1.set_title('Balanced Training Model - Loss Curves', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

# Plot 2: Balanced Model Accuracy
if history_balanced and 'train_acc' in history_balanced:
    ax2 = fig.add_subplot(gs[0, 1])
    epochs_balanced = range(1, len(history_balanced['train_acc']) + 1)
    ax2.plot(epochs_balanced, history_balanced['train_acc'], 'o-', label='Train Accuracy', linewidth=2, markersize=4, color='#2ecc71')
    ax2.plot(epochs_balanced, history_balanced['val_acc'], 's-', label='Val Accuracy', linewidth=2, markersize=4, color='#f39c12')
    ax2.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
    ax2.set_title('Balanced Training Model - Accuracy Curves', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

# Plot 3: Fast Quality Model Training Curves
if history_fast:
    ax3 = fig.add_subplot(gs[1, 0])
    epochs_fast = range(1, len(history_fast['train_loss']) + 1)
    ax3.plot(epochs_fast, history_fast['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=4, color='#3498db')
    ax3.plot(epochs_fast, history_fast['val_loss'], 's-', label='Val Loss', linewidth=2, markersize=4, color='#e74c3c')
    ax3.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Loss', fontsize=10, fontweight='bold')
    ax3.set_title('Fast Quality Model - Loss Curves', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

# Plot 4: Fast Quality Model Accuracy
if history_fast and 'train_acc' in history_fast:
    ax4 = fig.add_subplot(gs[1, 1])
    epochs_fast = range(1, len(history_fast['train_acc']) + 1)
    ax4.plot(epochs_fast, history_fast['train_acc'], 'o-', label='Train Accuracy', linewidth=2, markersize=4, color='#2ecc71')
    ax4.plot(epochs_fast, history_fast['val_acc'], 's-', label='Val Accuracy', linewidth=2, markersize=4, color='#f39c12')
    ax4.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
    ax4.set_title('Fast Quality Model - Accuracy Curves', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

plt.suptitle('CNN Models - Training Curves Analysis', fontsize=13, fontweight='bold')
output_path = Path('10_Outputs/Reports/cnn_training_curves.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Training curves plot saved to {output_path}")

# Create convergence analysis figure
fig2 = plt.figure(figsize=(14, 8))
gs2 = fig2.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Convergence speed analysis
ax5 = fig2.add_subplot(gs2[0, :])
convergence_data = {
    'Balanced Model': {
        'epochs': 48,
        'final_val_acc': 63.82,
        'best_epoch': 35  # estimate
    },
    'Fast Quality': {
        'epochs': 24,
        'final_val_acc': 71.24,
        'best_epoch': 18  # estimate
    },
    'Run 20260210': {
        'epochs': 11,
        'final_val_acc': 100.0,
        'best_epoch': 11
    }
}

models_conv = list(convergence_data.keys())
epochs_list = [convergence_data[m]['epochs'] for m in models_conv]
final_accs = [convergence_data[m]['final_val_acc'] for m in models_conv]

x_pos = np.arange(len(models_conv))
width = 0.35

bars1 = ax5.bar(x_pos - width/2, epochs_list, width, label='Total Epochs', alpha=0.8, color='#3498db', edgecolor='black')
ax5_right = ax5.twinx()
bars2 = ax5_right.bar(x_pos + width/2, final_accs, width, label='Final Val Accuracy', alpha=0.8, color='#2ecc71', edgecolor='black')

ax5.set_xlabel('Model', fontsize=11, fontweight='bold')
ax5.set_ylabel('Number of Epochs', fontsize=11, fontweight='bold', color='#3498db')
ax5_right.set_ylabel('Final Validation Accuracy (%)', fontsize=11, fontweight='bold', color='#2ecc71')
ax5.set_title('Training Convergence Analysis', fontsize=12, fontweight='bold')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(models_conv)
ax5.tick_params(axis='y', labelcolor='#3498db')
ax5_right.tick_params(axis='y', labelcolor='#2ecc71')
ax5.legend(loc='upper left', fontsize=9)
ax5_right.legend(loc='upper right', fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# Distribution of metrics
ax6 = fig2.add_subplot(gs2[1, 0])
all_accuracies = [96.06, 72.26, 71.24, 67.42, 63.82]
ax6.hist(all_accuracies, bins=5, color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=1.5)
ax6.axvline(np.mean(all_accuracies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_accuracies):.1f}%')
ax6.axvline(np.median(all_accuracies), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_accuracies):.1f}%')
ax6.set_xlabel('Top-1 Accuracy (%)', fontsize=10, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax6.set_title('Accuracy Distribution Across Models', fontsize=11, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

# Performance metrics radar
ax7 = fig2.add_subplot(gs2[1, 1])
# Normalize metrics for radar chart
metrics_normalized = {
    'Model V3': [96.06/100, 98.74/100, 0.94, 0.9],  # High performance
    'Eval Batch 1': [72.26/100, 85.55/100, 0.632, 0.7],
    'Fast Quality': [71.24/100, 85.0/100, 0.6, 0.75],
}

# Create a simple 2D comparison
models_list = list(metrics_normalized.keys())
scores = [np.mean([metrics_normalized[m][0], metrics_normalized[m][1], metrics_normalized[m][2]]) * 100 
          for m in models_list]

colors_chart = ['#2ecc71', '#3498db', '#9b59b6']
wedges, texts, autotexts = ax7.pie(scores, labels=models_list, autopct='%1.1f%%', 
                                     colors=colors_chart, startangle=90, 
                                     explode=(0.1, 0.05, 0.05))
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)
ax7.set_title('Average Performance Score Distribution', fontsize=11, fontweight='bold')

plt.suptitle('CNN Models - Convergence & Analysis', fontsize=13, fontweight='bold')
output_path2 = Path('10_Outputs/Reports/cnn_convergence_analysis.png')
output_path2.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"Convergence analysis plot saved to {output_path2}")

print("\n" + "="*80)
print("TRAINING METRICS SUMMARY")
print("="*80)
print(f"\nBalanced Model:")
print(f"  - Total Epochs: 48")
print(f"  - Final Validation Accuracy: 63.82%")
print(f"  - Training Time: ~{48 * 30:.0f} seconds (estimated)")

print(f"\nFast Quality Model:")
print(f"  - Total Epochs: 24")
print(f"  - Final Validation Accuracy: 71.24%")
print(f"  - Training Time: ~{24 * 30:.0f} seconds (estimated)")

print(f"\nRun 20260210 Model:")
print(f"  - Total Epochs: 11")
print(f"  - Final Validation Accuracy: 100.0%")
print(f"  - Training Time: ~{11 * 30:.0f} seconds (estimated)")

print("\n" + "="*80)
