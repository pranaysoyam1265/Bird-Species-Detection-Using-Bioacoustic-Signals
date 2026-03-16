"""
Updated CNN Comparison Plot - without Multi-Metric (now separate)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Compile metrics for all 5 models
models_data = {
    'Model V3 (Best)': {
        'test_accuracy': 96.06,
        'validation_accuracy': 95.90,
        'top5_accuracy': 98.74,
        'f1_score': 0.94
    },
    'Evaluation Set (Batch 1)': {
        'test_accuracy': 72.26,
        'validation_accuracy': 72.26,
        'top5_accuracy': 85.55,
        'f1_score': 0.632
    },
    'Evaluation Set (Batch 2)': {
        'test_accuracy': 67.42,
        'validation_accuracy': 67.42,
        'top5_accuracy': 82.57,
        'f1_score': 0.664
    },
    'Balanced Training': {
        'test_accuracy': 63.82,
        'validation_accuracy': 63.82,
        'top5_accuracy': 75.0,
        'f1_score': 0.52
    },
    'Fast Quality': {
        'test_accuracy': 71.24,
        'validation_accuracy': 71.24,
        'top5_accuracy': 85.0,
        'f1_score': 0.60
    }
}

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# 1. Top-1 Accuracy Comparison
ax1 = fig.add_subplot(gs[0, 0])
models = list(models_data.keys())
top1_accs = [models_data[m]['test_accuracy'] for m in models]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
bars1 = ax1.bar(range(len(models)), top1_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Top-1 Accuracy Comparison', fontsize=12, fontweight='bold')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 2. Top-5 Accuracy Comparison
ax2 = fig.add_subplot(gs[0, 1])
top5_accs = [models_data[m]['top5_accuracy'] for m in models]
bars2 = ax2.bar(range(len(models)), top5_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('Top-5 Accuracy Comparison', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax2.set_ylim(0, 105)
ax2.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 3. F1 Score Comparison
ax3 = fig.add_subplot(gs[1, 0])
f1_scores = [models_data[m]['f1_score'] for m in models]
bars3 = ax3.bar(range(len(models)), f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
ax3.set_title('F1 Score Comparison', fontsize=12, fontweight='bold')
ax3.set_xticks(range(len(models)))
ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax3.set_ylim(0, 1.0)
ax3.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 4. Accuracy Metrics Heatmap
ax4 = fig.add_subplot(gs[1, 1])
normalized_top1 = top1_accs
normalized_top5 = top5_accs
normalized_f1 = [f * 100 for f in f1_scores]

heatmap_data = np.array([
    normalized_top1,
    normalized_top5,
    normalized_f1
])
im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
ax4.set_yticks([0, 1, 2])
ax4.set_yticklabels(['Top-1 Accuracy', 'Top-5 Accuracy', 'F1 Score'], fontsize=10, fontweight='bold')
ax4.set_xticks(range(len(models)))
ax4.set_xticklabels(models, rotation=45, ha='right', fontsize=9)

# Add text annotations
for i in range(3):
    for j in range(len(models)):
        text = ax4.text(j, i, f'{heatmap_data[i, j]:.1f}',
                       ha="center", va="center", color="black", fontweight='bold', fontsize=9)

# Add colorbar
cbar = plt.colorbar(im, ax=ax4)
cbar.set_label('Score (%)', fontsize=10, fontweight='bold')

plt.suptitle('CNN Algorithms Core Comparison (See Multi-Metric Comparison for Grouped Bars)', 
             fontsize=13, fontweight='bold', y=0.98)

# Save
output_path = Path('10_Outputs/Reports/cnn_comparison_plot_core.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Core comparison plot saved to {output_path}")
plt.close()

print("\nFiles generated:")
print("  1. cnn_comparison_plot_core.png - Top-1, Top-5, F1 Score, Heatmap")
print("  2. cnn_multimetric_comparison.png - Grouped bar chart (all 3 metrics)")
