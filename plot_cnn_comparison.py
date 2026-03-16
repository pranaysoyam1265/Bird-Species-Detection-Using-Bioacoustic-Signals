"""
Create comparison plots for the 5 CNN algorithms
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
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
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

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

# 4. Multi-metric Comparison (grouped bar chart)
ax4 = fig.add_subplot(gs[1, 1])
metrics = ['Top-1 Acc', 'Top-5 Acc', 'F1 Score']
x = np.arange(len(models))
width = 0.25

# Normalize metrics to 0-100 scale for comparison
normalized_top1 = top1_accs
normalized_top5 = top5_accs
normalized_f1 = [f * 100 for f in f1_scores]

bars_a = ax4.bar(x - width, normalized_top1, width, label='Top-1 Acc', alpha=0.8, color='#3498db', edgecolor='black')
bars_b = ax4.bar(x, normalized_top5, width, label='Top-5 Acc', alpha=0.8, color='#2ecc71', edgecolor='black')
bars_c = ax4.bar(x + width, normalized_f1, width, label='F1 Score', alpha=0.8, color='#e74c3c', edgecolor='black')

ax4.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
ax4.set_title('Multi-Metric Comparison', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax4.legend(loc='lower right', fontsize=9)
ax4.set_ylim(0, 105)
ax4.grid(axis='y', alpha=0.3)

# 5. Accuracy Metrics Heatmap
ax5 = fig.add_subplot(gs[2, :])
heatmap_data = np.array([
    top1_accs,
    top5_accs,
    normalized_f1
])
im = ax5.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
ax5.set_yticks([0, 1, 2])
ax5.set_yticklabels(['Top-1 Accuracy', 'Top-5 Accuracy', 'F1 Score'], fontsize=10, fontweight='bold')
ax5.set_xticks(range(len(models)))
ax5.set_xticklabels(models, rotation=45, ha='right', fontsize=9)

# Add text annotations
for i in range(3):
    for j in range(len(models)):
        text = ax5.text(j, i, f'{heatmap_data[i, j]:.1f}',
                       ha="center", va="center", color="black", fontweight='bold', fontsize=10)

# Add colorbar
cbar = plt.colorbar(im, ax=ax5, orientation='horizontal', pad=0.15)
cbar.set_label('Score (%)', fontsize=10, fontweight='bold')

plt.suptitle('CNN Algorithms Performance Comparison', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

# Save and show
output_path = Path('10_Outputs/Reports/cnn_comparison_plot.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")
plt.show()

# Create a detailed comparison table
print("\n" + "="*100)
print("CNN ALGORITHMS PERFORMANCE COMPARISON")
print("="*100)
print(f"{'Model':<30} {'Top-1 Acc':<15} {'Top-5 Acc':<15} {'F1 Score':<15}")
print("-"*100)
for model in models:
    print(f"{model:<30} {models_data[model]['test_accuracy']:<15.2f}% {models_data[model]['top5_accuracy']:<15.2f}% {models_data[model]['f1_score']:<15.4f}")
print("="*100)
