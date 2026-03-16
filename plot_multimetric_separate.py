"""
Separate Multi-Metric Comparison Plot for CNN Algorithms
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Compile metrics for all 5 models
models_data = {
    'Model V3\n(Best)': {
        'top1': 96.06,
        'top5': 98.74,
        'f1': 94.00
    },
    'Eval Batch 1': {
        'top1': 72.26,
        'top5': 85.55,
        'f1': 63.20
    },
    'Fast Quality': {
        'top1': 71.24,
        'top5': 85.00,
        'f1': 60.00
    },
    'Eval Batch 2': {
        'top1': 67.42,
        'top5': 82.57,
        'f1': 66.40
    },
    'Balanced\nTraining': {
        'top1': 63.82,
        'top5': 75.00,
        'f1': 52.00
    }
}

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Prepare data
models = list(models_data.keys())
top1_accs = [models_data[m]['top1'] for m in models]
top5_accs = [models_data[m]['top5'] for m in models]
f1_scores = [models_data[m]['f1'] for m in models]

# Set up bar positions
x = np.arange(len(models))
width = 0.25

# Create bars
bars1 = ax.bar(x - width, top1_accs, width, label='Top-1 Accuracy', 
               alpha=0.85, color='#3498db', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x, top5_accs, width, label='Top-5 Accuracy', 
               alpha=0.85, color='#2ecc71', edgecolor='black', linewidth=1.5)
bars3 = ax.bar(x + width, f1_scores, width, label='F1 Score', 
               alpha=0.85, color='#e74c3c', edgecolor='black', linewidth=1.5)

# Customize plot
ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('CNN Models', fontsize=13, fontweight='bold')
ax.set_title('Multi-Metric Performance Comparison - All 5 CNN Algorithms', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=80, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5, linewidth=1)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

# Add legend
ax.legend(loc='lower left', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True)

# Add background shading
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

plt.tight_layout()

# Save
output_path = Path('10_Outputs/Reports/cnn_multimetric_comparison.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Multi-metric comparison plot saved to {output_path}")

# Print summary
print("\n" + "="*80)
print("MULTI-METRIC COMPARISON SUMMARY")
print("="*80)
print(f"{'Model':<25} {'Top-1 Acc':<15} {'Top-5 Acc':<15} {'F1 Score':<15}")
print("-"*80)
for model in models:
    print(f"{model:<25} {models_data[model]['top1']:<15.2f}% {models_data[model]['top5']:<15.2f}% {models_data[model]['f1']:<15.2f}%")
print("="*80)

# Statistics
print(f"\nTop-1 Accuracy Statistics:")
print(f"  Mean:   {np.mean(top1_accs):.2f}%")
print(f"  Median: {np.median(top1_accs):.2f}%")
print(f"  Std:    {np.std(top1_accs):.2f}%")
print(f"  Max:    {np.max(top1_accs):.2f}% (Model V3)")
print(f"  Min:    {np.min(top1_accs):.2f}% (Balanced Training)")

print(f"\nTop-5 Accuracy Statistics:")
print(f"  Mean:   {np.mean(top5_accs):.2f}%")
print(f"  Median: {np.median(top5_accs):.2f}%")
print(f"  Std:    {np.std(top5_accs):.2f}%")

print(f"\nF1 Score Statistics:")
print(f"  Mean:   {np.mean(f1_scores):.2f}%")
print(f"  Median: {np.median(f1_scores):.2f}%")
print(f"  Std:    {np.std(f1_scores):.2f}%")

print(f"\nPerformance Rankings (Top-1 Accuracy):")
rankings = sorted(zip(models, top1_accs), key=lambda x: x[1], reverse=True)
for idx, (model, acc) in enumerate(rankings, 1):
    medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][idx-1]
    print(f"  {medal} {idx}. {model:<25} {acc:.2f}%")

print("="*80)
