import matplotlib.pyplot as plt
import os

def create_comparison_plot():
    # Model data derived from frontend configuration (models.ts)
    model_names = ['MobileNetV3', 'VGG-16', 'ResNet-50', 'EfficientNet-B0', 'EfficientNet-B2']
    accuracies = [88.7, 89.3, 91.2, 94.2, 95.03]
    inference_times = [12, 78, 45, 18, 24]
    params = [5.4, 138, 25.6, 5.3, 9.2]
    
    # Colors: VGG (red/bad), MobileNet (purple/edge), ResNet (gray/baseline), Eff-B0 (blue/good), Eff-B2 (green/winner)
    colors = ['#9b59b6', '#e74c3c', '#95a5a6', '#3498db', '#2ecc71']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scale bubble sizes (area proportional to parameters)
    sizes = [p * 20 for p in params]
    
    scatter = ax.scatter(inference_times, accuracies, s=sizes, c=colors, alpha=0.7, edgecolors='white', linewidths=2)
    
    # Add annotations for each model
    for i, txt in enumerate(model_names):
        # Adjust text offset for VGG so it's not buried
        y_offset = 20 if txt != 'VGG-16' else 60
        ax.annotate(txt, (inference_times[i], accuracies[i]), xytext=(0, y_offset), textcoords='offset points', ha='center', fontweight='bold', fontsize=11)
        
        # Add parameter count below the model name
        ax.annotate(f"{params[i]}M params", (inference_times[i], accuracies[i]), xytext=(0, y_offset-12), textcoords='offset points', ha='center', fontsize=9, color='darkgray')
        
    ax.set_title("CNN Architecture Comparison for Bird Species Detection\nAccuracy vs. Inference Time Trade-offs", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Inference Time (ms) - Lower is faster →", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%) - Higher is better ↑", fontsize=12)
    
    # Set axis limits with some padding
    ax.set_xlim(5, 90)
    ax.set_ylim(87, 97)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Highlight the 'ideal' region
    ax.axvspan(5, 30, ymin=0.6, ymax=1, alpha=0.1, color='green', label='Ideal Zone (Fast & Accurate)')
    
    # Custom legend for the ideal zone
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    output_dir = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\10_Outputs\Visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, "model_comparison_bubble.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to: {output_path}")

if __name__ == "__main__":
    create_comparison_plot()
