"""
Script: analyze_species_performance.py
Purpose: Deep analysis of per-species performance
Location: 09_Utils/Scripts/analyze_species_performance.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"

TRAIN_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split_Fixed", "train.csv")
VAL_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split_Fixed", "val.csv")
TEST_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split_Fixed", "test.csv")
REPORT_CSV = os.path.join(BASE_DIR, "07_Evaluation", "Test_Results", "classification_report.csv")
PREDICTIONS_CSV = os.path.join(BASE_DIR, "07_Evaluation", "Test_Results", "test_predictions.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "07_Evaluation", "Species_Analysis")

# English names mapping
ENGLISH_NAMES = {
    "Amphispiza bilineata": "Black-throated Sparrow",
    "Anthus rubescens": "American Pipit",
    "Archilochus alexandri": "Black-chinned Hummingbird",
    "Artemisiospiza belli": "Bell's Sparrow",
    "Botaurus lentiginosus": "American Bittern",
    "Bucephala albeola": "Bufflehead",
    "Buteo platypterus": "Broad-winged Hawk",
    "Calidris bairdii": "Baird's Sandpiper",
    "Calypte anna": "Anna's Hummingbird",
    "Certhia americana": "Brown Creeper",
    "Chroicocephalus philadelphia": "Bonaparte's Gull",
    "Coccyzus erythropthalmus": "Black-billed Cuckoo",
    "Corvus brachyrhynchos": "American Crow",
    "Cyanocitta cristata": "Blue Jay",
    "Dolichonyx oryzivorus": "Bobolink",
    "Empidonax alnorum": "Alder Flycatcher",
    "Euphagus cyanocephalus": "Brewer's Blackbird",
    "Falco sparverius": "American Kestrel",
    "Haliaeetus leucocephalus": "Bald Eagle",
    "Hirundo rustica": "Barn Swallow",
    "Icterus bullockii": "Bullock's Oriole",
    "Icterus galbula": "Baltimore Oriole",
    "Mareca americana": "American Wigeon",
    "Megaceryle alcyon": "Belted Kingfisher",
    "Mniotilta varia": "Black-and-white Warbler",
    "Molothrus ater": "Brown-headed Cowbird",
    "Myiarchus cinerascens": "Ash-throated Flycatcher",
    "Passerina caerulea": "Blue Grosbeak",
    "Pheucticus melanocephalus": "Black-headed Grosbeak",
    "Pica hudsonia": "Black-billed Magpie",
    "Poecile atricapillus": "Black-capped Chickadee",
    "Polioptila caerulea": "Blue-gray Gnatcatcher",
    "Psaltriparus minimus": "Bushtit",
    "Recurvirostra americana": "American Avocet",
    "Riparia riparia": "Bank Swallow",
    "Sayornis nigricans": "Black Phoebe",
    "Scolopax minor": "American Woodcock",
    "Selasphorus platycercus": "Broad-tailed Hummingbird",
    "Setophaga caerulescens": "Black-throated Blue Warbler",
    "Setophaga fusca": "Blackburnian Warbler",
    "Setophaga nigrescens": "Black-throated Gray Warbler",
    "Setophaga ruticilla": "American Redstart",
    "Setophaga striata": "Blackpoll Warbler",
    "Setophaga virens": "Black-throated Green Warbler",
    "Spatula discors": "Blue-winged Teal",
    "Spinus tristis": "American Goldfinch",
    "Spizella breweri": "Brewer's Sparrow",
    "Spizelloides arborea": "American Tree Sparrow",
    "Strix varia": "Barred Owl",
    "Thryomanes bewickii": "Bewick's Wren",
    "Toxostoma rufum": "Brown Thrasher",
    "Turdus migratorius": "American Robin",
    "Vermivora cyanoptera": "Blue-winged Warbler",
    "Vireo solitarius": "Blue-headed Vireo",
}


def main():
    print("=" * 70)
    print("📊 SPECIES PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("\n📂 Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    report_df = pd.read_csv(REPORT_CSV, index_col=0)
    predictions_df = pd.read_csv(PREDICTIONS_CSV)
    
    # Count samples per species in each split
    train_counts = train_df['species'].value_counts()
    val_counts = val_df['species'].value_counts()
    test_counts = test_df['species'].value_counts()
    
    # Build analysis dataframe
    species_list = list(ENGLISH_NAMES.keys())
    
    analysis_data = []
    for species in species_list:
        english = ENGLISH_NAMES.get(species, species)
        
        # Sample counts
        n_train = train_counts.get(species, 0)
        n_val = val_counts.get(species, 0)
        n_test = test_counts.get(species, 0)
        n_total = n_train + n_val + n_test
        
        # Performance metrics (from classification report)
        if species in report_df.index:
            f1 = report_df.loc[species, 'f1-score']
            precision = report_df.loc[species, 'precision']
            recall = report_df.loc[species, 'recall']
            support = report_df.loc[species, 'support']
        else:
            f1 = precision = recall = 0
            support = 0
        
        # Confusion analysis
        species_preds = predictions_df[predictions_df['true_species'] == species]
        n_correct = species_preds['correct'].sum()
        n_incorrect = len(species_preds) - n_correct
        
        # Most common misclassifications
        incorrect_preds = species_preds[~species_preds['correct']]
        if len(incorrect_preds) > 0:
            top_confusion = incorrect_preds['predicted_species'].value_counts().head(3)
            top_confused_with = ', '.join([f"{s} ({c})" for s, c in top_confusion.items()])
        else:
            top_confused_with = "None"
        
        analysis_data.append({
            'species': species,
            'english_name': english,
            'train_samples': n_train,
            'val_samples': n_val,
            'test_samples': n_test,
            'total_samples': n_total,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'test_correct': n_correct,
            'test_incorrect': n_incorrect,
            'confused_with': top_confused_with
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    analysis_df = analysis_df.sort_values('f1_score', ascending=False)
    
    # Save full analysis
    analysis_path = os.path.join(OUTPUT_DIR, "species_analysis.csv")
    analysis_df.to_csv(analysis_path, index=False)
    print(f"\n📊 Full analysis saved to: {analysis_path}")
    
    # =========================================================
    # Print Summary
    # =========================================================
    print("\n" + "=" * 70)
    print("📈 PERFORMANCE BY TRAINING SAMPLE SIZE")
    print("=" * 70)
    
    # Group by sample size ranges
    def sample_category(n):
        if n < 100:
            return "Very Low (<100)"
        elif n < 300:
            return "Low (100-299)"
        elif n < 600:
            return "Medium (300-599)"
        else:
            return "High (600+)"
    
    analysis_df['sample_category'] = analysis_df['train_samples'].apply(sample_category)
    
    category_stats = analysis_df.groupby('sample_category').agg({
        'f1_score': ['mean', 'min', 'max', 'count'],
        'train_samples': 'mean'
    }).round(3)
    
    print("\n" + category_stats.to_string())
    
    # =========================================================
    # Correlation Analysis
    # =========================================================
    correlation = analysis_df['train_samples'].corr(analysis_df['f1_score'])
    print(f"\n📊 Correlation (Train Samples vs F1): {correlation:.3f}")
    
    if correlation > 0.5:
        print("   → Strong positive correlation: More samples = Better performance")
    elif correlation > 0.3:
        print("   → Moderate correlation: Sample size matters")
    else:
        print("   → Weak correlation: Other factors important")
    
    # =========================================================
    # Problem Species Report
    # =========================================================
    print("\n" + "=" * 70)
    print("⚠️ PROBLEM SPECIES (F1 < 0.3)")
    print("=" * 70)
    
    problem_species = analysis_df[analysis_df['f1_score'] < 0.3].sort_values('f1_score')
    
    print(f"\n{'Species':<35} {'English':<25} {'Train':<8} {'F1':<8} {'Confused With'}")
    print("-" * 110)
    
    for _, row in problem_species.iterrows():
        print(f"{row['species']:<35} {row['english_name']:<25} {row['train_samples']:<8} "
              f"{row['f1_score']:.3f}    {row['confused_with'][:40]}")
    
    # =========================================================
    # Top Performers
    # =========================================================
    print("\n" + "=" * 70)
    print("🏆 TOP PERFORMING SPECIES (F1 > 0.8)")
    print("=" * 70)
    
    top_species = analysis_df[analysis_df['f1_score'] > 0.8].sort_values('f1_score', ascending=False)
    
    print(f"\n{'Species':<35} {'English':<25} {'Train':<8} {'F1':<8}")
    print("-" * 80)
    
    for _, row in top_species.iterrows():
        print(f"{row['species']:<35} {row['english_name']:<25} {row['train_samples']:<8} {row['f1_score']:.3f}")
    
    # =========================================================
    # Visualization
    # =========================================================
    print("\n📊 Generating visualizations...")
    
    # Plot 1: F1 Score vs Training Samples
    plt.figure(figsize=(12, 8))
    
    colors = analysis_df['f1_score'].apply(
        lambda x: 'green' if x > 0.7 else ('orange' if x > 0.3 else 'red')
    )
    
    plt.scatter(analysis_df['train_samples'], analysis_df['f1_score'], 
                c=colors, alpha=0.7, s=100)
    
    # Add trend line
    z = np.polyfit(analysis_df['train_samples'], analysis_df['f1_score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(analysis_df['train_samples'].min(), analysis_df['train_samples'].max(), 100)
    plt.plot(x_line, p(x_line), "b--", alpha=0.5, label=f'Trend (r={correlation:.2f})')
    
    # Annotate worst performers
    for _, row in problem_species.head(5).iterrows():
        plt.annotate(row['english_name'][:15], 
                    (row['train_samples'], row['f1_score']),
                    fontsize=8, alpha=0.7)
    
    plt.xlabel('Training Samples', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Species Performance vs Training Sample Size', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot1_path = os.path.join(OUTPUT_DIR, "f1_vs_samples.png")
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    print(f"   Saved: {plot1_path}")
    
    # Plot 2: F1 Score Distribution
    plt.figure(figsize=(14, 6))
    
    # Sort by F1 score
    sorted_df = analysis_df.sort_values('f1_score', ascending=True)
    
    colors = sorted_df['f1_score'].apply(
        lambda x: '#2ecc71' if x > 0.7 else ('#f39c12' if x > 0.3 else '#e74c3c')
    )
    
    plt.barh(range(len(sorted_df)), sorted_df['f1_score'], color=colors)
    plt.yticks(range(len(sorted_df)), 
               [f"{row['english_name'][:20]}" for _, row in sorted_df.iterrows()],
               fontsize=7)
    plt.xlabel('F1 Score', fontsize=12)
    plt.title('F1 Score by Species (Sorted)', fontsize=14)
    plt.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7)')
    plt.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='Poor (0.3)')
    plt.legend()
    plt.tight_layout()
    
    plot2_path = os.path.join(OUTPUT_DIR, "f1_distribution.png")
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"   Saved: {plot2_path}")
    
    # =========================================================
    # Recommendations
    # =========================================================
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATIONS FOR IMPROVEMENT")
    print("=" * 70)
    
    n_problem = len(problem_species)
    n_zero_f1 = len(analysis_df[analysis_df['f1_score'] == 0])
    avg_problem_samples = problem_species['train_samples'].mean()
    
    print(f"""
   Current Status:
   ├── Problem species (F1 < 0.3): {n_problem}
   ├── Zero F1 species:           {n_zero_f1}
   └── Avg samples for problems:  {avg_problem_samples:.0f}

   Recommended Actions:

   1. CLASS BALANCING (High Impact):
      • Use weighted sampling during training
      • Oversample minority species (Bucephala albeola, Hirundo rustica)
      • Or use class weights in loss function

   2. DATA AUGMENTATION for Rare Species:
      • More aggressive augmentation for species with <200 samples
      • Consider synthetic data generation

   3. FOCAL LOSS (Medium Impact):
      • Replace CrossEntropy with Focal Loss
      • Automatically down-weights easy examples
      • Focuses learning on hard/rare classes

   4. COLLECT MORE DATA (If Possible):
      • Priority species: Bucephala albeola, Hirundo rustica
      • These species have 0% F1 - need more examples

   5. CONFUSION ANALYSIS:
      • Species being confused might have similar calls
      • Consider grouping similar species
    """)
    
    print(f"\n✅ Analysis complete! Check {OUTPUT_DIR} for all outputs.")


if __name__ == "__main__":
    main()