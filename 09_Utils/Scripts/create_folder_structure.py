"""
Project Folder Structure Creation Script (Updated)
===================================================
Creates organized folder structure for Bird Detection Project
Accounts for existing files in the directory
"""

import os
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

# Base project folder
BASE_FOLDER = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"

# ============================================================
# FOLDER STRUCTURE DEFINITION
# ============================================================

FOLDER_STRUCTURE = {
    # Phase 0: Raw Data & Metadata
    "01_Raw_Data": {
        "Audio_Recordings": "Original 4521 WAV files from Xeno-Canto",
        "Metadata": "CSV files with recording metadata",
        "External_Data": "External datasets (BirdCLEF spectrograms, etc.)",
    },
    
    # Phase 1: Preprocessing
    "02_Preprocessed": {
        "Standardized_Audio": "Resampled, mono audio files (22050 Hz)",
        "Audio_Chunks": "Fixed-length 5-second segments",
        "Quality_Reports": "Audio analysis and quality reports",
    },
    
    # Phase 2: Features
    "03_Features": {
        "Spectrograms": "Mel-spectrograms generated from our audio",
        "Embeddings": "Audio embeddings (if using pretrained models)",
    },
    
    # Phase 3: Labels
    "04_Labels": {
        "Raw_Labels": "Original label files from metadata",
        "Processed_Labels": "Multi-label encoded files",
        "Train_Val_Test_Split": "Data split information",
    },
    
    # Phase 4-5: Model
    "05_Model": {
        "Checkpoints": "Model checkpoints during training",
        "Saved_Models": "Final trained models (.pth, .h5)",
        "Training_Logs": "TensorBoard logs, training history",
        "Configs": "Model configuration YAML/JSON files",
    },
    
    # Phase 6: Explainability
    "06_Explainability": {
        "GradCAM": "Grad-CAM visualizations",
        "Attention_Maps": "Attention heatmaps",
        "Temporal_Localization": "Time-based detection results",
    },
    
    # Phase 7: Evaluation
    "07_Evaluation": {
        "Metrics": "Performance metrics and reports",
        "Confusion_Matrices": "Confusion matrix visualizations",
        "Predictions": "Model predictions on test set",
    },
    
    # Phase 8: Deployment
    "08_Deployment": {
        "API": "FastAPI backend code",
        "Frontend": "Streamlit/Gradio UI code",
        "Docker": "Docker configuration files",
    },
    
    # Utilities & Scripts
    "09_Utils": {
        "Scripts": "All Python utility scripts",
        "Notebooks": "Jupyter notebooks for exploration",
        "Logs": "General processing logs",
        "Temp": "Temporary files (can be deleted)",
    },
    
    # Outputs & Results
    "10_Outputs": {
        "Reports": "Generated reports (PDF, HTML)",
        "Visualizations": "Charts, graphs, figures",
        "Exports": "Exported data for sharing",
    },
}

# ============================================================
# FOLDER CREATION FUNCTION
# ============================================================

def create_folder_structure():
    """Create the complete project folder structure"""
    
    print("=" * 65)
    print("📁 PROJECT FOLDER STRUCTURE CREATION")
    print("=" * 65)
    print(f"\n📍 Base folder: {BASE_FOLDER}")
    print(f"📅 Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if base folder exists
    if not os.path.exists(BASE_FOLDER):
        print(f"\n❌ Base folder does not exist: {BASE_FOLDER}")
        return False
    
    created_count = 0
    existing_count = 0
    
    print("\n" + "-" * 65)
    print("📂 Creating folder structure...")
    print("-" * 65)
    
    # Create main folders and subfolders
    for main_folder, subfolders in FOLDER_STRUCTURE.items():
        main_path = os.path.join(BASE_FOLDER, main_folder)
        
        # Create main folder
        if not os.path.exists(main_path):
            os.makedirs(main_path)
            created_count += 1
            print(f"\n✅ Created: {main_folder}/")
        else:
            existing_count += 1
            print(f"\n📂 Exists:  {main_folder}/")
        
        # Create subfolders
        for subfolder, description in subfolders.items():
            sub_path = os.path.join(main_path, subfolder)
            
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)
                created_count += 1
                print(f"   ✅ Created: {subfolder}/")
            else:
                existing_count += 1
                print(f"   📂 Exists:  {subfolder}/")
            
            # Create README in each subfolder
            readme_path = os.path.join(sub_path, "README.txt")
            if not os.path.exists(readme_path):
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(f"{'=' * 50}\n")
                    f.write(f"Folder: {subfolder}\n")
                    f.write(f"{'=' * 50}\n\n")
                    f.write(f"Purpose: {description}\n\n")
                    f.write(f"Parent: {main_folder}\n")
                    f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create main PROJECT_STRUCTURE.md
    create_project_readme()
    
    # Summary
    print("\n" + "=" * 65)
    print("📊 CREATION SUMMARY")
    print("=" * 65)
    print(f"\n✅ New folders created: {created_count}")
    print(f"📂 Existing folders: {existing_count}")
    print(f"📄 Documentation: PROJECT_STRUCTURE.md")
    
    return True


def create_project_readme():
    """Create detailed PROJECT_STRUCTURE.md documentation"""
    
    readme_path = os.path.join(BASE_FOLDER, "PROJECT_STRUCTURE.md")
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# 🐦 Bird Detection Project - Folder Structure\n\n")
        f.write("## Project: Confidence-Aware, Explainable Multi-Species Bird Detection\n\n")
        f.write(f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Visual tree
        f.write("## 📁 Directory Tree\n\n")
        f.write("```\n")
        f.write("ML Conf-BioFSL/\n")
        f.write("│\n")
        
        folder_items = list(FOLDER_STRUCTURE.items())
        for i, (main_folder, subfolders) in enumerate(folder_items):
            is_last_main = (i == len(folder_items) - 1)
            main_prefix = "└──" if is_last_main else "├──"
            f.write(f"{main_prefix} {main_folder}/\n")
            
            subfolder_items = list(subfolders.items())
            for j, (subfolder, desc) in enumerate(subfolder_items):
                is_last_sub = (j == len(subfolder_items) - 1)
                if is_last_main:
                    sub_prefix = "    └──" if is_last_sub else "    ├──"
                else:
                    sub_prefix = "│   └──" if is_last_sub else "│   ├──"
                f.write(f"{sub_prefix} {subfolder}/\n")
            
            if not is_last_main:
                f.write("│\n")
        
        f.write("```\n\n")
        
        # Detailed descriptions
        f.write("---\n\n")
        f.write("## 📋 Folder Descriptions\n\n")
        
        for main_folder, subfolders in FOLDER_STRUCTURE.items():
            f.write(f"### 📂 {main_folder}\n\n")
            f.write("| Subfolder | Purpose |\n")
            f.write("|-----------|----------|\n")
            for subfolder, description in subfolders.items():
                f.write(f"| `{subfolder}/` | {description} |\n")
            f.write("\n")
        
        # Phase mapping
        f.write("---\n\n")
        f.write("## 🗺️ Phase to Folder Mapping\n\n")
        f.write("| Phase | Primary Folders |\n")
        f.write("|-------|----------------|\n")
        f.write("| Phase 0: Metadata | `01_Raw_Data/Metadata/` |\n")
        f.write("| Phase 1: Preprocessing | `01_Raw_Data/` → `02_Preprocessed/` |\n")
        f.write("| Phase 2: Augmentation | `02_Preprocessed/` → `03_Features/` |\n")
        f.write("| Phase 3: Labels | `04_Labels/` |\n")
        f.write("| Phase 4-5: Model | `05_Model/` |\n")
        f.write("| Phase 6: Explainability | `06_Explainability/` |\n")
        f.write("| Phase 7: Evaluation | `07_Evaluation/` |\n")
        f.write("| Phase 8: Deployment | `08_Deployment/` |\n")
        f.write("\n")
    
    print(f"\n📄 Created: PROJECT_STRUCTURE.md")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "🐦" * 25)
    print("\n   BIRD DETECTION PROJECT - FOLDER SETUP")
    print("\n" + "🐦" * 25 + "\n")
    
    success = create_folder_structure()
    
    if success:
        print("\n" + "=" * 65)
        print("✅ FOLDER STRUCTURE CREATED SUCCESSFULLY!")
        print("=" * 65)
        print("""
📋 NEXT STEP:
   
   Run the file organization script to move existing files
   to their correct locations.
        """)
    else:
        print("\n❌ Failed to create folder structure")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()