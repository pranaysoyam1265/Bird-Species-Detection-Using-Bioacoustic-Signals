import os
import glob
import json
import shutil
import pandas as pd
import numpy as np

# Set project root to the parent of 08_Deployment
import sys
PROJECT_ROOT = r"c:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"
BACKEND_DIR = os.path.join(PROJECT_ROOT, "08_Deployment", "Backend")
sys.path.append(BACKEND_DIR)

# Import services
from services.inference import load_model, predict_batch, get_english_name
from services.audio import load_audio, process_audio_to_spectrograms

def main():
    print("Loading test labels...")
    test_csv = os.path.join(PROJECT_ROOT, "04_Labels", "Train_Val_Test_Split", "test_v3.csv")
    df = pd.read_csv(test_csv)
    
    # xc_id to expected english name
    df["xc_id"] = df["xc_id"].astype(str)
    xc_to_species = dict(zip(df["xc_id"], df["species_english"]))
    
    # Search locations
    search_paths = [
        os.path.join(PROJECT_ROOT, "01_Raw_Data", "Audio_Recordings"),
        r"C:\BirdSense_P2\data\audio"
    ]
    
    candidates = []
    print("Searching for audio files...")
    for path in search_paths:
        if not os.path.exists(path):
            print(f"Path not found: {path}")
            continue
        files = glob.glob(os.path.join(path, "**", "*.*"), recursive=True)
        print(f"Found {len(files)} files in {path}")
        for f in files:
            basename = os.path.basename(f)
            if basename.startswith("XC"):
                xc_id = basename.split(".")[0].split("_")[0].replace("XC", "")
                if xc_id in xc_to_species:
                    candidates.append((xc_id, f))
                    
    print(f"Total processed candidates mapped to test set: {len(candidates)}")
    
    # Shuffle candidates to get a variety
    import random
    random.seed(42)
    random.shuffle(candidates)
    
    print("Loading model...")
    try:
        load_model()
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        return
    
    output_dir = os.path.join(PROJECT_ROOT, "08_Deployment", "Frontend", "public", "test_samples")
    os.makedirs(output_dir, exist_ok=True)
    
    accepted_samples = []
    seen_species = set()
    
    print("Evaluating models (aiming for 25 samples to have buffer)...")
    for xc_id, file_path in candidates:
        if len(accepted_samples) >= 30: # Get a few extra
            break
            
        expected_species = xc_to_species.get(xc_id)
        if not expected_species or expected_species in seen_species:
            continue
            
        try:
            audio, sr = load_audio(file_path)
            # Take only first 15 seconds to speed up
            if len(audio) > sr * 15:
                audio = audio[:sr * 15]
            chunks = process_audio_to_spectrograms(audio, sr)
        except Exception as e:
            # print(f"Error loading {file_path}: {e}")
            continue
            
        if not chunks:
            continue
            
        spectrograms = [c[0] for c in chunks]
        results = predict_batch(spectrograms)
        
        best_conf = 0.0
        for res in results:
            predicted_english = get_english_name(res["predicted_class"])
            if predicted_english == expected_species:
                if res["confidence"] > best_conf:
                    best_conf = res["confidence"]
                    
        if best_conf > 0.85: # Slightly lower threshold to ensure we get 20+ variety
            print(f"[{len(accepted_samples)+1}] Accepted {os.path.basename(file_path)}: {expected_species} (Conf: {best_conf:.2f})")
            
            clean_name = expected_species.replace(" ", "_").replace("'", "").replace("-", "_")
            ext = os.path.splitext(file_path)[1]
            new_filename = f"{clean_name}{ext}"
            
            dest_path = os.path.join(output_dir, new_filename)
            shutil.copy2(file_path, dest_path)
            
            accepted_samples.append({
                "species": expected_species,
                "scientific_name": xc_id, # Using xc_id as a proxy if needed, but the user asked for names as IDs?
                "id": clean_name, # This will be used as the identifier in Frontend
                "file": f"/test_samples/{new_filename}",
                "confidence": float(best_conf),
                "original_file": os.path.basename(file_path)
            })
            seen_species.add(expected_species)
            
    print(f"Finished selecting {len(accepted_samples)} samples.")
    
    meta_path = os.path.join(output_dir, "test_samples_meta.json")
    with open(meta_path, "w") as f:
        json.dump(accepted_samples, f, indent=2)
    print(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    main()
