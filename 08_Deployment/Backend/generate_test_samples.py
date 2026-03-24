import os
import glob
import json
import shutil
import pandas as pd
import numpy as np

import sys
sys.path.append(os.path.dirname(__file__))

# Import services
from services.inference import load_model, predict_batch, get_english_name, get_all_species
from services.audio import load_audio, process_audio_to_spectrograms

def main():
    print("Loading test labels...")
    test_csv = r"c:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\04_Labels\Train_Val_Test_Split\test_v3.csv"
    df = pd.read_csv(test_csv)
    
    # xc_id to expected english name
    df["xc_id"] = df["xc_id"].astype(str)
    xc_to_species = dict(zip(df["xc_id"], df["species_english"]))
    
    with open(r"c:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\tmp_audio_test_candidates.json", "r") as f:
        candidates = json.load(f)
        
    print(f"Loaded {len(candidates)} candidate audio files")
    
    print("Loading model...")
    load_model()
    
    output_dir = r"c:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\08_Deployment\Frontend\public\test_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    accepted_samples = []
    seen_species = set()
    
    print("Evaluating models...")
    for file_path in candidates:
        if len(accepted_samples) >= 20:
            break
            
        basename = os.path.basename(file_path)
        xc_id = basename.split(".")[0].replace("XC", "")
        expected_species = xc_to_species.get(xc_id)
        
        if not expected_species:
            continue
            
        if expected_species in seen_species:
            continue
            
        try:
            audio, sr = load_audio(file_path)
            chunks = process_audio_to_spectrograms(audio, sr)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
            
        if not chunks:
            continue
            
        spectrograms = [c[0] for c in chunks]
        results = predict_batch(spectrograms)
        
        # Check if the highest confidence prediction matches expected
        best_pred = None
        best_conf = 0.0
        
        for res in results:
            predicted_english = get_english_name(res["predicted_class"])
            if predicted_english == expected_species:
                if res["confidence"] > best_conf:
                    best_conf = res["confidence"]
                    best_pred = predicted_english
                    
        if best_conf > 0.90:
            print(f"[{len(accepted_samples)+1}/20] Accepted {basename}: {expected_species} (Conf: {best_conf:.2f})")
            
            # Use English name directly or format it cleanly
            formatted_name = expected_species.replace(" ", "_").replace("'", "")
            new_filename = f"{formatted_name}.ogg"
            
            dest_path = os.path.join(output_dir, new_filename)
            shutil.copy2(file_path, dest_path)
            
            accepted_samples.append({
                "id": expected_species,
                "file": f"/test_samples/{new_filename}",
                "confidence": float(best_conf),
                "original_file": basename
            })
            seen_species.add(expected_species)
            
    print(f"Finished selecting {len(accepted_samples)} samples.")
    
    with open(os.path.join(output_dir, "test_samples_meta.json"), "w") as f:
        json.dump(accepted_samples, f, indent=2)

if __name__ == "__main__":
    main()
