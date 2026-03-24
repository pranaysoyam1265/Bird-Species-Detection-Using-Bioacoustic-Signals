import os
import shutil
import requests
import json
from pathlib import Path
import random

API_URL = "https://pranaysoyam126-birdsense-backend.hf.space"
TARGET_COUNT = 20
OUTPUT_DIR = Path(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\08_Deployment\Frontend\public\test_samples")

def find_confident_samples():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Search paths
    search_paths = [
        Path(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\01_Raw_Data\Audio_Recordings"),
        Path(r"C:\BirdSense_P2\data\audio"),
        Path(r"C:\BirdSense_P2\data\phase1_rebuilt\downloads"),
        Path(r"C:\BirdSense_P2\data\phase2_rebuilt\downloads")
    ]
    
    all_files = []
    for sp in search_paths:
        if sp.exists():
            all_files.extend(list(sp.rglob("*.wav")) + list(sp.rglob("*.mp3")))
            
    print(f"Found {len(all_files)} total audio files to search.")
    random.shuffle(all_files) # shuffle to get diverse species
    
    success_count = 0
    collected_species = set()
    
    for audio_path in all_files:
        if success_count >= TARGET_COUNT:
            break
            
        # Skip large files to save API time
        try:
            if audio_path.stat().st_size > 1_000_000:
                continue
        except:
            continue
            
        try:
            with open(audio_path, 'rb') as f:
                files = {'audio_file': (audio_path.name, f)}
                data = {'top_k': 1, 'confidence_threshold': 0.01, 'noise_reduction': False, 'chunk_duration': 5.0}
                r = requests.post(f"{API_URL}/detect", files=files, data=data, timeout=60)
                
            if r.status_code == 200:
                result = r.json()
                species = result.get('top_species')
                conf = result.get('top_confidence', 0)
                
                # Only take if confident > 75% and we haven't already collected this species
                if conf > 75.0 and species not in collected_species:
                    # Clean the species name for filename
                    safe_name = "".join([c if c.isalnum() else "_" for c in species]).replace("__", "_").strip("_")
                    ext = audio_path.suffix
                    dest_path = OUTPUT_DIR / f"{safe_name}{ext}"
                    
                    shutil.copy2(audio_path, dest_path)
                    collected_species.add(species)
                    success_count += 1
                    print(f"[{success_count}/{TARGET_COUNT}] Found {species} ({conf}%) -> Saved to {dest_path.name}")
        except Exception as e:
            pass

if __name__ == "__main__":
    find_confident_samples()
    print("Done collecting samples!")
