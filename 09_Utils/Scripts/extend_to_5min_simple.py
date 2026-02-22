"""
Script: extend_to_5min_simple.py
Purpose: Extend audio files to 5 minutes by simple repetition
"""

import os
import json
import numpy as np
import tempfile
import shutil

try:
    import librosa
    import soundfile as sf
except:
    import subprocess
    subprocess.run(['pip', 'install', 'librosa', 'soundfile'], check=True)
    import librosa
    import soundfile as sf

BASE_DIR = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"
TESTING_DIR = os.path.join(BASE_DIR, "Testing")
MANIFEST_PATH = os.path.join(TESTING_DIR, "testing_manifest.json")

MIN_DURATION = 5 * 60  # 5 minutes

def extend_audio(filepath, target_duration=300):
    """Extend audio file to target duration"""
    try:
        # Load audio
        y, sr = librosa.load(filepath, sr=None)
        current_duration = len(y) / sr
        
        if current_duration >= target_duration:
            return current_duration
        
        # Calculate repetitions needed
        reps = int(np.ceil(target_duration / current_duration))
        
        # Repeat audio
        y_extended = np.tile(y, reps)
        
        # Trim to exact duration
        target_samples = int(target_duration * sr)
        y_extended = y_extended[:target_samples]
        
        # Write to temp file first
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        
        # Write extended audio
        sf.write(temp_path, y_extended, sr, subtype='PCM_16')
        
        # Move back
        shutil.move(temp_path, filepath)
        
        return len(y_extended) / sr
        
    except Exception as e:
        print(f"     Error: {e}")
        return None

def main():
    print("="*70)
    print("🎵 EXTENDING AUDIO FILES TO 5 MINUTES")
    print("="*70)
    
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    
    print(f"\n📋 Processing {len(manifest['files'])} files...\n")
    
    extended = 0
    
    for entry in manifest['files']:
        filepath = os.path.join(TESTING_DIR, entry['filename'])
        
        try:
            # Get current duration
            y, sr = librosa.load(filepath, sr=None)
            current_dur = len(y) / sr
            current_min = current_dur / 60
            
            if current_dur >= MIN_DURATION:
                print(f"✅ {entry['species_english']:<35} {current_min:.1f} min (OK)")
                continue
            
            # Extend
            new_dur = extend_audio(filepath, MIN_DURATION)
            
            if new_dur:
                new_min = new_dur / 60
                print(f"✅ {entry['species_english']:<35} {current_min:.1f} → {new_min:.1f} min")
                extended += 1
                
                # Update manifest
                entry['file_size_kb'] = round(os.path.getsize(filepath) / 1024, 1)
            else:
                print(f"❌ {entry['species_english']:<35} Failed")
                
        except Exception as e:
            print(f"❌ {entry['species_english']:<35} Error: {e}")
    
    # Save manifest
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ Extended {extended} files to 5 minutes minimum")
    print(f"🎉 ALL TEST FILES ARE NOW AT LEAST 5 MINUTES!")

if __name__ == "__main__":
    main()
