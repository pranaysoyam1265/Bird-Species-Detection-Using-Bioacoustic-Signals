"""
Script: extend_to_5min.py
Purpose: Extend audio files to at least 5 minutes by repeating or concatenating
Uses librosa and soundfile for audio processing
"""

import os
import json
import numpy as np
from pathlib import Path

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.run(['pip', 'install', 'librosa', 'soundfile', 'scikit-learn', 'numba'], check=True)
    import librosa
    import soundfile as sf

BASE_DIR = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"
TESTING_DIR = os.path.join(BASE_DIR, "Testing")
AUDIO_CHUNKS_DIR = os.path.join(BASE_DIR, "02_Preprocessed", "Audio_Chunks")
MANIFEST_PATH = os.path.join(TESTING_DIR, "testing_manifest.json")

MIN_DURATION = 5 * 60  # 5 minutes in seconds

def get_audio_duration(filepath):
    """Get duration of audio file"""
    try:
        y, sr = librosa.load(filepath, sr=None)
        return len(y) / sr
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return 0

def extend_audio(input_file, output_file, target_duration=300):
    """Extend audio to target duration by concatenating with itself"""
    try:
        # Load audio
        y, sr = librosa.load(input_file, sr=None)
        current_duration = len(y) / sr
        
        if current_duration >= target_duration:
            # Already long enough, just copy
            sf.write(output_file, y, sr)
            return current_duration
        
        # Calculate how many times to repeat
        repetitions = int(np.ceil(target_duration / current_duration))
        
        # Concatenate audio multiple times
        extended = np.tile(y, repetitions)
        
        # Trim to exact target duration
        target_samples = int(target_duration * sr)
        extended = extended[:target_samples]
        
        # Save
        sf.write(output_file, extended, sr)
        return len(extended) / sr
        
    except Exception as e:
        print(f"Error extending {input_file}: {e}")
        return None

def extend_with_chunks(species, input_file, output_file, target_duration=300):
    """Extend by adding chunks from other recordings of same species if available"""
    try:
        # Load main audio
        y_main, sr = librosa.load(input_file, sr=None)
        current_duration = len(y_main) / sr
        
        if current_duration >= target_duration:
            sf.write(output_file, y_main, sr)
            return current_duration
        
        # Try to find chunks from same species
        audio_list = [y_main]
        species_name = species.replace(" ", "_")
        
        remaining_duration = target_duration - current_duration
        
        # Search for species subdirectory in chunks
        species_dir = os.path.join(AUDIO_CHUNKS_DIR, species_name)
        if os.path.exists(species_dir):
            chunk_files = [f for f in os.listdir(species_dir) if f.endswith('.wav')]
            chunk_files.sort()
            
            for chunk_file in chunk_files:
                if remaining_duration <= 0:
                    break
                
                chunk_path = os.path.join(species_dir, chunk_file)
                try:
                    y_chunk, sr_chunk = librosa.load(chunk_path, sr=sr)
                    audio_list.append(y_chunk)
                    remaining_duration -= len(y_chunk) / sr
                except:
                    continue
        
        # If still not enough, just repeat the main audio
        if remaining_duration > 0:
            repetitions = int(np.ceil(remaining_duration / current_duration))
            for _ in range(repetitions):
                audio_list.append(y_main)
        
        # Concatenate all
        extended = np.concatenate(audio_list)
        
        # Trim to target
        target_samples = int(target_duration * sr)
        extended = extended[:target_samples]
        
        # Save
        sf.write(output_file, extended, sr)
        return len(extended) / sr
        
    except Exception as e:
        print(f"Error extending with chunks {input_file}: {e}")
        return None

def main():
    print("=" * 70)
    print("🎵 EXTENDING AUDIO FILES TO 5 MINUTES MINIMUM")
    print("=" * 70)
    
    # Load manifest
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    
    print(f"\n📋 Processing {len(manifest['files'])} files...")
    
    extended_count = 0
    skipped = 0
    
    for entry in manifest['files']:
        testing_file = os.path.join(TESTING_DIR, entry['filename'])
        species = entry['species_scientific']
        
        try:
            duration = get_audio_duration(testing_file)
            minutes = duration / 60
            
            if duration >= MIN_DURATION:
                print(f"   ✅ {entry['species_english']:<35} {minutes:.1f} min (already long enough)")
                skipped += 1
            else:
                # Try to extend with chunks first
                new_duration = extend_with_chunks(species, testing_file, testing_file, MIN_DURATION)
                
                if new_duration:
                    new_minutes = new_duration / 60
                    print(f"   ✅ {entry['species_english']:<35} {minutes:.1f} → {new_minutes:.1f} min")
                    extended_count += 1
                    
                    # Update manifest
                    entry['file_size_kb'] = round(os.path.getsize(testing_file) / 1024, 1)
                else:
                    print(f"   ❌ {entry['species_english']:<35} Failed to extend")
        
        except Exception as e:
            print(f"   ❌ {entry['species_english']:<35} Error: {str(e)}")
    
    # Update manifest
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY")
    print(f"   ✅ Extended:  {extended_count}")
    print(f"   ⏭️  Skipped (already ≥5min): {skipped}")
    print(f"\n🎉 ALL FILES ARE NOW AT LEAST 5 MINUTES LONG!")
    print(f"📁 Testing folder: {TESTING_DIR}")

if __name__ == "__main__":
    main()
