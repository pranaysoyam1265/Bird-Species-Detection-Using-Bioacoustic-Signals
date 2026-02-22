"""
Script: ensure_5min_duration.py
Purpose: Verify all test files are at least 5 minutes long, replace shorter ones
"""

import os
import shutil
import json
import subprocess
from pathlib import Path

BASE_DIR = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"
TESTING_DIR = os.path.join(BASE_DIR, "Testing")
STANDARDIZED_AUDIO_DIR = os.path.join(BASE_DIR, "02_Preprocessed", "Standardized_Audio")
RAW_AUDIO_DIR = os.path.join(BASE_DIR, "01_Raw_Data", "Audio_Recordings")
MANIFEST_PATH = os.path.join(TESTING_DIR, "testing_manifest.json")

MIN_DURATION = 5 * 60  # 5 minutes in seconds

def get_audio_duration(filepath):
    """Get duration of audio file using ffprobe (fallback to estimate)"""
    try:
        # Try ffprobe first
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
             '-of', 'default=noprint_wrappers=1:nokey=1:novalue=1', filepath],
            capture_output=True, text=True, timeout=10
        )
        if result.stdout:
            return float(result.stdout.strip())
    except:
        pass
    
    try:
        # Try with librosa
        import librosa
        y, sr = librosa.load(filepath, sr=None)
        return len(y) / sr
    except:
        pass
    
    # Fallback: estimate from file size (assume 128-192 kbps)
    size_bytes = os.path.getsize(filepath)
    size_bits = size_bytes * 8
    # Assume 160 kbps average
    duration = size_bits / 160000
    return duration

def find_longest_available(recording_id):
    """Find the longest available version of a recording"""
    candidates = []
    
    # Search in standardized audio
    if os.path.exists(STANDARDIZED_AUDIO_DIR):
        for f in os.listdir(STANDARDIZED_AUDIO_DIR):
            if recording_id in f or f.startswith(recording_id):
                full_path = os.path.join(STANDARDIZED_AUDIO_DIR, f)
                try:
                    duration = get_audio_duration(full_path)
                    candidates.append((duration, full_path))
                except:
                    pass
    
    # Search in raw audio
    if os.path.exists(RAW_AUDIO_DIR):
        for f in os.listdir(RAW_AUDIO_DIR):
            if recording_id in f or f.startswith(recording_id):
                full_path = os.path.join(RAW_AUDIO_DIR, f)
                try:
                    duration = get_audio_duration(full_path)
                    candidates.append((duration, full_path))
                except:
                    pass
    
    # Return the longest one
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0]
    
    return None, None

def main():
    print("=" * 70)
    print("🎵 ENSURING ALL FILES ARE AT LEAST 5 MINUTES LONG")
    print("=" * 70)
    
    # Load manifest
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    
    print(f"\n📋 Checking {len(manifest['files'])} files...")
    
    short_files = []
    long_enough = []
    
    for entry in manifest['files']:
        testing_file = os.path.join(TESTING_DIR, entry['filename'])
        
        try:
            duration = get_audio_duration(testing_file)
            minutes = duration / 60
            
            if duration < MIN_DURATION:
                short_files.append((entry, duration, testing_file))
                status = f"⚠️  {minutes:.1f} min"
            else:
                long_enough.append(entry)
                status = f"✅ {minutes:.1f} min"
            
            print(f"   {entry['species_english']:<35} {status}")
        except Exception as e:
            print(f"   {entry['species_english']:<35} ❌ Error: {str(e)}")
    
    print(f"\n📊 SUMMARY")
    print(f"   {'✅ At least 5 min:':<25} {len(long_enough)}")
    print(f"   {'⚠️  Shorter than 5 min:':<25} {len(short_files)}")
    
    if short_files:
        print(f"\n🔄 REPLACING SHORT FILES WITH LONGER VERSIONS...")
        replaced = 0
        
        for entry, duration, testing_file in short_files:
            recording_id = entry['recording_id']
            duration_min = duration / 60
            
            # Find longer version
            found_duration, longer_file = find_longest_available(recording_id)
            
            if longer_file and found_duration and found_duration >= MIN_DURATION:
                try:
                    os.remove(testing_file)
                    ext = os.path.splitext(testing_file)[1]
                    shutil.copy2(longer_file, testing_file)
                    
                    new_duration = found_duration / 60
                    print(f"   ✅ {entry['species_english']:<35} {duration_min:.1f}min → {new_duration:.1f}min")
                    replaced += 1
                    
                    # Update manifest
                    entry['file_size_kb'] = round(os.path.getsize(testing_file) / 1024, 1)
                    
                except Exception as e:
                    print(f"   ❌ {entry['species_english']:<35} {str(e)}")
            else:
                if found_duration:
                    new_min = found_duration / 60
                    print(f"   ⚠️  {entry['species_english']:<35} Best available: {new_min:.1f}min (< 5min)")
                else:
                    print(f"   ❌ {entry['species_english']:<35} No longer version found")
        
        # Update manifest
        with open(MANIFEST_PATH, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✅ Manifest updated")
    
    if all(duration >= MIN_DURATION for entry, duration, _ in [(e, get_audio_duration(os.path.join(TESTING_DIR, e['filename'])), os.path.join(TESTING_DIR, e['filename'])) for e in manifest['files']]):
        print(f"\n🎉 ALL FILES ARE AT LEAST 5 MINUTES LONG!")
    
if __name__ == "__main__":
    main()
