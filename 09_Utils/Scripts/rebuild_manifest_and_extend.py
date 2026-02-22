"""
Script: rebuild_manifest_and_extend.py
Purpose: Rebuild manifest from actual files and extend remaining ones
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
MIN_DURATION = 5 * 60

def rebuild_manifest():
    """Rebuild manifest from actual files"""
    manifest_list = []
    wav_files = sorted([f for f in os.listdir(TESTING_DIR) if f.endswith('.wav')])
    
    for i, filename in enumerate(wav_files, 1):
        filepath = os.path.join(TESTING_DIR, filename)
        size_kb = round(os.path.getsize(filepath) / 1024, 1)
        
        # Try to get duration
        try:
            y, sr = librosa.load(filepath, sr=None)
            duration = len(y) / sr
            duration_min = duration / 60
        except:
            duration = None
            duration_min = 0
        
        manifest_list.append({
            'index': i,
            'filename': filename,
            'file_size_kb': size_kb,
            'duration_seconds': duration,
            'duration_minutes': round(duration_min, 1) if duration else 0
        })
    
    manifest = {
        'description': 'Testing audio files with 5+ minute recordings',
        'total_files': len(manifest_list),
        'files': manifest_list
    }
    
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest

def extend_audio(filepath, target_duration=300):
    """Extend audio to target duration"""
    try:
        y, sr = librosa.load(filepath, sr=None)
        current_duration = len(y) / sr
        
        if current_duration >= target_duration:
            return current_duration
        
        reps = int(np.ceil(target_duration / current_duration))
        y_extended = np.tile(y, reps)
        
        target_samples = int(target_duration * sr)
        y_extended = y_extended[:target_samples]
        
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        
        sf.write(temp_path, y_extended, sr, subtype='PCM_16')
        shutil.move(temp_path, filepath)
        
        return len(y_extended) / sr
    except Exception as e:
        print(f"     Error: {e}")
        return None

def main():
    print("="*70)
    print("REBUILDING MANIFEST & EXTENDING TO 5 MINUTES")
    print("="*70)
    
    # Rebuild manifest
    print("\nRebuilding manifest from files...")
    manifest = rebuild_manifest()
    
    print(f"   Found {manifest['total_files']} files\n")
    
    extended = 0
    failed = 0
    
    for entry in manifest['files']:
        filepath = os.path.join(TESTING_DIR, entry['filename'])
        duration_min = entry['duration_minutes']
        
        if duration_min >= 5:
            print(f"OK  {entry['filename']:<25} {duration_min:.1f} min")
            continue
        
        # Extend
        new_dur = extend_audio(filepath, MIN_DURATION)
        
        if new_dur:
            new_min = new_dur / 60
            print(f"EXTENDED {entry['filename']:<20} {duration_min:.1f} to {new_min:.1f} min")
            extended += 1
            entry['duration_seconds'] = new_dur
            entry['duration_minutes'] = round(new_dur / 60, 1)
            entry['file_size_kb'] = round(os.path.getsize(filepath) / 1024, 1)
        else:
            print(f"FAILED {entry['filename']:<25}")
            failed += 1
    
    # Save updated manifest
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Extended: {extended}")
    print(f"Failed: {failed}")
    print(f"Testing folder ready: {TESTING_DIR}")

if __name__ == "__main__":
    main()
