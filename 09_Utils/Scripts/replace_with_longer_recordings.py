"""
Script: replace_with_longer_recordings.py
Purpose: Replace short audio chunks with longer full recordings that include background noise
"""

import os
import shutil
import json
from pathlib import Path

# Configuration
BASE_DIR = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"
TESTING_DIR = os.path.join(BASE_DIR, "Testing")
STANDARDIZED_AUDIO_DIR = os.path.join(BASE_DIR, "02_Preprocessed", "Standardized_Audio")
RAW_AUDIO_DIR = os.path.join(BASE_DIR, "01_Raw_Data", "Audio_Recordings")
MANIFEST_PATH = os.path.join(TESTING_DIR, "testing_manifest.json")

def get_longer_version(recording_id):
    """
    Find the full/longer version of a recording by its ID.
    The full versions are typically in Standardized_Audio directory.
    """
    # Try standardized audio first (these are full recordings, not chunks)
    std_file = os.path.join(STANDARDIZED_AUDIO_DIR, f"{recording_id}.wav")
    if os.path.exists(std_file):
        return std_file
    
    # Try raw audio directory
    raw_file = os.path.join(RAW_AUDIO_DIR, f"{recording_id}.wav")
    if os.path.exists(raw_file):
        return raw_file
    
    # Try with .mp3
    for ext in ['.mp3', '.ogg', '.flac']:
        std_file = os.path.join(STANDARDIZED_AUDIO_DIR, f"{recording_id}{ext}")
        if os.path.exists(std_file):
            return std_file
        
        raw_file = os.path.join(RAW_AUDIO_DIR, f"{recording_id}{ext}")
        if os.path.exists(raw_file):
            return raw_file
    
    return None

def main():
    print("=" * 70)
    print("🎵 REPLACING WITH LONGER RECORDINGS (WITH BACKGROUND NOISE)")
    print("=" * 70)
    
    # Load manifest
    if not os.path.exists(MANIFEST_PATH):
        print(f"\n❌ Manifest not found: {MANIFEST_PATH}")
        return
    
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
    
    print(f"\n📋 Found {len(manifest['files'])} files to replace")
    
    replaced = 0
    not_found = []
    
    for entry in manifest['files']:
        recording_id = entry['recording_id']
        current_file = os.path.join(TESTING_DIR, os.path.basename(entry['filename']))
        
        # Find longer version
        longer_file = get_longer_version(recording_id)
        
        if longer_file and os.path.exists(longer_file):
            try:
                # Get original extension
                ext = os.path.splitext(current_file)[1]
                
                # Remove old file
                if os.path.exists(current_file):
                    os.remove(current_file)
                
                # Copy longer version
                new_filename = f"{recording_id}{ext}"
                new_path = os.path.join(TESTING_DIR, new_filename)
                shutil.copy2(longer_file, new_path)
                
                old_size = entry['file_size_kb']
                new_size = os.path.getsize(new_path) / 1024
                size_increase = ((new_size - old_size) / old_size) * 100
                
                print(f"✅ {entry['species_english']:<35} {old_size:>8.0f}KB → {new_size:>8.0f}KB (+{size_increase:.0f}%)")
                replaced += 1
                
                # Update manifest
                entry['filename'] = new_filename
                entry['file_size_kb'] = round(new_size, 1)
                entry['source_file'] = f"{recording_id} (full recording)"
                
            except Exception as e:
                print(f"❌ {entry['species_english']:<35} ERROR: {str(e)}")
                not_found.append(recording_id)
        else:
            print(f"⚠️  {entry['species_english']:<35} Full recording not found")
            not_found.append(recording_id)
    
    # Update manifest file
    if replaced > 0:
        with open(MANIFEST_PATH, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"\n✅ Manifest updated: {MANIFEST_PATH}")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"📊 SUMMARY")
    print("=" * 70)
    print(f"   ✅ Replaced:  {replaced}/{len(manifest['files'])}")
    print(f"   ⚠️  Not found: {len(not_found)}")
    
    if not_found:
        print(f"\n❌ Could not find full recordings for:")
        for rid in not_found:
            print(f"   - {rid}")
    
    if replaced == len(manifest['files']):
        print(f"\n🎉 ALL FILES REPLACED WITH LONGER RECORDINGS!")
        print(f"\n📂 Testing folder: {TESTING_DIR}")
        print(f"   Files are now full recordings with background noise")
    
    # Show some stats
    print(f"\n📈 NEW FILE SIZES:")
    total_size = sum(f['file_size_kb'] for f in manifest['files'])
    avg_size = total_size / len(manifest['files'])
    print(f"   Total: {total_size:.0f} KB")
    print(f"   Average: {avg_size:.0f} KB per file")
    print(f"   Estimated duration: {avg_size * 8 / 128:.1f} seconds per file (at 128 kbps)")

if __name__ == "__main__":
    main()
