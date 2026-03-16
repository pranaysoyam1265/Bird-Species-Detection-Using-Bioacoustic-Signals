"""
Script to identify XC recordings not used in training and check their duration
"""
import os
import re
import csv
from pathlib import Path
import subprocess
import json

# Path to audio recordings
audio_dir = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\01_Raw_Data\Audio_Recordings"
train_csv = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\04_Labels\Train_Val_Test_Split\train.csv"
val_csv = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\04_Labels\Train_Val_Test_Split\val.csv"
test_csv = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\04_Labels\Train_Val_Test_Split\test.csv"

# Get used recording IDs
used_ids = set()

for csv_file in [train_csv, val_csv, test_csv]:
    if os.path.exists(csv_file):
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract XC ID from chunk_file or recording_id
                if 'recording_id' in row:
                    used_ids.add(row['recording_id'])
                elif 'chunk_file' in row:
                    match = re.search(r'XC(\d+)_', row['chunk_file'])
                    if match:
                        used_ids.add(match.group(1))

print(f"✓ Found {len(used_ids)} recording IDs used in training/val/test")

# Get all available recordings
all_files = sorted([f for f in os.listdir(audio_dir) if f.startswith('XC') and (f.endswith('.mp3') or f.endswith('.wav'))])
all_ids = set()
id_to_file = {}

for filename in all_files:
    match = re.match(r'XC(\d+)', filename)
    if match:
        xc_id = match.group(1)
        all_ids.add(xc_id)
        if xc_id not in id_to_file:
            id_to_file[xc_id] = []
        id_to_file[xc_id].append(filename)

print(f"✓ Found {len(all_ids)} total unique recording IDs in audio folder")

# Find unused recordings
unused_ids = all_ids - used_ids
print(f"\n✓ Found {len(unused_ids)} unused recording IDs")

# Get duration of audio files (first 30 unused records for quick check)
print("\n📊 Checking duration of first 30 unused recordings...")
unused_list = sorted(list(unused_ids))[:30]

durations = {}
for xc_id in unused_list:
    for filename in id_to_file[xc_id]:
        filepath = os.path.join(audio_dir, filename)
        try:
            # Use ffprobe to get duration
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1:nokey=1',
                filepath
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.stdout.strip():
                duration_sec = float(result.stdout.strip())
                duration_min = duration_sec / 60
                durations[filename] = duration_min
                status = "✓ Long" if duration_min > 5 else "✗ Short"
                print(f"  {filename}: {duration_min:.1f} min {status}")
        except Exception as e:
            print(f"  {filename}: Error - {e}")

# Recommendation
print("\n" + "="*70)
print("RECOMMENDATIONS FOR TEST DATA:")
print("="*70)
print(f"\n1. UNUSED RECORDINGS: {len(unused_ids)} total available")
print(f"   - These are original Xeno-Canto recordings NOT used in training")
print(f"   - Location: {audio_dir}")
print(f"   - Format: Mix of MP3 and WAV files")
print(f"\n2. TO GET 20+ LONG SAMPLES (>5 min):")
print(f"   - Filter from {len(unused_ids)} unused recordings")
print(f"   - Quick sample suggests ~{len([f for f in durations.values() if f > 5])}/{len(durations)} are >5min")
print(f"   - Should easily find 20+ samples\n")

# Save list of unused IDs to file for later use
output_file = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\unused_recording_ids.json"
with open(output_file, 'w') as f:
    json.dump({
        'total_unused': len(unused_ids),
        'unused_ids': sorted(list(unused_ids)),
        'sample_durations': {k: f"{v:.1f}min" for k, v in durations.items()}
    }, f, indent=2)
print(f"✓ Saved complete list to: unused_recording_ids.json\n")
