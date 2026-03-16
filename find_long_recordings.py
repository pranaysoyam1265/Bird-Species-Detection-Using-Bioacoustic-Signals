"""
Find all unused XC recordings longer than 5 minutes
"""
import os
import re
import csv
import subprocess
from pathlib import Path

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
                if 'recording_id' in row:
                    used_ids.add(row['recording_id'])

# Get all files
all_files = sorted([f for f in os.listdir(audio_dir) if f.startswith('XC') and (f.endswith('.mp3') or f.endswith('.wav'))])

print("🔍 Scanning all audio files for recordings >5 minutes...")
print("This may take a few minutes...\n")

long_recordings = []

for i, filename in enumerate(all_files):
    if (i + 1) % 500 == 0:
        print(f"  Scanned {i+1}/{len(all_files)} files...")
    
    filepath = os.path.join(audio_dir, filename)
    match = re.match(r'XC(\d+)', filename)
    if not match:
        continue
    
    xc_id = match.group(1)
    
    # Skip if used in training
    if xc_id in used_ids:
        continue
    
    try:
        # Get duration using ffprobe
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.stdout.strip():
            duration_sec = float(result.stdout.strip())
            duration_min = duration_sec / 60
            
            if duration_min > 5:  # Longer than 5 minutes
                long_recordings.append({
                    'filename': filename,
                    'xc_id': xc_id,
                    'duration_min': duration_min,
                    'filepath': filepath
                })
    except Exception:
        pass

print(f"\n{'='*70}")
print(f"RESULTS: Found {len(long_recordings)} unused recordings >5 minutes\n")

if long_recordings:
    long_recordings.sort(key=lambda x: x['duration_min'], reverse=True)
    print("Top 25 longest unused recordings (>5 min):\n")
    print(f"{'#':<3} {'XC ID':<10} {'Duration':<12} {'Filename':<40}")
    print("-" * 70)
    for i, rec in enumerate(long_recordings[:25], 1):
        print(f"{i:<3} {rec['xc_id']:<10} {rec['duration_min']:>6.1f} min  {rec['filename']:<40}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  • Total unused recordings >5 min: {len(long_recordings)}")
    print(f"  • ✓ You have MORE THAN ENOUGH for testing!")
    print(f"  • Average length: {sum(r['duration_min'] for r in long_recordings) / len(long_recordings):.1f} minutes")
    print(f"  • Longest: {long_recordings[0]['duration_min']:.1f} minutes")
    print(f"{'='*70}\n")
    
    # Save results
    with open(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\long_unused_recordings.txt", 'w') as f:
        f.write("Unused XC Recordings > 5 Minutes (Available for Testing)\n")
        f.write("=" * 70 + "\n\n")
        for i, rec in enumerate(long_recordings[:25], 1):
            f.write(f"{i}. {rec['filename']} - {rec['duration_min']:.1f} min (Path: {rec['filepath']})\n")
    
    print("✓ Saved detailed list to: long_unused_recordings.txt\n")
else:
    print("⚠ No recordings >5 minutes found")
    print("\nThis is normal for Xeno-Canto - most are short bird calls (<2 min)\n")
