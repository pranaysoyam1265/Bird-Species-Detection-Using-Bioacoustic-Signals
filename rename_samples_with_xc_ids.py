"""
Rename audio samples with their source XC (Xeno-Canto) IDs
Format: XC_id1_id2_id3_...idN.wav
"""

import json
import os
from pathlib import Path

# Load metadata
metadata_path = "10_Outputs/Test_Samples_Long/test_samples_metadata.json"
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

samples_dir = Path("10_Outputs/Test_Samples_Long")
renamed_count = 0
rename_mapping = []

print("🎵 Renaming audio samples with XC IDs...\n")
print("=" * 70)

for sample in metadata:
    sample_id = sample['sample_id']
    old_filename = sample['output_file']
    old_path = samples_dir / old_filename
    
    # Extract XC IDs from source recordings
    xc_ids = [rec['xc_id'] for rec in sample['source_recordings']]
    xc_ids_str = '_'.join(xc_ids)
    
    # Create new filename
    new_filename = f"XC_{xc_ids_str}.wav"
    new_path = samples_dir / new_filename
    
    # Rename file
    if old_path.exists():
        try:
            old_path.rename(new_path)
            renamed_count += 1
            
            rename_mapping.append({
                'sample_id': sample_id,
                'old_name': old_filename,
                'new_name': new_filename,
                'xc_ids': xc_ids,
                'num_xc_ids': len(xc_ids),
                'status': 'SUCCESS'
            })
            
            print(f"✓ Sample {sample_id:2d}: {old_filename}")
            print(f"            → {new_filename}")
            print()
        except Exception as e:
            print(f"✗ Sample {sample_id:2d}: ERROR - {str(e)}")
            rename_mapping.append({
                'sample_id': sample_id,
                'old_name': old_filename,
                'status': 'FAILED',
                'error': str(e)
            })
    else:
        print(f"✗ Sample {sample_id:2d}: FILE NOT FOUND - {old_filename}")
        rename_mapping.append({
            'sample_id': sample_id,
            'old_name': old_filename,
            'status': 'FILE_NOT_FOUND'
        })

print("=" * 70)
print(f"\n✅ Successfully renamed: {renamed_count}/20 samples\n")

# Save rename mapping
mapping_path = samples_dir / "rename_mapping.json"
with open(mapping_path, 'w') as f:
    json.dump(rename_mapping, f, indent=2)
print(f"📄 Rename mapping saved to: {mapping_path}")

# List renamed files
print("\n📋 New filenames:")
print("=" * 70)
for item in rename_mapping:
    if item['status'] == 'SUCCESS':
        print(f"Sample {item['sample_id']:2d}: {item['new_name']}")

print("\n" + "=" * 70)
print("✨ Renaming complete!")
