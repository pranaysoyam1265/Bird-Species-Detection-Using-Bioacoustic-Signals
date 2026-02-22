# save as: 09_Utils/Scripts/inventory_check.py

import os
import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")
AUDIO_DIR = PROJECT_ROOT / "01_Raw_Data" / "Audio_Recordings"
METADATA_FILE = PROJECT_ROOT / "01_Raw_Data" / "Metadata" / "bird_metadata_complete.csv"

def extract_xc_id(filename):
    """Extract XC ID from filename like 'XC123456.wav' or 'XC123456_Speciesname.wav'"""
    name = Path(filename).stem  # Remove extension
    # Handle various formats
    if name.startswith("XC"):
        # Extract just the number part
        xc_part = name.split("_")[0] if "_" in name else name
        xc_id = xc_part.replace("XC", "")
        try:
            return int(xc_id)
        except ValueError:
            return None
    return None

def main():
    print("=" * 60)
    print("📁 AUDIO FILE INVENTORY CHECK")
    print("=" * 60)
    
    # 1. Get all audio files
    audio_extensions = {'.wav', '.mp3', '.ogg', '.flac'}
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(AUDIO_DIR.glob(f"**/*{ext}"))
    
    print(f"\n📂 Audio directory: {AUDIO_DIR}")
    print(f"🎵 Total audio files found: {len(audio_files)}")
    
    # 2. Extract XC IDs from filenames
    file_data = []
    for f in audio_files:
        xc_id = extract_xc_id(f.name)
        file_data.append({
            'filename': f.name,
            'filepath': str(f),
            'xc_id': xc_id,
            'size_mb': f.stat().st_size / (1024 * 1024)
        })
    
    files_df = pd.DataFrame(file_data)
    files_with_id = files_df[files_df['xc_id'].notna()]
    files_without_id = files_df[files_df['xc_id'].isna()]
    
    print(f"✅ Files with XC ID: {len(files_with_id)}")
    print(f"❓ Files without XC ID: {len(files_without_id)}")
    
    # 3. Load existing metadata
    if METADATA_FILE.exists():
        metadata_df = pd.read_csv(METADATA_FILE)
        print(f"\n📋 Metadata file: {METADATA_FILE.name}")
        print(f"📊 Recordings with metadata: {len(metadata_df)}")
        
        # Get XC IDs from metadata
        if 'xc_id' in metadata_df.columns:
            metadata_ids = set(pd.to_numeric(metadata_df['xc_id'], errors='coerce').dropna().astype(int))
        elif 'id' in metadata_df.columns:
            metadata_ids = set(pd.to_numeric(metadata_df['id'], errors='coerce').dropna().astype(int))
        elif 'recording_id' in metadata_df.columns:
            metadata_ids = set(pd.to_numeric(metadata_df['recording_id'], errors='coerce').dropna().astype(int))
        else:
            print("⚠️ Could not find ID column in metadata!")
            print(f"   Available columns: {list(metadata_df.columns)}")
            metadata_ids = set()
        
        # 4. Find missing
        file_ids = set(pd.to_numeric(files_with_id['xc_id'], errors='coerce').dropna().astype(int))
        
        missing_metadata_ids = file_ids - metadata_ids
        missing_files_ids = metadata_ids - file_ids
        
        print(f"\n{'=' * 60}")
        print("📊 ANALYSIS RESULTS")
        print("=" * 60)
        print(f"🎵 Audio files with XC ID:     {len(file_ids)}")
        print(f"📋 IDs in metadata:            {len(metadata_ids)}")
        print(f"❌ Files MISSING metadata:     {len(missing_metadata_ids)}")
        print(f"❌ Metadata MISSING files:     {len(missing_files_ids)}")
        
        # 5. Save missing IDs for scraping
        if missing_metadata_ids:
            missing_df = files_with_id[
                files_with_id['xc_id'].astype(int).isin(missing_metadata_ids)
            ]
            
            output_file = PROJECT_ROOT / "01_Raw_Data" / "Metadata" / "missing_metadata_ids.csv"
            missing_df.to_csv(output_file, index=False)
            print(f"\n💾 Saved missing IDs to: {output_file}")
            print(f"   Ready to scrape {len(missing_metadata_ids)} recordings!")
            
            # Show sample
            print(f"\n📝 Sample missing XC IDs (first 10):")
            sample_ids = sorted(list(missing_metadata_ids))[:10]
            for xc_id in sample_ids:
                print(f"   - XC{xc_id}")
    else:
        print(f"\n⚠️ Metadata file not found: {METADATA_FILE}")
    
    # 6. Summary
    print(f"\n{'=' * 60}")
    print("📋 NEXT STEPS")
    print("=" * 60)
    print("1. Run metadata scraper on missing IDs")
    print("2. Download more data for rare species")
    print("3. Preprocess new audio files")
    print("=" * 60)

if __name__ == "__main__":
    main()