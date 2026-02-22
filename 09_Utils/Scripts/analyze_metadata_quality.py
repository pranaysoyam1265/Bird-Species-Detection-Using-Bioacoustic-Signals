# save as: 09_Utils/Scripts/analyze_metadata_quality.py

import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")
METADATA_FILE = PROJECT_ROOT / "01_Raw_Data" / "Metadata" / "bird_metadata_complete.csv"

def main():
    print("=" * 70)
    print("📊 METADATA QUALITY ANALYSIS")
    print("=" * 70)
    
    # Load metadata
    df = pd.read_csv(METADATA_FILE)
    
    print(f"\n📋 Total rows in CSV: {len(df)}")
    print(f"\n📝 Columns in CSV:")
    print("-" * 70)
    
    for col in df.columns:
        non_null = df[col].notna().sum()
        non_empty = df[col].astype(str).str.strip().ne('').sum()
        pct = (non_null / len(df)) * 100
        print(f"   {col:30} → {non_null:5} non-null ({pct:5.1f}%) | {non_empty} non-empty")
    
    # Find the species column
    print("\n" + "=" * 70)
    print("🐦 SPECIES COLUMN ANALYSIS")
    print("=" * 70)
    
    # Try common column names for species
    species_cols = ['species', 'Species', 'scientific_name', 'en', 'english_name', 
                    'common_name', 'gen', 'sp', 'bird_species']
    
    species_col = None
    for col in species_cols:
        if col in df.columns:
            species_col = col
            break
    
    if species_col:
        print(f"\n✅ Found species column: '{species_col}'")
        
        # Count valid species entries
        valid_species = df[species_col].notna() & (df[species_col].astype(str).str.strip() != '')
        
        print(f"   Total rows:          {len(df)}")
        print(f"   With species info:   {valid_species.sum()}")
        print(f"   Missing species:     {(~valid_species).sum()}")
        
        # Show sample of valid species
        print(f"\n📝 Sample species (first 10 unique):")
        unique_species = df.loc[valid_species, species_col].unique()[:10]
        for sp in unique_species:
            print(f"   - {sp}")
        
        # Show sample of missing rows
        missing_rows = df[~valid_species]
        if len(missing_rows) > 0:
            print(f"\n❌ Sample rows with missing species (first 5):")
            print(missing_rows.head().to_string())
    else:
        print(f"\n⚠️ Could not find species column!")
        print(f"   Available columns: {list(df.columns)}")
    
    # Identify complete vs incomplete rows
    print("\n" + "=" * 70)
    print("📊 ROW COMPLETENESS ANALYSIS")
    print("=" * 70)
    
    # Define required columns (adjust based on your CSV structure)
    # We'll check what columns exist and which are important
    
    # Count rows where ALL columns have values
    complete_rows = df.dropna().shape[0]
    
    # Count rows where CRITICAL columns have values
    # (assuming species/scientific name is critical)
    if species_col:
        critical_complete = valid_species.sum()
    else:
        critical_complete = 0
    
    print(f"\n   Completely filled rows (all columns): {complete_rows}")
    print(f"   Rows with species info:               {critical_complete}")
    print(f"   Rows missing species:                 {len(df) - critical_complete}")
    
    # Separate good and bad data
    print("\n" + "=" * 70)
    print("💾 SEPARATING GOOD vs INCOMPLETE DATA")
    print("=" * 70)
    
    if species_col:
        good_data = df[valid_species].copy()
        incomplete_data = df[~valid_species].copy()
        
        # Save separated files
        good_file = PROJECT_ROOT / "01_Raw_Data" / "Metadata" / "metadata_complete_valid.csv"
        incomplete_file = PROJECT_ROOT / "01_Raw_Data" / "Metadata" / "metadata_incomplete.csv"
        
        good_data.to_csv(good_file, index=False)
        incomplete_data.to_csv(incomplete_file, index=False)
        
        print(f"\n✅ Saved {len(good_data)} complete records to:")
        print(f"   {good_file}")
        print(f"\n❌ Saved {len(incomplete_data)} incomplete records to:")
        print(f"   {incomplete_file}")
        
        # Check for XC IDs in incomplete data
        id_cols = ['xc_id', 'id', 'recording_id', 'XC_ID', 'ID']
        id_col = None
        for col in id_cols:
            if col in incomplete_data.columns:
                id_col = col
                break
        
        if id_col:
            incomplete_ids = incomplete_data[id_col].dropna().astype(int).tolist()
            print(f"\n📝 XC IDs needing metadata scraping: {len(incomplete_ids)}")
            
            # Save IDs for scraping
            ids_file = PROJECT_ROOT / "01_Raw_Data" / "Metadata" / "ids_to_scrape.txt"
            with open(ids_file, 'w') as f:
                for xc_id in incomplete_ids:
                    f.write(f"{xc_id}\n")
            print(f"💾 Saved IDs to: {ids_file}")
    
    # Also find audio files not in metadata at all
    print("\n" + "=" * 70)
    print("🎵 AUDIO FILES NOT IN METADATA")
    print("=" * 70)
    
    AUDIO_DIR = PROJECT_ROOT / "01_Raw_Data" / "Audio_Recordings"
    audio_files = list(AUDIO_DIR.glob("**/*.wav")) + list(AUDIO_DIR.glob("**/*.mp3"))
    
    # Extract XC IDs from audio filenames
    audio_ids = set()
    for f in audio_files:
        name = f.stem
        if name.startswith("XC"):
            try:
                xc_id = int(name.split("_")[0].replace("XC", ""))
                audio_ids.add(xc_id)
            except:
                pass
    
    # Get IDs from metadata
    if id_col and id_col in df.columns:
        metadata_ids = set(df[id_col].dropna().astype(int))
    else:
        metadata_ids = set()
    
    missing_from_metadata = audio_ids - metadata_ids
    
    print(f"\n   Audio files:              {len(audio_ids)} unique XC IDs")
    print(f"   In metadata CSV:          {len(metadata_ids)} IDs")
    print(f"   Missing from CSV:         {len(missing_from_metadata)} IDs")
    
    if missing_from_metadata:
        # Save these IDs too
        missing_file = PROJECT_ROOT / "01_Raw_Data" / "Metadata" / "ids_not_in_csv.txt"
        with open(missing_file, 'w') as f:
            for xc_id in sorted(missing_from_metadata):
                f.write(f"{xc_id}\n")
        print(f"💾 Saved missing IDs to: {missing_file}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    total_to_scrape = len(missing_from_metadata) + (len(df) - critical_complete if species_col else 0)
    print(f"""
    Total audio files:           4,521
    With complete metadata:      {critical_complete if species_col else 'Unknown'}
    Needing metadata scraping:   ~{total_to_scrape}
    
    NEXT STEP: Run Xeno-Canto API scraper on these IDs
    """)

if __name__ == "__main__":
    main()