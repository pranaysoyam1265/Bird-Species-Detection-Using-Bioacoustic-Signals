"""
metadata_diagnostic_v2.py
Detailed diagnostic showing WHICH recording IDs have issues

Outputs:
- Console summary
- CSV of all problematic recording IDs
- Separate lists for each issue type
"""

import pandas as pd
from pathlib import Path
import os

# ============================================================
# CONFIGURATION - Update if needed
# ============================================================

BASE_DIR = Path(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")
METADATA_PATH = BASE_DIR / "01_Raw_Data" / "Metadata" / "bird_metadata_complete.csv"
AUDIO_DIR = BASE_DIR / "01_Raw_Data" / "Audio_Recordings"
OUTPUT_DIR = BASE_DIR / "01_Raw_Data" / "Metadata"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_missing(value):
    """Check if a value is missing/empty"""
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() in ['', 'None', 'null', 'nan', 'NaN']:
        return True
    return False

def get_numeric_id(recording_id):
    """Extract numeric ID from various formats"""
    id_str = str(recording_id)
    if id_str.upper().startswith('XC'):
        return id_str[2:]
    return id_str

def check_audio_exists(recording_id, audio_dir):
    """Check if audio file exists for this recording ID"""
    numeric_id = get_numeric_id(recording_id)
    file_path = audio_dir / f"XC{numeric_id}.wav"
    return file_path.exists()

# ============================================================
# MAIN DIAGNOSTIC
# ============================================================

def main():
    print("=" * 70)
    print("📊 DETAILED METADATA DIAGNOSTIC - BY RECORDING ID")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading: {METADATA_PATH}")
    
    if not METADATA_PATH.exists():
        print(f"❌ ERROR: File not found: {METADATA_PATH}")
        print("\nAvailable files in Metadata folder:")
        metadata_dir = BASE_DIR / "01_Raw_Data" / "Metadata"
        if metadata_dir.exists():
            for f in metadata_dir.iterdir():
                print(f"  - {f.name}")
        return
    
    df = pd.read_csv(METADATA_PATH)
    total_rows = len(df)
    
    print(f"✅ Loaded {total_rows} rows")
    print(f"Columns: {list(df.columns)}")
    
    # --------------------------------------------------------
    # ISSUE TRACKING - Create columns for each issue type
    # --------------------------------------------------------
    
    print("\n" + "-" * 70)
    print("🔍 CHECKING EACH RECORDING ID FOR ISSUES...")
    print("-" * 70)
    
    # Issue 1: Missing scientific_name (CRITICAL - can't train without this)
    df['issue_no_species'] = df['scientific_name'].apply(is_missing)
    
    # Issue 2: Missing english_name
    df['issue_no_english_name'] = df['english_name'].apply(is_missing)
    
    # Issue 3: Audio file doesn't exist (CRITICAL)
    print("Checking audio file existence (may take a moment)...")
    df['issue_no_audio_file'] = ~df['recording_id'].apply(
        lambda x: check_audio_exists(x, AUDIO_DIR)
    )
    
    # Issue 4: Missing duration
    df['issue_no_duration'] = df['duration_seconds'].apply(is_missing) | (df['duration_seconds'] == 0)
    
    # Issue 5: Missing date
    df['issue_no_date'] = df['date'].apply(is_missing) if 'date' in df.columns else True
    
    # Issue 6: Missing time
    df['issue_no_time'] = df['time'].apply(is_missing) if 'time' in df.columns else True
    
    # Issue 7: Missing country
    df['issue_no_country'] = df['country'].apply(is_missing) if 'country' in df.columns else True
    
    # Issue 8: Missing coordinates
    df['issue_no_latitude'] = df['latitude'].apply(is_missing) if 'latitude' in df.columns else True
    df['issue_no_longitude'] = df['longitude'].apply(is_missing) if 'longitude' in df.columns else True
    df['issue_no_coordinates'] = df['issue_no_latitude'] | df['issue_no_longitude']
    
    # Issue 9: Missing quality
    df['issue_no_quality'] = df['quality'].apply(is_missing) if 'quality' in df.columns else True
    
    # Issue 10: Scraping failed (status != success)
    if 'status' in df.columns:
        df['issue_scrape_failed'] = df['status'] != 'success'
    else:
        df['issue_scrape_failed'] = False
    
    # Combined: Has ANY critical issue (no species OR no audio)
    df['has_critical_issue'] = df['issue_no_species'] | df['issue_no_audio_file']
    
    # Combined: Has ANY issue at all
    issue_columns = [col for col in df.columns if col.startswith('issue_')]
    df['has_any_issue'] = df[issue_columns].any(axis=1)
    
    # Count issues per recording
    df['issue_count'] = df[issue_columns].sum(axis=1)
    
    # --------------------------------------------------------
    # SUMMARY REPORT
    # --------------------------------------------------------
    
    print("\n" + "=" * 70)
    print("📋 ISSUE SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Issue Type':<40} {'Count':<10} {'Percentage':<10}")
    print("-" * 60)
    
    issues_summary = {
        'Missing species (scientific_name)': df['issue_no_species'].sum(),
        'Missing english_name': df['issue_no_english_name'].sum(),
        'Audio file not found': df['issue_no_audio_file'].sum(),
        'Missing/zero duration': df['issue_no_duration'].sum(),
        'Missing date': df['issue_no_date'].sum(),
        'Missing time': df['issue_no_time'].sum(),
        'Missing country': df['issue_no_country'].sum(),
        'Missing coordinates (lat/lon)': df['issue_no_coordinates'].sum(),
        'Missing quality': df['issue_no_quality'].sum(),
        'Scraping failed': df['issue_scrape_failed'].sum(),
    }
    
    for issue_name, count in issues_summary.items():
        pct = 100 * count / total_rows
        status = "🔴" if pct > 50 else "🟡" if pct > 20 else "🟢"
        print(f"{status} {issue_name:<38} {count:<10} {pct:.1f}%")
    
    # --------------------------------------------------------
    # CRITICAL ISSUES (What affects ML training)
    # --------------------------------------------------------
    
    print("\n" + "=" * 70)
    print("🚨 CRITICAL ISSUES (Affects ML Training)")
    print("=" * 70)
    
    critical_count = df['has_critical_issue'].sum()
    usable_count = total_rows - critical_count
    
    print(f"""
    Total recordings:              {total_rows}
    
    🔴 UNUSABLE (no species OR no audio): {critical_count} ({100*critical_count/total_rows:.1f}%)
    🟢 USABLE (has species AND audio):    {usable_count} ({100*usable_count/total_rows:.1f}%)
    
    ➡️  You can train on {usable_count} recordings
    """)
    
    # --------------------------------------------------------
    # LIST OF PROBLEMATIC RECORDING IDs
    # --------------------------------------------------------
    
    print("\n" + "=" * 70)
    print("📝 RECORDING IDs WITH ISSUES")
    print("=" * 70)
    
    # IDs missing species
    ids_no_species = df.loc[df['issue_no_species'], 'recording_id'].tolist()
    print(f"\n🔴 Recording IDs missing species ({len(ids_no_species)} total):")
    if len(ids_no_species) <= 20:
        print(f"   {ids_no_species}")
    else:
        print(f"   First 20: {ids_no_species[:20]}")
        print(f"   ... and {len(ids_no_species) - 20} more")
    
    # IDs missing audio
    ids_no_audio = df.loc[df['issue_no_audio_file'], 'recording_id'].tolist()
    print(f"\n🔴 Recording IDs with no audio file ({len(ids_no_audio)} total):")
    if len(ids_no_audio) <= 20:
        print(f"   {ids_no_audio}")
    else:
        print(f"   First 20: {ids_no_audio[:20]}")
        print(f"   ... and {len(ids_no_audio) - 20} more")
    
    # IDs with scraping failures
    ids_scrape_failed = df.loc[df['issue_scrape_failed'], 'recording_id'].tolist()
    print(f"\n🟡 Recording IDs where scraping failed ({len(ids_scrape_failed)} total):")
    if len(ids_scrape_failed) <= 20:
        print(f"   {ids_scrape_failed}")
    else:
        print(f"   First 20: {ids_scrape_failed[:20]}")
        print(f"   ... and {len(ids_scrape_failed) - 20} more")
    
    # --------------------------------------------------------
    # USABLE DATA ANALYSIS
    # --------------------------------------------------------
    
    print("\n" + "=" * 70)
    print("🐦 USABLE DATA ANALYSIS")
    print("=" * 70)
    
    # Filter to usable only
    df_usable = df[~df['has_critical_issue']].copy()
    
    if len(df_usable) > 0:
        unique_species = df_usable['scientific_name'].nunique()
        print(f"\nUsable recordings: {len(df_usable)}")
        print(f"Unique species: {unique_species}")
        
        print("\n📊 Top 15 species by recording count:")
        print("-" * 50)
        species_counts = df_usable['scientific_name'].value_counts()
        for i, (species, count) in enumerate(species_counts.head(15).items()):
            print(f"  {i+1:2}. {species:<40} {count:>5} recordings")
        
        print("\n📊 Species with FEWEST recordings (may need removal):")
        print("-" * 50)
        rare_species = species_counts[species_counts < 5]
        print(f"  Species with < 5 recordings: {len(rare_species)}")
        if len(rare_species) > 0:
            for species, count in rare_species.head(10).items():
                print(f"    - {species}: {count} recordings")
        
        # Class imbalance
        print("\n📊 Class imbalance check:")
        print("-" * 50)
        max_count = species_counts.max()
        min_count = species_counts.min()
        mean_count = species_counts.mean()
        print(f"  Max recordings per species: {max_count}")
        print(f"  Min recordings per species: {min_count}")
        print(f"  Mean recordings per species: {mean_count:.1f}")
        print(f"  Imbalance ratio (max/min): {max_count/min_count:.1f}x")
    else:
        print("\n⚠️ No usable recordings found!")
    
    # --------------------------------------------------------
    # SAVE DETAILED REPORTS
    # --------------------------------------------------------
    
    print("\n" + "=" * 70)
    print("💾 SAVING DETAILED REPORTS")
    print("=" * 70)
    
    # 1. Save full diagnostic with issue flags
    diagnostic_path = OUTPUT_DIR / "metadata_with_issues.csv"
    df.to_csv(diagnostic_path, index=False)
    print(f"✅ Full data with issue flags: {diagnostic_path}")
    
    # 2. Save list of problematic IDs
    problems_df = df[df['has_any_issue']][['recording_id', 'scientific_name', 'status', 'issue_count'] + issue_columns]
    problems_path = OUTPUT_DIR / "problematic_recording_ids.csv"
    problems_df.to_csv(problems_path, index=False)
    print(f"✅ Problematic IDs list: {problems_path}")
    
    # 3. Save usable data only
    usable_path = OUTPUT_DIR / "metadata_usable_only.csv"
    df_usable_clean = df[~df['has_critical_issue']].drop(columns=[col for col in df.columns if col.startswith('issue_') or col.startswith('has_')])
    df_usable_clean.to_csv(usable_path, index=False)
    print(f"✅ Usable data only: {usable_path}")
    
    # 4. Save species summary
    if len(df_usable) > 0:
        species_summary = df_usable.groupby('scientific_name').agg({
            'recording_id': 'count',
            'english_name': 'first',
            'duration_seconds': 'sum'
        }).rename(columns={'recording_id': 'num_recordings', 'duration_seconds': 'total_duration_sec'})
        species_summary = species_summary.sort_values('num_recordings', ascending=False)
        species_path = OUTPUT_DIR / "species_summary.csv"
        species_summary.to_csv(species_path)
        print(f"✅ Species summary: {species_path}")
    
    # --------------------------------------------------------
    # FINAL RECOMMENDATION
    # --------------------------------------------------------
    
    print("\n" + "=" * 70)
    print("🎯 RECOMMENDATION")
    print("=" * 70)
    
    if usable_count >= 1000:
        print(f"""
    ✅ GOOD TO PROCEED!
    
    You have {usable_count} usable recordings.
    This is enough for training a bird detection model.
    
    Next step: Run the Label Preparation script to:
    1. Map species to audio chunks
    2. Create train/val/test splits
    3. Handle class imbalance
        """)
    elif usable_count >= 500:
        print(f"""
    ⚠️ PROCEED WITH CAUTION
    
    You have {usable_count} usable recordings.
    This is minimal but workable. Consider:
    - Using data augmentation
    - Reducing number of species (remove rare ones)
    - Using transfer learning
        """)
    else:
        print(f"""
    🔴 INSUFFICIENT DATA
    
    You have only {usable_count} usable recordings.
    Options:
    - Re-run scraper for failed IDs
    - Download more recordings
    - Use pretrained model (BirdNET)
        """)
    
    print("\n" + "=" * 70)
    print("✅ DIAGNOSTIC COMPLETE")
    print("=" * 70)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()