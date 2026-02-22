"""
Script: prepare_testing_folder.py
Purpose: Copy 20 audio recordings (different species) to a Testing folder
Location: Save to 09_Utils/Scripts/prepare_testing_folder.py
"""

import os
import shutil
import pandas as pd
import random
import json

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"

# Source paths
TEST_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split", "test.csv")
TRAIN_CSV = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split", "train.csv")
AUDIO_CHUNKS_DIR = os.path.join(BASE_DIR, "02_Preprocessed", "Audio_Chunks")
STANDARDIZED_DIR = os.path.join(BASE_DIR, "02_Preprocessed", "Standardized_Audio")
RAW_AUDIO_DIR = os.path.join(BASE_DIR, "01_Raw_Data", "Audio_Recordings")
LABEL_MAPPING = os.path.join(BASE_DIR, "04_Labels", "Processed_Labels", "label_mapping.json")

# Destination
TESTING_DIR = os.path.join(BASE_DIR, "Testing")

# Settings
NUM_SPECIES = 30  # Increased to ensure at least 15 recordings are created
RANDOM_SEED = 42

# ============================================================
# SPECIES ENGLISH NAMES
# ============================================================
ENGLISH_NAMES = {
    "Amphispiza bilineata": "Black-throated Sparrow",
    "Anthus rubescens": "American Pipit",
    "Archilochus alexandri": "Black-chinned Hummingbird",
    "Artemisiospiza belli": "Bell's Sparrow",
    "Botaurus lentiginosus": "American Bittern",
    "Bucephala albeola": "Bufflehead",
    "Buteo platypterus": "Broad-winged Hawk",
    "Calidris bairdii": "Baird's Sandpiper",
    "Calypte anna": "Anna's Hummingbird",
    "Certhia americana": "Brown Creeper",
    "Chroicocephalus philadelphia": "Bonaparte's Gull",
    "Coccyzus erythropthalmus": "Black-billed Cuckoo",
    "Corvus brachyrhynchos": "American Crow",
    "Cyanocitta cristata": "Blue Jay",
    "Dolichonyx oryzivorus": "Bobolink",
    "Empidonax alnorum": "Alder Flycatcher",
    "Euphagus cyanocephalus": "Brewer's Blackbird",
    "Falco sparverius": "American Kestrel",
    "Haliaeetus leucocephalus": "Bald Eagle",
    "Hirundo rustica": "Barn Swallow",
    "Icterus bullockii": "Bullock's Oriole",
    "Icterus galbula": "Baltimore Oriole",
    "Mareca americana": "American Wigeon",
    "Megaceryle alcyon": "Belted Kingfisher",
    "Mniotilta varia": "Black-and-white Warbler",
    "Molothrus ater": "Brown-headed Cowbird",
    "Myiarchus cinerascens": "Ash-throated Flycatcher",
    "Passerina caerulea": "Blue Grosbeak",
    "Pheucticus melanocephalus": "Black-headed Grosbeak",
    "Pica hudsonia": "Black-billed Magpie",
    "Poecile atricapillus": "Black-capped Chickadee",
    "Polioptila caerulea": "Blue-gray Gnatcatcher",
    "Psaltriparus minimus": "Bushtit",
    "Recurvirostra americana": "American Avocet",
    "Riparia riparia": "Bank Swallow",
    "Sayornis nigricans": "Black Phoebe",
    "Scolopax minor": "American Woodcock",
    "Selasphorus platycercus": "Broad-tailed Hummingbird",
    "Setophaga caerulescens": "Black-throated Blue Warbler",
    "Setophaga fusca": "Blackburnian Warbler",
    "Setophaga nigrescens": "Black-throated Gray Warbler",
    "Setophaga ruticilla": "American Redstart",
    "Setophaga striata": "Blackpoll Warbler",
    "Setophaga virens": "Black-throated Green Warbler",
    "Spatula discors": "Blue-winged Teal",
    "Spinus tristis": "American Goldfinch",
    "Spizella breweri": "Brewer's Sparrow",
    "Spizelloides arborea": "American Tree Sparrow",
    "Strix varia": "Barred Owl",
    "Thryomanes bewickii": "Bewick's Wren",
    "Toxostoma rufum": "Brown Thrasher",
    "Turdus migratorius": "American Robin",
    "Vermivora cyanoptera": "Blue-winged Warbler",
    "Vireo solitarius": "Blue-headed Vireo",
}


def find_audio_file(filename_hint, species_name):
    """
    Search for the original audio file across multiple directories.
    The chunks are named like: XC123456_chunk_0.wav
    We want the ORIGINAL full recording: XC123456.wav
    """
    
    # Extract recording ID from chunk filename
    # Chunk format: XC123456_chunk_0.wav  OR  species_XC123456_chunk_0.wav
    parts = filename_hint.replace(".wav", "").replace(".mp3", "")
    
    # Try to extract XC ID
    xc_id = None
    for part in parts.split("_"):
        if part.startswith("XC") and part[2:].isdigit():
            xc_id = part
            break
    
    if not xc_id:
        # Try just using the first part
        xc_id = parts.split("_")[0]
    
    # Search locations (in priority order)
    search_dirs = [
        RAW_AUDIO_DIR,
        STANDARDIZED_DIR,
        AUDIO_CHUNKS_DIR,
    ]
    
    # Also check species-specific subdirectories
    for search_dir in search_dirs.copy():
        if os.path.exists(search_dir):
            for subdir in os.listdir(search_dir):
                full_subdir = os.path.join(search_dir, subdir)
                if os.path.isdir(full_subdir):
                    search_dirs.append(full_subdir)
    
    # Search for the file
    extensions = [".wav", ".mp3", ".ogg", ".flac"]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        
        for ext in extensions:
            # Try exact XC ID match
            candidate = os.path.join(search_dir, f"{xc_id}{ext}")
            if os.path.exists(candidate):
                return candidate
            
            # Try with species prefix
            safe_species = species_name.replace(" ", "_")
            candidate = os.path.join(search_dir, f"{safe_species}_{xc_id}{ext}")
            if os.path.exists(candidate):
                return candidate
    
    # Last resort: search chunk directory for any chunk of this recording
    if os.path.exists(AUDIO_CHUNKS_DIR):
        for root, dirs, files in os.walk(AUDIO_CHUNKS_DIR):
            for f in files:
                if xc_id and xc_id in f:
                    return os.path.join(root, f)
    
    return None


def main():
    print("=" * 65)
    print("🐦 BIRD DETECTION - TESTING FOLDER PREPARATION")
    print("=" * 65)
    
    random.seed(RANDOM_SEED)
    
    # ---------------------------------------------------------
    # Step 1: Load test.csv (or train.csv if you want training samples)
    # ---------------------------------------------------------
    # ⚡ CHANGE THIS TO TRAIN_CSV IF YOU WANT TRAINING SAMPLES
    source_csv = TEST_CSV  
    source_name = "TEST SET"
    
    if not os.path.exists(source_csv):
        print(f"\n❌ CSV not found: {source_csv}")
        print("   Trying train.csv instead...")
        source_csv = TRAIN_CSV
        source_name = "TRAIN SET"
    
    if not os.path.exists(source_csv):
        print(f"\n❌ Neither test.csv nor train.csv found!")
        print("   Let me scan for available CSVs...")
        
        # Scan for any CSV
        split_dir = os.path.join(BASE_DIR, "04_Labels", "Train_Val_Test_Split")
        if os.path.exists(split_dir):
            csvs = [f for f in os.listdir(split_dir) if f.endswith('.csv')]
            print(f"   Found: {csvs}")
        return
    
    print(f"\n📂 Loading {source_name}: {source_csv}")
    df = pd.read_csv(source_csv)
    print(f"   Total entries: {len(df)}")
    
    # ---------------------------------------------------------
    # Step 2: Examine CSV structure
    # ---------------------------------------------------------
    print(f"\n📋 CSV Columns: {list(df.columns)}")
    print(f"   First row: {df.iloc[0].to_dict()}")
    
    # Detect species column
    species_col = None
    for col in ['species', 'label', 'class', 'scientific_name', 'bird_species']:
        if col in df.columns:
            species_col = col
            break
    
    if species_col is None:
        print("\n❌ Could not find species column!")
        print(f"   Available columns: {list(df.columns)}")
        return
    
    # Detect filename column
    file_col = None
    for col in ['filename', 'file', 'filepath', 'path', 'chunk_file', 'audio_file']:
        if col in df.columns:
            file_col = col
            break
    
    if file_col is None:
        print("\n❌ Could not find filename column!")
        print(f"   Available columns: {list(df.columns)}")
        return
    
    print(f"\n   Species column: '{species_col}'")
    print(f"   Filename column: '{file_col}'")
    
    # ---------------------------------------------------------
    # Step 3: Get unique species and select 20
    # ---------------------------------------------------------
    all_species = df[species_col].unique()
    print(f"\n🐦 Total unique species in {source_name}: {len(all_species)}")
    
    # Select 20 diverse species (mix of common and rare)
    species_counts = df[species_col].value_counts()
    
    if len(all_species) < NUM_SPECIES:
        selected_species = list(all_species)
        print(f"   ⚠️ Only {len(all_species)} species available, using all")
    else:
        # Pick a diverse set: some common, some rare, some medium
        sorted_species = species_counts.index.tolist()
        
        # Strategy: pick evenly across the frequency distribution
        step = len(sorted_species) // NUM_SPECIES
        selected_indices = [i * step for i in range(NUM_SPECIES)]
        # Make sure we don't go out of bounds
        selected_indices = [min(i, len(sorted_species) - 1) for i in selected_indices]
        selected_species = [sorted_species[i] for i in selected_indices]
        
        # Ensure uniqueness
        selected_species = list(dict.fromkeys(selected_species))
        
        # If we need more, add random ones
        remaining = [s for s in all_species if s not in selected_species]
        while len(selected_species) < NUM_SPECIES and remaining:
            pick = random.choice(remaining)
            selected_species.append(pick)
            remaining.remove(pick)
    
    selected_species = selected_species[:NUM_SPECIES]
    
    print(f"   Selected {len(selected_species)} species for testing")
    
    # ---------------------------------------------------------
    # Step 4: For each species, pick the BEST chunk
    #         (one with most clear detection potential)
    # ---------------------------------------------------------
    print(f"\n🔍 Selecting best recording for each species...")
    
    selected_files = []
    
    for species in selected_species:
        species_df = df[df[species_col] == species]
        
        # Get unique recordings (by XC ID)
        species_df = species_df.copy()
        species_df['recording_id'] = species_df[file_col].apply(
            lambda x: x.split("_chunk_")[0] if "_chunk_" in str(x) else str(x).split(".")[0]
        )
        
        unique_recordings = species_df['recording_id'].unique()
        
        # Pick a recording that has the MOST chunks (likely longer/clearer)
        best_recording = species_df.groupby('recording_id').size().idxmax()
        
        # Get the first chunk of that recording (chunk_0 is start)
        recording_chunks = species_df[species_df['recording_id'] == best_recording]
        
        # Try to get chunk_0 for the cleanest start
        chunk_0 = recording_chunks[recording_chunks[file_col].str.contains("chunk_0", na=False)]
        if len(chunk_0) > 0:
            selected_file = chunk_0.iloc[0][file_col]
        else:
            selected_file = recording_chunks.iloc[0][file_col]
        
        english = ENGLISH_NAMES.get(species, "Unknown")
        num_chunks = len(recording_chunks)
        
        selected_files.append({
            'species': species,
            'english_name': english,
            'filename': selected_file,
            'recording_id': best_recording,
            'num_chunks': num_chunks,
            'total_available': len(species_df),
        })
    
    # ---------------------------------------------------------
    # Step 5: Create Testing folder and copy files
    # ---------------------------------------------------------
    os.makedirs(TESTING_DIR, exist_ok=True)
    print(f"\n📁 Testing folder: {TESTING_DIR}")
    
    # Clear existing files
    existing_files = [f for f in os.listdir(TESTING_DIR) if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
    if existing_files:
        print(f"   Clearing {len(existing_files)} existing files...")
        for f in existing_files:
            os.remove(os.path.join(TESTING_DIR, f))
    
    print(f"\n{'='*65}")
    print(f"{'#':<3} {'Species':<35} {'English Name':<28} {'Status'}")
    print(f"{'='*65}")
    
    copied_count = 0
    not_found = []
    manifest = []
    
    for i, entry in enumerate(selected_files, 1):
        species = entry['species']
        english = entry['english_name']
        filename = entry['filename']
        
        # Find the actual audio file
        source_path = find_audio_file(filename, species)
        
        # Also try direct path in chunks directory
        if source_path is None:
            direct_path = os.path.join(AUDIO_CHUNKS_DIR, filename)
            if os.path.exists(direct_path):
                source_path = direct_path
        
        # Try in species subdirectory
        if source_path is None:
            species_dir = os.path.join(AUDIO_CHUNKS_DIR, species.replace(" ", "_"))
            if os.path.exists(species_dir):
                chunk_path = os.path.join(species_dir, filename)
                if os.path.exists(chunk_path):
                    source_path = chunk_path
        
        # Try standardized audio directory
        if source_path is None:
            for ext in ['.wav', '.mp3']:
                std_path = os.path.join(STANDARDIZED_DIR, f"{entry['recording_id']}{ext}")
                if os.path.exists(std_path):
                    source_path = std_path
                    break
        
        if source_path and os.path.exists(source_path):
            # Create descriptive filename
            safe_english = english.replace(" ", "_").replace("'", "").replace("-", "_")
            safe_species = species.replace(" ", "_")
            ext = os.path.splitext(source_path)[1]
            
            dest_filename = f"{i:02d}_{safe_english}_{safe_species}_{entry['recording_id']}{ext}"
            dest_path = os.path.join(TESTING_DIR, dest_filename)
            
            shutil.copy2(source_path, dest_path)
            
            file_size = os.path.getsize(dest_path) / 1024  # KB
            
            print(f"{i:<3} {species:<35} {english:<28} ✅ ({file_size:.0f} KB)")
            copied_count += 1
            
            manifest.append({
                'index': i,
                'filename': dest_filename,
                'species_scientific': species,
                'species_english': english,
                'source_file': filename,
                'recording_id': entry['recording_id'],
                'file_size_kb': round(file_size, 1),
                'source': source_name,
            })
        else:
            print(f"{i:<3} {species:<35} {english:<28} ❌ NOT FOUND")
            not_found.append({
                'species': species,
                'english': english,
                'filename': filename,
            })
    
    # ---------------------------------------------------------
    # Step 6: Save manifest (what's in the Testing folder)
    # ---------------------------------------------------------
    manifest_path = os.path.join(TESTING_DIR, "testing_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({
            'description': 'Testing audio files for Bird Detection demo',
            'total_files': copied_count,
            'source': source_name,
            'species_count': len(set(m['species_scientific'] for m in manifest)),
            'files': manifest,
        }, f, indent=2, ensure_ascii=False)
    
    # Also save as CSV for easy viewing
    if manifest:
        manifest_csv_path = os.path.join(TESTING_DIR, "testing_manifest.csv")
        pd.DataFrame(manifest).to_csv(manifest_csv_path, index=False)
    
    # ---------------------------------------------------------
    # Step 7: Summary
    # ---------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"\n📊 SUMMARY")
    print(f"{'─'*40}")
    print(f"   ✅ Files copied:     {copied_count}/{NUM_SPECIES}")
    print(f"   ❌ Not found:        {len(not_found)}")
    print(f"   📁 Testing folder:   {TESTING_DIR}")
    print(f"   📋 Manifest:         {manifest_path}")
    
    if not_found:
        print(f"\n⚠️  FILES NOT FOUND:")
        for nf in not_found:
            print(f"   - {nf['english']} ({nf['species']})")
            print(f"     Searched for: {nf['filename']}")
        
        print(f"\n💡 TROUBLESHOOTING:")
        print(f"   Let me scan your actual directory structure...")
        
        # Scan what directories exist
        for check_dir in [AUDIO_CHUNKS_DIR, STANDARDIZED_DIR, RAW_AUDIO_DIR]:
            if os.path.exists(check_dir):
                contents = os.listdir(check_dir)[:5]
                print(f"\n   📂 {check_dir}")
                print(f"      Items ({len(os.listdir(check_dir))} total): {contents}...")
            else:
                print(f"\n   ❌ {check_dir} - DOES NOT EXIST")
    
    if copied_count > 0:
        print(f"\n✅ TESTING FOLDER READY!")
        print(f"\n🎯 TO TEST WITH STREAMLIT APP:")
        print(f"   1. Run the app:")
        print(f'      streamlit run "08_Deployment/Frontend/app_enhanced.py"')
        print(f"   2. Upload any file from: {TESTING_DIR}")
        print(f"   3. Check if the model correctly identifies the species!")
        
        print(f"\n📋 TESTING CHECKLIST:")
        for m in manifest[:5]:
            print(f"   □ Upload {m['filename']}")
            print(f"     Expected: {m['species_english']} ({m['species_scientific']})")


if __name__ == "__main__":
    main()