"""
FIXED: Enhanced Xeno-Canto Scraper for Maximum Dataset
- Multi-threaded downloading
- Quality filtering (A/B ratings only)
- Targeted species collection
- Progress tracking with resume capability
- FIXED: Length parsing for MM:SS format
"""

import os
import requests
import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

import os

API_KEY = os.getenv("XENO_CANTO_API_KEY", "your_api_key_here")
BASE_URL = "https://xeno-canto.org/api/3/recordings"

# Paths
PROJECT_ROOT = Path(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")
AUDIO_DIR = PROJECT_ROOT / "01_Raw_Data" / "Audio_Recordings"
METADATA_DIR = PROJECT_ROOT / "01_Raw_Data" / "Metadata"
CHECKPOINT_FILE = METADATA_DIR / "aggressive_scrape_checkpoint.json"

# Download settings
MAX_WORKERS = 8  # Parallel downloads
QUALITY_THRESHOLD = ['A', 'B']  # Only high-quality recordings
MIN_LENGTH = 5  # Minimum recording length in seconds
MAX_RECORDINGS_PER_SPECIES = 500  # Per species limit

# ============================================================
# PRIORITY SPECIES (Need more data)
# ============================================================

PRIORITY_SPECIES = [
    "Bucephala albeola",  # Bufflehead
    "Mareca americana",  # American Wigeon
    "Recurvirostra americana",  # American Avocet
    "Calidris bairdii",  # Baird's Sandpiper
    "Spatula discors",  # Blue-winged Teal
]

# ============================================================
# NEW SPECIES TO ADD (Expand to 100-150 total)
# ============================================================

NEW_SPECIES = [
    # North American Species
    "Turdus migratorius",  # American Robin
    "Cardinalis cardinalis",  # Northern Cardinal
    "Cyanocitta cristata",  # Blue Jay
    "Corvus brachyrhynchos",  # American Crow
    "Zenaida macroura",  # Mourning Dove
    "Melanerpes carolinus",  # Red-bellied Woodpecker
    "Sitta carolinensis",  # White-breasted Nuthatch
    "Poecile atricapillus",  # Black-capped Chickadee
    "Baeolophus bicolor",  # Tufted Titmouse
    "Thryothorus ludovicianus",  # Carolina Wren
    
    # Raptors
    "Buteo jamaicensis",  # Red-tailed Hawk
    "Accipiter cooperii",  # Cooper's Hawk
    "Falco sparverius",  # American Kestrel
    "Haliaeetus leucocephalus",  # Bald Eagle
    "Bubo virginianus",  # Great Horned Owl
    
    # Waterfowl
    "Anas platyrhynchos",  # Mallard
    "Branta canadensis",  # Canada Goose
    "Aix sponsa",  # Wood Duck
    "Mergus merganser",  # Common Merganser
    "Chen caerulescens",  # Snow Goose
    
    # Warblers
    "Setophaga petechia",  # Yellow Warbler
    "Setophaga coronata",  # Yellow-rumped Warbler
    "Setophaga americana",  # Northern Parula
    "Geothlypis trichas",  # Common Yellowthroat
    "Cardellina pusilla",  # Wilson's Warbler
    
    # Sparrows
    "Melospiza melodia",  # Song Sparrow
    "Zonotrichia albicollis",  # White-throated Sparrow
    "Junco hyemalis",  # Dark-eyed Junco
    "Passerella iliaca",  # Fox Sparrow
    "Spizella passerina",  # Chipping Sparrow
    
    # Thrushes
    "Catharus guttatus",  # Hermit Thrush
    "Catharus ustulatus",  # Swainson's Thrush
    "Sialia sialis",  # Eastern Bluebird
    "Myadestes townsendi",  # Townsend's Solitaire
    
    # Flycatchers
    "Sayornis phoebe",  # Eastern Phoebe
    "Contopus virens",  # Eastern Wood-Pewee
    "Empidonax alnorum",  # Alder Flycatcher
    "Tyrannus tyrannus",  # Eastern Kingbird
    
    # Woodpeckers
    "Dryobates pubescens",  # Downy Woodpecker
    "Dryobates villosus",  # Hairy Woodpecker
    "Colaptes auratus",  # Northern Flicker
    "Dryocopus pileatus",  # Pileated Woodpecker
]

# ============================================================
# FUNCTIONS
# ============================================================

def load_checkpoint():
    """Load scraping progress"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {
        'completed_species': [],
        'failed_species': [],
        'total_downloaded': 0,
        'metadata': []
    }

def save_checkpoint(checkpoint):
    """Save scraping progress"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def parse_length_to_seconds(length_str):
    """Convert MM:SS or HH:MM:SS format to seconds"""
    try:
        if ':' in str(length_str):
            parts = str(length_str).split(':')
            if len(parts) == 2:  # MM:SS
                minutes, seconds = parts
                return int(minutes) * 60 + int(seconds)
            elif len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        # If no colon, assume it's already in seconds
        return float(length_str)
    except Exception as e:
        print(f"    ⚠️ Could not parse length '{length_str}': {e}")
        return 0

def search_species(species_scientific, max_results=500):
    """Search Xeno-Canto for a species"""
    query = f"gen:{species_scientific.split()[0]} sp:{species_scientific.split()[1]}"
    
    params = {
        'query': query,
        'key': API_KEY
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            recordings = data.get('recordings', [])
            
            # Filter by quality and length
            filtered = []
            for r in recordings:
                # Check quality
                if r.get('q') not in QUALITY_THRESHOLD:
                    continue
                
                # Parse and check length
                length_seconds = parse_length_to_seconds(r.get('length', '0'))
                if length_seconds >= MIN_LENGTH:
                    filtered.append(r)
            
            return filtered[:max_results]
        else:
            print(f"  ⚠️ HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return []

def download_recording(recording_info):
    """Download a single recording"""
    try:
        xc_id = recording_info['id']
        filename = f"XC{xc_id}.mp3"
        filepath = AUDIO_DIR / filename
        
        # Skip if already exists
        if filepath.exists():
            return {'success': True, 'xc_id': xc_id, 'status': 'exists'}
        
        # Download
        download_url = f"https://xeno-canto.org/{xc_id}/download"
        response = requests.get(download_url, timeout=60, stream=True)
        
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return {'success': True, 'xc_id': xc_id, 'status': 'downloaded'}
        else:
            return {'success': False, 'xc_id': xc_id, 'error': f'HTTP {response.status_code}'}
            
    except Exception as e:
        return {'success': False, 'xc_id': recording_info['id'], 'error': str(e)}

def download_species_parallel(species_scientific, recordings):
    """Download all recordings for a species in parallel"""
    print(f"\n📥 Downloading {len(recordings)} recordings for {species_scientific}")
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_recording = {
            executor.submit(download_recording, rec): rec 
            for rec in recordings
        }
        
        with tqdm(total=len(recordings), desc=f"  {species_scientific[:30]}") as pbar:
            for future in as_completed(future_to_recording):
                result = future.result()
                results.append(result)
                pbar.update(1)
    
    success_count = sum(1 for r in results if r['success'])
    print(f"  ✅ {success_count}/{len(recordings)} downloaded successfully")
    
    return results

def scrape_all_species(target_species_list, checkpoint):
    """Main scraping function"""
    
    print("=" * 70)
    print("🚀 AGGRESSIVE XENO-CANTO SCRAPER (FIXED)")
    print("=" * 70)
    print(f"Target species: {len(target_species_list)}")
    print(f"Already completed: {len(checkpoint['completed_species'])}")
    print(f"Max per species: {MAX_RECORDINGS_PER_SPECIES}")
    print(f"Quality filter: {QUALITY_THRESHOLD}")
    print(f"Min length: {MIN_LENGTH} seconds")
    print(f"Parallel workers: {MAX_WORKERS}")
    print("=" * 70)
    
    for i, species in enumerate(target_species_list, 1):
        # Skip if already done
        if species in checkpoint['completed_species']:
            print(f"\n[{i}/{len(target_species_list)}] ⏭️ Skipping {species} (already done)")
            continue
        
        print(f"\n[{i}/{len(target_species_list)}] 🔍 Searching: {species}")
        
        # Search for recordings
        recordings = search_species(species, MAX_RECORDINGS_PER_SPECIES)
        
        if not recordings:
            print(f"  ⚠️ No high-quality recordings found (≥{MIN_LENGTH}s)")
            checkpoint['failed_species'].append(species)
            save_checkpoint(checkpoint)
            continue
        
        print(f"  📊 Found {len(recordings)} high-quality recordings")
        
        # Download in parallel
        results = download_species_parallel(species, recordings)
        
        # Update metadata
        for rec in recordings:
            checkpoint['metadata'].append({
                'xc_id': rec['id'],
                'species_scientific': species,
                'species_english': rec.get('en', ''),
                'quality': rec.get('q', ''),
                'length': rec.get('length', ''),
                'length_seconds': parse_length_to_seconds(rec.get('length', '0')),
                'country': rec.get('cnt', ''),
                'location': rec.get('loc', ''),
                'recordist': rec.get('rec', ''),
                'date': rec.get('date', ''),
                'time': rec.get('time', ''),
                'url': rec.get('url', ''),
                'file_name': rec.get('file-name', ''),
            })
        
        # Mark as complete
        checkpoint['completed_species'].append(species)
        checkpoint['total_downloaded'] += len(recordings)
        
        # Save checkpoint after each species
        save_checkpoint(checkpoint)
        
        print(f"  💾 Checkpoint saved. Total downloaded: {checkpoint['total_downloaded']}")
        
        # Rate limiting (be nice to the server)
        time.sleep(2)
    
    # Save final metadata
    df = pd.DataFrame(checkpoint['metadata'])
    output_file = METADATA_DIR / "aggressive_scrape_metadata.csv"
    df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 70)
    print("✅ SCRAPING COMPLETE!")
    print("=" * 70)
    print(f"Total species processed: {len(checkpoint['completed_species'])}")
    print(f"Total recordings downloaded: {checkpoint['total_downloaded']}")
    print(f"Failed species: {len(checkpoint['failed_species'])}")
    print(f"Metadata saved to: {output_file}")
    print("=" * 70)
    
    # Summary statistics
    if checkpoint['metadata']:
        df = pd.DataFrame(checkpoint['metadata'])
        print("\n📊 SUMMARY STATISTICS")
        print("=" * 70)
        print(f"Total recordings: {len(df)}")
        print(f"Unique species: {df['species_scientific'].nunique()}")
        print(f"Average length: {df['length_seconds'].mean():.1f} seconds")
        print(f"\nTop 10 species by count:")
        print(df['species_scientific'].value_counts().head(10))
        print("=" * 70)

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Create directories
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    
    # Combine priority + new species
    all_target_species = list(set(PRIORITY_SPECIES + NEW_SPECIES))
    
    # Remove already completed species
    remaining_species = [s for s in all_target_species if s not in checkpoint['completed_species']]
    
    print(f"\n🎯 Total target species: {len(all_target_species)}")
    print(f"✅ Already completed: {len(checkpoint['completed_species'])}")
    print(f"⏳ Remaining: {len(remaining_species)}")
    print(f"📊 Expected new recordings: ~{len(remaining_species) * MAX_RECORDINGS_PER_SPECIES:,}")
    print(f"💾 Expected new chunks: ~{len(remaining_species) * MAX_RECORDINGS_PER_SPECIES * 10:,}")
    
    if remaining_species:
        confirm = input(f"\n⚠️ Continue scraping {len(remaining_species)} remaining species? (yes/no): ")
        
        if confirm.lower() == 'yes':
            scrape_all_species(all_target_species, checkpoint)
        else:
            print("❌ Cancelled")
    else:
        print("\n✅ All species already scraped!")
        print(f"Total downloaded: {checkpoint['total_downloaded']} recordings")