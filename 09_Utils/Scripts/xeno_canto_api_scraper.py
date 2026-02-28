"""
XENO-CANTO API v3 SCRAPER WITH API KEY
"""

import os
import json
import time
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"
AUDIO_FOLDER = os.path.join(PROJECT_ROOT, "01_Raw_Data", "Audio_Recordings")
METADATA_FOLDER = os.path.join(PROJECT_ROOT, "01_Raw_Data", "Metadata")
OUTPUT_FILE = os.path.join(METADATA_FOLDER, "scraped_metadata_v3.csv")
CHECKPOINT_FILE = os.path.join(METADATA_FOLDER, "checkpoint_v3.json")

# YOUR API KEY
API_KEY = os.getenv("XENO_CANTO_API_KEY", "your_api_key_here")

# API Settings
BASE_URL = "https://xeno-canto.org/api/3/recordings"
DELAY = 0.5  # seconds between requests
TIMEOUT = 20

# ============================================================
# FUNCTIONS
# ============================================================

def extract_xc_id(filename):
    name = os.path.splitext(filename)[0]
    if name.upper().startswith('XC'):
        return name[2:]
    return name


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'failed': []}


def save_checkpoint(completed, failed):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'completed': list(completed), 'failed': list(failed)}, f)


def fetch_metadata(xc_id):
    """Fetch metadata for one recording"""
    
    url = f"{BASE_URL}?query=nr:{xc_id}&key={API_KEY}"
    
    try:
        response = requests.get(url, timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('numRecordings', '0') != '0' and len(data.get('recordings', [])) > 0:
                rec = data['recordings'][0]
                
                return {
                    'xc_id': xc_id,
                    'species_scientific': f"{rec.get('gen', '')} {rec.get('sp', '')}".strip(),
                    'species_english': rec.get('en', ''),
                    'country': rec.get('cnt', ''),
                    'location': rec.get('loc', ''),
                    'latitude': rec.get('lat', ''),
                    'longitude': rec.get('lng', ''),
                    'date': rec.get('date', ''),
                    'time': rec.get('time', ''),
                    'recordist': rec.get('rec', ''),
                    'quality': rec.get('q', ''),
                    'length': rec.get('length', ''),
                    'type': rec.get('type', ''),
                    'scraped_at': datetime.now().isoformat()
                }, True
        
        return None, False
        
    except Exception as e:
        return None, False


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("XENO-CANTO API v3 SCRAPER")
    print("=" * 60)
    
    os.makedirs(METADATA_FOLDER, exist_ok=True)
    
    # Test API first
    print("\n🧪 Testing API...")
    test_url = f"{BASE_URL}?query=nr:475302&key={API_KEY}"
    
    try:
        resp = requests.get(test_url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('numRecordings', '0') != '0':
                print("✅ API working!")
                rec = data['recordings'][0]
                print(f"   Test: {rec.get('en', 'Unknown')} ({rec.get('gen', '')} {rec.get('sp', '')})")
            else:
                print("⚠️ API returned no results")
                return
        else:
            print(f"❌ API error: {resp.status_code}")
            print(resp.text[:200])
            return
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return
    
    # Get all audio files
    audio_files = [f for f in os.listdir(AUDIO_FOLDER) 
                   if f.lower().endswith(('.mp3', '.wav', '.ogg', '.flac'))]
    
    all_ids = list(set([extract_xc_id(f) for f in audio_files if extract_xc_id(f).isdigit()]))
    print(f"\n📁 Audio files: {len(audio_files)}")
    print(f"🔢 Unique IDs: {len(all_ids)}")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    completed = set(checkpoint.get('completed', []))
    failed = set(checkpoint.get('failed', []))
    
    print(f"✅ Already done: {len(completed)}")
    
    # Load existing results
    results = []
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        results = existing.to_dict('records')
    
    # IDs to scrape
    ids_to_scrape = [xc_id for xc_id in all_ids if xc_id not in completed]
    print(f"📋 Remaining: {len(ids_to_scrape)}")
    
    if len(ids_to_scrape) == 0:
        print("\n✅ All done!")
        return
    
    # Scrape
    print(f"\n🚀 Starting...")
    print("-" * 60)
    
    success = len(completed)
    fail = len(failed)
    
    try:
        for i, xc_id in enumerate(tqdm(ids_to_scrape, desc="Scraping")):
            
            metadata, ok = fetch_metadata(xc_id)
            
            if ok:
                results.append(metadata)
                completed.add(xc_id)
                success += 1
            else:
                failed.add(xc_id)
                fail += 1
            
            # Save every 100
            if (i + 1) % 100 == 0:
                save_checkpoint(completed, failed)
                pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
                tqdm.write(f"💾 Saved: ✅{success} ❌{fail}")
            
            time.sleep(DELAY)
            
    except KeyboardInterrupt:
        print("\n⚠️ Stopped by user")
    
    finally:
        # Final save
        save_checkpoint(completed, failed)
        if results:
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
        
        print("\n" + "=" * 60)
        print("DONE!")
        print("=" * 60)
        print(f"✅ Success: {success}")
        print(f"❌ Failed: {fail}")
        print(f"📁 Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()