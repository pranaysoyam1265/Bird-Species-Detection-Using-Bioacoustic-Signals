"""
remaining_scraper_v2.py
Scrapes metadata for remaining 3,521 Xeno-Canto recording IDs
Resumes from checkpoint, merges with Batch 1 when complete

Author: Bird Detection Project
Version: 2.0
"""

import os
import re
import time
import random
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# ============================================================
# CONFIGURATION
# ============================================================

# Paths - UPDATE THESE TO MATCH YOUR SETUP
BASE_DIR = Path(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")
AUDIO_DIR = BASE_DIR / "01_Raw_Data" / "Audio_Recordings"
METADATA_DIR = BASE_DIR / "01_Raw_Data" / "Metadata"
SCRIPTS_DIR = BASE_DIR / "09_Utils" / "Scripts"
LOGS_DIR = BASE_DIR / "09_Utils" / "Logs"

# Files
BATCH1_FILE = METADATA_DIR / "bird_metadata_batch1_fixed.csv"
CHECKPOINT_FILE = METADATA_DIR / "bird_metadata_remaining_checkpoint.csv"
REMAINING_OUTPUT = METADATA_DIR / "bird_metadata_remaining.csv"
COMPLETE_OUTPUT = METADATA_DIR / "bird_metadata_complete.csv"
LOG_FILE = LOGS_DIR / "scraping_log_remaining.txt"

# Scraping settings
CHECKPOINT_INTERVAL = 50  # Save every 50 recordings (more frequent)
BASE_DELAY = 2.0  # Base delay between requests
MAX_DELAY = 5.0  # Max delay for random variation
MAX_RETRIES = 3  # Retries per recording
PAGE_TIMEOUT = 15  # Seconds to wait for page load

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def log_message(message, log_file=None):
    """Print and optionally log a message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(formatted + '\n')

def get_all_recording_ids(audio_dir):
    """Extract all XC IDs from audio files"""
    ids = []
    pattern = re.compile(r'XC(\d+)\.wav', re.IGNORECASE)
    
    for file in os.listdir(audio_dir):
        match = pattern.match(file)
        if match:
            ids.append(int(match.group(1)))
    
    return sorted(ids)

def get_batch1_ids(batch1_file):
    """Get IDs already scraped in Batch 1"""
    if not batch1_file.exists():
        return set()
    
    df = pd.read_csv(batch1_file)
    return set(df['recording_id'].tolist())

def get_checkpoint_ids(checkpoint_file):
    """Get IDs already scraped in remaining batch"""
    if not checkpoint_file.exists():
        return set(), pd.DataFrame()
    
    df = pd.read_csv(checkpoint_file)
    return set(df['recording_id'].tolist()), df

def setup_driver():
    """Setup Selenium WebDriver with anti-detection"""
    options = Options()
    
    # Anti-detection settings
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    
    # Custom user agent
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    # Disable automation flags
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    
    # Initialize driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    # Additional anti-detection
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        '''
    })
    
    return driver

def extract_metadata(driver, xc_id):
    """Extract metadata from a Xeno-Canto recording page"""
    url = f"https://xeno-canto.org/{xc_id}"
    
    metadata = {
        'recording_id': xc_id,
        'scientific_name': None,
        'genus': None,
        'species': None,
        'subspecies': None,
        'english_name': None,
        'call_type': None,
        'sex': None,
        'life_stage': None,
        'duration_seconds': 0.0,  # Will calculate from audio file
        'date': None,
        'time': None,
        'country': None,
        'location': None,
        'latitude': None,
        'longitude': None,
        'altitude': None,
        'quality': None,
        'quality_flag': 'unknown',
        'description': None,
        'license': None,
        'remarks': None,
        'background_species': None,
        'status': 'pending'
    }
    
    try:
        driver.get(url)
        time.sleep(1)  # Brief wait for page load
        
        # Wait for main content
        WebDriverWait(driver, PAGE_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Check for 404 or error page
        if "404" in driver.title or "not found" in driver.page_source.lower():
            metadata['status'] = 'not_found'
            return metadata
        
        # Extract species name (scientific)
        try:
            species_elem = driver.find_element(By.CSS_SELECTOR, "span.species-name, h1.scientific-name, .sci-name")
            metadata['scientific_name'] = species_elem.text.strip()
            
            # Parse genus and species
            parts = metadata['scientific_name'].split()
            if len(parts) >= 2:
                metadata['genus'] = parts[0]
                metadata['species'] = parts[1]
                if len(parts) >= 3:
                    metadata['subspecies'] = ' '.join(parts[2:])
        except NoSuchElementException:
            # Try alternative selector
            try:
                h1_elem = driver.find_element(By.TAG_NAME, "h1")
                if h1_elem:
                    text = h1_elem.text.strip()
                    # Parse "Common Name (Scientific Name)" format
                    if '(' in text and ')' in text:
                        metadata['english_name'] = text.split('(')[0].strip()
                        metadata['scientific_name'] = text.split('(')[1].replace(')', '').strip()
            except:
                pass
        
        # Extract English name
        try:
            english_elem = driver.find_element(By.CSS_SELECTOR, "span.common-name, .english-name")
            metadata['english_name'] = english_elem.text.strip()
        except NoSuchElementException:
            pass
        
        # Extract recording details from table
        try:
            # Look for detail rows
            rows = driver.find_elements(By.CSS_SELECTOR, "tr, .detail-row, .recording-detail")
            
            for row in rows:
                try:
                    text = row.text.lower()
                    row_text = row.text
                    
                    if 'type' in text and 'call' in text.lower():
                        metadata['call_type'] = row_text.split(':')[-1].strip() if ':' in row_text else None
                    elif 'sex' in text:
                        metadata['sex'] = row_text.split(':')[-1].strip() if ':' in row_text else None
                    elif 'life stage' in text or 'age' in text:
                        metadata['life_stage'] = row_text.split(':')[-1].strip() if ':' in row_text else None
                    elif 'date' in text and 'time' not in text:
                        metadata['date'] = row_text.split(':')[-1].strip() if ':' in row_text else None
                    elif 'time' in text:
                        metadata['time'] = row_text.split(':')[-1].strip() if ':' in row_text else None
                    elif 'country' in text:
                        metadata['country'] = row_text.split(':')[-1].strip() if ':' in row_text else None
                    elif 'location' in text:
                        metadata['location'] = row_text.split(':')[-1].strip() if ':' in row_text else None
                    elif 'latitude' in text or 'lat' in text:
                        lat_text = row_text.split(':')[-1].strip() if ':' in row_text else None
                        if lat_text:
                            try:
                                metadata['latitude'] = float(re.findall(r'-?\d+\.?\d*', lat_text)[0])
                            except:
                                pass
                    elif 'longitude' in text or 'lon' in text:
                        lon_text = row_text.split(':')[-1].strip() if ':' in row_text else None
                        if lon_text:
                            try:
                                metadata['longitude'] = float(re.findall(r'-?\d+\.?\d*', lon_text)[0])
                            except:
                                pass
                    elif 'elevation' in text or 'altitude' in text:
                        metadata['altitude'] = row_text.split(':')[-1].strip() if ':' in row_text else None
                    elif 'quality' in text:
                        metadata['quality'] = row_text.split(':')[-1].strip() if ':' in row_text else None
                    elif 'recordist' in text or 'recorded by' in text:
                        metadata['description'] = row_text.split(':')[-1].strip() if ':' in row_text else None
                    elif 'license' in text:
                        metadata['license'] = row_text.split(':')[-1].strip() if ':' in row_text else None
                    elif 'remark' in text:
                        metadata['remarks'] = row_text.split(':')[-1].strip() if ':' in row_text else None
                    elif 'background' in text:
                        metadata['background_species'] = row_text.split(':')[-1].strip() if ':' in row_text else None
                        
                except Exception:
                    continue
                    
        except NoSuchElementException:
            pass
        
        # Try to get coordinates from map or embedded data
        try:
            # Look for coordinates in page source
            page_source = driver.page_source
            
            # Try to find lat/lon in page
            lat_match = re.search(r'lat["\s:=]+(-?\d+\.?\d*)', page_source, re.IGNORECASE)
            lon_match = re.search(r'(?:lon|lng)["\s:=]+(-?\d+\.?\d*)', page_source, re.IGNORECASE)
            
            if lat_match and metadata['latitude'] is None:
                metadata['latitude'] = float(lat_match.group(1))
            if lon_match and metadata['longitude'] is None:
                metadata['longitude'] = float(lon_match.group(1))
                
        except:
            pass
        
        # Set quality flag based on quality rating
        if metadata['quality']:
            q = metadata['quality'].upper()
            if 'A' in q:
                metadata['quality_flag'] = 'good'
            elif 'B' in q:
                metadata['quality_flag'] = 'good'
            elif 'C' in q:
                metadata['quality_flag'] = 'good'
            elif 'D' in q:
                metadata['quality_flag'] = 'poor'
            elif 'E' in q:
                metadata['quality_flag'] = 'poor'
            else:
                metadata['quality_flag'] = 'unknown'
        
        metadata['status'] = 'success'
        
    except TimeoutException:
        metadata['status'] = 'timeout'
    except Exception as e:
        metadata['status'] = f'error: {str(e)[:50]}'
    
    return metadata

def calculate_durations(df, audio_dir):
    """Calculate actual duration from audio files using librosa"""
    import librosa
    
    log_message("Calculating audio durations from files...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating durations"):
        xc_id = row['recording_id']
        audio_file = audio_dir / f"XC{xc_id}.wav"
        
        if audio_file.exists():
            try:
                duration = librosa.get_duration(path=str(audio_file))
                df.at[idx, 'duration_seconds'] = round(duration, 2)
            except Exception as e:
                df.at[idx, 'duration_seconds'] = 0.0
        else:
            df.at[idx, 'duration_seconds'] = 0.0
    
    return df

def merge_datasets(batch1_file, remaining_file, output_file):
    """Merge Batch 1 and Remaining into complete dataset"""
    log_message("Merging Batch 1 and Remaining datasets...")
    
    df1 = pd.read_csv(batch1_file) if batch1_file.exists() else pd.DataFrame()
    df2 = pd.read_csv(remaining_file) if remaining_file.exists() else pd.DataFrame()
    
    if df1.empty and df2.empty:
        log_message("ERROR: No data to merge!")
        return None
    
    # Combine
    df_complete = pd.concat([df1, df2], ignore_index=True)
    
    # Remove duplicates
    df_complete = df_complete.drop_duplicates(subset=['recording_id'], keep='first')
    
    # Sort by recording ID
    df_complete = df_complete.sort_values('recording_id').reset_index(drop=True)
    
    # Add file path and existence check
    df_complete['local_file_path'] = df_complete['recording_id'].apply(
        lambda x: str(AUDIO_DIR / f"XC{x}.wav")
    )
    df_complete['file_exists'] = df_complete['local_file_path'].apply(
        lambda x: os.path.exists(x)
    )
    
    # Save
    df_complete.to_csv(output_file, index=False)
    log_message(f"Complete dataset saved: {output_file}")
    log_message(f"Total recordings: {len(df_complete)}")
    
    return df_complete

# ============================================================
# MAIN SCRAPING FUNCTION
# ============================================================

def main():
    """Main scraping function"""
    
    # Setup logging
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_message("=" * 60, LOG_FILE)
    log_message("REMAINING METADATA SCRAPER - STARTING", LOG_FILE)
    log_message("=" * 60, LOG_FILE)
    
    # Get all recording IDs
    log_message("Loading recording IDs...", LOG_FILE)
    all_ids = get_all_recording_ids(AUDIO_DIR)
    log_message(f"Total audio files: {len(all_ids)}", LOG_FILE)
    
    # Get already scraped IDs (Batch 1)
    batch1_ids = get_batch1_ids(BATCH1_FILE)
    log_message(f"Batch 1 already scraped: {len(batch1_ids)}", LOG_FILE)
    
    # Get checkpoint IDs (Remaining - already done)
    checkpoint_ids, checkpoint_df = get_checkpoint_ids(CHECKPOINT_FILE)
    log_message(f"Remaining checkpoint: {len(checkpoint_ids)}", LOG_FILE)
    
    # Calculate IDs to scrape
    already_done = batch1_ids.union(checkpoint_ids)
    ids_to_scrape = [xc_id for xc_id in all_ids if xc_id not in already_done]
    
    log_message(f"IDs remaining to scrape: {len(ids_to_scrape)}", LOG_FILE)
    
    if len(ids_to_scrape) == 0:
        log_message("All IDs already scraped! Merging datasets...", LOG_FILE)
        merge_datasets(BATCH1_FILE, REMAINING_OUTPUT, COMPLETE_OUTPUT)
        return
    
    # Initialize results with checkpoint data
    results = checkpoint_df.to_dict('records') if not checkpoint_df.empty else []
    
    # Setup WebDriver
    log_message("Setting up WebDriver...", LOG_FILE)
    driver = setup_driver()
    
    try:
        # Scraping loop
        log_message(f"Starting scraping of {len(ids_to_scrape)} recordings...", LOG_FILE)
        
        for i, xc_id in enumerate(tqdm(ids_to_scrape, desc="Scraping")):
            
            # Extract metadata with retries
            for attempt in range(MAX_RETRIES):
                metadata = extract_metadata(driver, xc_id)
                
                if metadata['status'] == 'success':
                    break
                elif attempt < MAX_RETRIES - 1:
                    time.sleep(2)  # Wait before retry
            
            results.append(metadata)
            
            # Random delay between requests
            delay = BASE_DELAY + random.uniform(0, MAX_DELAY - BASE_DELAY)
            time.sleep(delay)
            
            # Checkpoint save
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                df_checkpoint = pd.DataFrame(results)
                df_checkpoint.to_csv(CHECKPOINT_FILE, index=False)
                
                success_count = len([r for r in results if r['status'] == 'success'])
                log_message(f"Checkpoint saved: {len(results)} scraped, {success_count} successful", LOG_FILE)
            
            # Progress update every 100
            if (i + 1) % 100 == 0:
                success_count = len([r for r in results if r['status'] == 'success'])
                log_message(f"Progress: {i+1}/{len(ids_to_scrape)} | Success: {success_count}", LOG_FILE)
    
    except KeyboardInterrupt:
        log_message("Interrupted by user. Saving checkpoint...", LOG_FILE)
    
    except Exception as e:
        log_message(f"Error: {str(e)}. Saving checkpoint...", LOG_FILE)
    
    finally:
        # Close driver
        driver.quit()
        
        # Save final results
        df_remaining = pd.DataFrame(results)
        df_remaining.to_csv(REMAINING_OUTPUT, index=False)
        log_message(f"Remaining data saved: {REMAINING_OUTPUT}", LOG_FILE)
        
        # Calculate durations
        log_message("Calculating audio durations...", LOG_FILE)
        df_remaining = calculate_durations(df_remaining, AUDIO_DIR)
        df_remaining.to_csv(REMAINING_OUTPUT, index=False)
        
        # Merge with Batch 1
        df_complete = merge_datasets(BATCH1_FILE, REMAINING_OUTPUT, COMPLETE_OUTPUT)
        
        # Summary
        log_message("=" * 60, LOG_FILE)
        log_message("SCRAPING COMPLETE - SUMMARY", LOG_FILE)
        log_message("=" * 60, LOG_FILE)
        
        if df_complete is not None:
            success_count = len(df_complete[df_complete['status'] == 'success'])
            species_count = df_complete['scientific_name'].nunique()
            
            log_message(f"Total recordings: {len(df_complete)}", LOG_FILE)
            log_message(f"Successful: {success_count}", LOG_FILE)
            log_message(f"Unique species: {species_count}", LOG_FILE)
            log_message(f"Output file: {COMPLETE_OUTPUT}", LOG_FILE)
        
        log_message("Done!", LOG_FILE)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()