#!/usr/bin/env python3
"""
Test BirdSense API with real audio files.
"""

import requests
import json
from pathlib import Path

API_URL = "https://pranaysoyam126-birdsense-backend.hf.space"

def test_health():
    """Test health endpoint."""
    print("\n" + "=" * 50)
    print("TEST 1: Health Check")
    print("=" * 50)
    
    try:
        r = requests.get(f"{API_URL}/health", timeout=30)
        data = r.json()
        print(f"  Status: {data.get('status')}")
        print(f"  Model loaded: {data.get('model_loaded')}")
        print(f"  Species: {data.get('num_species')}")
        
        if data.get('model_loaded') and data.get('num_species') == 290:
            print("  ✅ PASSED")
        else:
            print("  ❌ FAILED")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")


def test_species():
    """Test species endpoint."""
    print("\n" + "=" * 50)
    print("TEST 2: Species List")
    print("=" * 50)
    
    try:
        r = requests.get(f"{API_URL}/species", timeout=30)
        data = r.json()
        
        species_list = data if isinstance(data, list) else data.get('species', [])
        print(f"  Total species: {len(species_list)}")
        
        if len(species_list) >= 290:
            print("  ✅ PASSED")
        else:
            print(f"  ❌ FAILED - Expected 290, got {len(species_list)}")
        
        # Show first 5
        print("\n  Sample species:")
        for sp in species_list[:5]:
            if isinstance(sp, dict):
                print(f"    - {sp.get('english_name', sp.get('species', '?'))}")
            else:
                print(f"    - {sp}")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")


def test_detection(audio_path):
    """Test detection with an audio file."""
    print("\n" + "=" * 50)
    print(f"TEST 3: Detection - {Path(audio_path).name}")
    print("=" * 50)
    
    if not Path(audio_path).exists():
        print(f"  ❌ File not found: {audio_path}")
        return
    
    try:
        with open(audio_path, 'rb') as f:
            files = {'audio_file': (Path(audio_path).name, f)}
            data = {
                'top_k': 5,
                'confidence_threshold': 0.01,
                'noise_reduction': False,
                'chunk_duration': 5.0
            }
            
            print("  Uploading and analyzing...")
            r = requests.post(
                f"{API_URL}/detect",
                files=files,
                data=data,
                timeout=120
            )
        
        if r.status_code == 200:
            result = r.json()
            
            print(f"\n  Status: {result.get('status')}")
            print(f"  Duration: {result.get('duration')}s")
            print(f"  Processing time: {result.get('processing_time_ms')}ms")
            print(f"\n  🏆 Top Species: {result.get('top_species')}")
            print(f"     Scientific: {result.get('top_scientific')}")
            print(f"     Confidence: {result.get('top_confidence')}%")
            
            preds = result.get('predictions', [])
            if preds:
                print(f"\n  Top {len(preds)} Predictions:")
                for p in preds:
                    print(f"    - {p.get('species')}: {p.get('confidence')}%")
            
            segments = result.get('segments', [])
            print(f"\n  Segments analyzed: {len(segments)}")
            
            if result.get('top_confidence', 0) > 50:
                print("\n  ✅ PASSED - High confidence detection")
            else:
                print("\n  ⚠️ LOW CONFIDENCE - Check audio quality")
        else:
            print(f"  ❌ FAILED - Status {r.status_code}")
            print(f"  Response: {r.text[:200]}")
    
    except Exception as e:
        print(f"  ❌ ERROR: {e}")


def test_with_sample_files():
    """Test with available audio files."""
    print("\n" + "=" * 50)
    print("Searching for test audio files...")
    print("=" * 50)
    
    # Search for audio files
    search_paths = [
        Path(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\01_Raw_Data\Audio_Recordings"),
        Path(r"C:\BirdSense_P2\data\audio"),
        Path(r"C:\BirdSense_P2\data\phase1_rebuilt\downloads"),
    ]
    
    test_files = []
    
    for search_dir in search_paths:
        if search_dir.exists():
            for ext in ['*.wav', '*.mp3']:
                files = list(search_dir.rglob(ext))
                if files:
                    test_files.extend(files[:2])  # Take 2 from each location
        
        if len(test_files) >= 4:
            break
    
    if test_files:
        print(f"\n  Found {len(test_files)} test files")
        for audio_file in test_files[:4]:
            test_detection(str(audio_file))
    else:
        print("\n  No audio files found for testing")
        print("  Please provide a path to test manually")


def main():
    print("\n" + "=" * 50)
    print("    BirdSense API Test Suite")
    print("=" * 50)
    
    # Test 1: Health
    test_health()
    
    # Test 2: Species
    test_species()
    
    # Test 3: Detection with sample files
    test_with_sample_files()
    
    print("\n" + "=" * 50)
    print("    Tests Complete!")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()