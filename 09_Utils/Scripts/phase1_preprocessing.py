"""
Phase 1: Audio Preprocessing Script
====================================
Bird Detection Project - Audio Preprocessing Pipeline

Settings:
- Sample Rate: 22050 Hz
- Chunk Length: 5 seconds
- Overlap: 50% (2.5 seconds)
- Channels: Mono
- Output Format: WAV
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import warnings
import json

warnings.filterwarnings('ignore')

# ============================================================
# CHECK AND INSTALL REQUIRED LIBRARIES
# ============================================================

def check_libraries():
    """Check if required libraries are installed"""
    
    missing = []
    
    try:
        import librosa
    except ImportError:
        missing.append("librosa")
    
    try:
        import soundfile as sf
    except ImportError:
        missing.append("soundfile")
    
    try:
        from scipy import signal
    except ImportError:
        missing.append("scipy")
    
    if missing:
        print("=" * 60)
        print("❌ MISSING LIBRARIES")
        print("=" * 60)
        print(f"\nPlease install: {', '.join(missing)}")
        print(f"\nRun: pip install {' '.join(missing)}")
        print("=" * 60)
        return False
    
    return True

# Check libraries before proceeding
if not check_libraries():
    exit()

# Now import libraries
import librosa
import soundfile as sf
from scipy import signal

# ============================================================
# CONFIGURATION
# ============================================================

# Base paths
BASE_FOLDER = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"

# Input/Output paths
INPUT_AUDIO_FOLDER = os.path.join(BASE_FOLDER, "01_Raw_Data", "Audio_Recordings")
OUTPUT_STANDARDIZED = os.path.join(BASE_FOLDER, "02_Preprocessed", "Standardized_Audio")
OUTPUT_CHUNKS = os.path.join(BASE_FOLDER, "02_Preprocessed", "Audio_Chunks")
OUTPUT_REPORTS = os.path.join(BASE_FOLDER, "02_Preprocessed", "Quality_Reports")

# Audio settings
TARGET_SAMPLE_RATE = 22050  # Hz
CHUNK_LENGTH_SEC = 5        # seconds
OVERLAP_PERCENT = 50        # percent overlap between chunks
CHANNELS = 1                # mono

# Derived settings
CHUNK_LENGTH_SAMPLES = TARGET_SAMPLE_RATE * CHUNK_LENGTH_SEC
OVERLAP_SAMPLES = int(CHUNK_LENGTH_SAMPLES * OVERLAP_PERCENT / 100)
HOP_SAMPLES = CHUNK_LENGTH_SAMPLES - OVERLAP_SAMPLES

# Quality thresholds
MIN_DURATION_SEC = 1.0      # Minimum duration to process
MAX_DURATION_SEC = 600.0    # Maximum duration (10 minutes)
MIN_RMS_THRESHOLD = 0.001   # Minimum RMS to consider non-silent

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_audio_info(filepath):
    """Get basic audio file information without loading full audio"""
    try:
        info = sf.info(filepath)
        return {
            'duration': info.duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'format': info.format,
            'subtype': info.subtype,
            'valid': True,
            'error': None
        }
    except Exception as e:
        return {
            'duration': 0,
            'sample_rate': 0,
            'channels': 0,
            'format': 'unknown',
            'subtype': 'unknown',
            'valid': False,
            'error': str(e)
        }


def load_audio(filepath, target_sr=TARGET_SAMPLE_RATE):
    """Load audio file and convert to mono with target sample rate"""
    try:
        # Load audio with librosa (automatically resamples)
        audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
        return audio, sr, None
    except Exception as e:
        return None, None, str(e)


def calculate_rms(audio):
    """Calculate RMS (Root Mean Square) energy of audio"""
    return np.sqrt(np.mean(audio ** 2))


def is_silent(audio, threshold=MIN_RMS_THRESHOLD):
    """Check if audio is mostly silent"""
    rms = calculate_rms(audio)
    return rms < threshold


def normalize_audio(audio, target_peak=0.95):
    """Normalize audio to target peak amplitude"""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio * (target_peak / max_val)
    return audio


def create_chunks(audio, sr, chunk_length_samples, hop_samples, min_chunk_ratio=0.5):
    """
    Split audio into overlapping chunks
    
    Parameters:
    - audio: numpy array of audio samples
    - sr: sample rate
    - chunk_length_samples: length of each chunk in samples
    - hop_samples: hop size between chunks
    - min_chunk_ratio: minimum ratio of chunk_length for last chunk to be kept
    """
    chunks = []
    total_samples = len(audio)
    
    if total_samples < chunk_length_samples * min_chunk_ratio:
        # Audio too short, pad it
        padded = np.zeros(chunk_length_samples)
        padded[:total_samples] = audio
        chunks.append({
            'audio': padded,
            'start_sample': 0,
            'end_sample': total_samples,
            'start_time': 0,
            'end_time': total_samples / sr,
            'is_padded': True
        })
    else:
        # Create overlapping chunks
        start = 0
        chunk_idx = 0
        
        while start < total_samples:
            end = start + chunk_length_samples
            
            if end <= total_samples:
                # Full chunk
                chunk_audio = audio[start:end]
                is_padded = False
            else:
                # Last chunk - check if long enough
                remaining = total_samples - start
                if remaining >= chunk_length_samples * min_chunk_ratio:
                    # Pad the last chunk
                    chunk_audio = np.zeros(chunk_length_samples)
                    chunk_audio[:remaining] = audio[start:total_samples]
                    is_padded = True
                    end = total_samples
                else:
                    # Too short, skip
                    break
            
            chunks.append({
                'audio': chunk_audio,
                'start_sample': start,
                'end_sample': min(end, total_samples),
                'start_time': start / sr,
                'end_time': min(end, total_samples) / sr,
                'is_padded': is_padded
            })
            
            start += hop_samples
            chunk_idx += 1
    
    return chunks


def save_audio(audio, filepath, sr=TARGET_SAMPLE_RATE):
    """Save audio to WAV file"""
    try:
        sf.write(filepath, audio, sr, subtype='PCM_16')
        return True, None
    except Exception as e:
        return False, str(e)


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def analyze_all_audio_files(input_folder):
    """Analyze all audio files and generate report"""
    
    print("\n" + "=" * 65)
    print("📊 PHASE 1.1: ANALYZING AUDIO FILES")
    print("=" * 65)
    
    # Get list of audio files
    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    total_files = len(audio_files)
    
    print(f"\n📁 Found {total_files} WAV files")
    print(f"📍 Location: {input_folder}")
    
    if total_files == 0:
        print("\n❌ No WAV files found!")
        return None
    
    # Analyze each file
    analysis_results = []
    
    print(f"\n🔍 Analyzing files...")
    
    for filename in tqdm(audio_files, desc="Analyzing"):
        filepath = os.path.join(input_folder, filename)
        
        # Get file info
        info = get_audio_info(filepath)
        
        # Extract recording ID
        recording_id = filename.replace('.wav', '')
        
        analysis_results.append({
            'recording_id': recording_id,
            'filename': filename,
            'duration_sec': round(info['duration'], 2),
            'original_sample_rate': info['sample_rate'],
            'original_channels': info['channels'],
            'format': info['format'],
            'valid': info['valid'],
            'error': info['error']
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(analysis_results)
    
    # Calculate statistics
    valid_df = df[df['valid'] == True]
    
    print("\n" + "-" * 65)
    print("📈 ANALYSIS SUMMARY")
    print("-" * 65)
    
    print(f"\n📁 Total files: {total_files}")
    print(f"✅ Valid files: {len(valid_df)}")
    print(f"❌ Invalid/corrupted: {total_files - len(valid_df)}")
    
    if len(valid_df) > 0:
        print(f"\n🕐 Duration Statistics:")
        print(f"   Min: {valid_df['duration_sec'].min():.2f} sec")
        print(f"   Max: {valid_df['duration_sec'].max():.2f} sec")
        print(f"   Mean: {valid_df['duration_sec'].mean():.2f} sec")
        print(f"   Total: {valid_df['duration_sec'].sum() / 3600:.2f} hours")
        
        print(f"\n🎵 Sample Rates Found:")
        for sr, count in valid_df['original_sample_rate'].value_counts().items():
            print(f"   {sr} Hz: {count} files")
        
        print(f"\n🔊 Channels Found:")
        for ch, count in valid_df['original_channels'].value_counts().items():
            ch_name = "Mono" if ch == 1 else "Stereo" if ch == 2 else f"{ch}-channel"
            print(f"   {ch_name}: {count} files")
    
    # Save analysis report
    report_path = os.path.join(OUTPUT_REPORTS, "audio_analysis_report.csv")
    df.to_csv(report_path, index=False)
    print(f"\n💾 Analysis report saved: {report_path}")
    
    return df


# ============================================================
# STANDARDIZATION FUNCTIONS
# ============================================================

def standardize_audio_files(analysis_df, input_folder, output_folder):
    """Standardize all audio files to target format"""
    
    print("\n" + "=" * 65)
    print("🔧 PHASE 1.2: STANDARDIZING AUDIO FILES")
    print("=" * 65)
    
    print(f"\n⚙️  Target Settings:")
    print(f"   Sample Rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"   Channels: Mono")
    print(f"   Format: WAV (PCM 16-bit)")
    
    # Filter valid files
    valid_df = analysis_df[analysis_df['valid'] == True].copy()
    total_files = len(valid_df)
    
    print(f"\n📁 Processing {total_files} valid files...")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each file
    results = []
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for idx, row in tqdm(valid_df.iterrows(), total=total_files, desc="Standardizing"):
        filename = row['filename']
        recording_id = row['recording_id']
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            # Skip if already processed
            if os.path.exists(output_path):
                skip_count += 1
                results.append({
                    'recording_id': recording_id,
                    'status': 'skipped',
                    'reason': 'already exists'
                })
                continue
            
            # Check duration limits
            if row['duration_sec'] < MIN_DURATION_SEC:
                skip_count += 1
                results.append({
                    'recording_id': recording_id,
                    'status': 'skipped',
                    'reason': f'too short ({row["duration_sec"]:.2f}s)'
                })
                continue
            
            if row['duration_sec'] > MAX_DURATION_SEC:
                skip_count += 1
                results.append({
                    'recording_id': recording_id,
                    'status': 'skipped',
                    'reason': f'too long ({row["duration_sec"]:.2f}s)'
                })
                continue
            
            # Load and resample audio
            audio, sr, error = load_audio(input_path, TARGET_SAMPLE_RATE)
            
            if error:
                error_count += 1
                results.append({
                    'recording_id': recording_id,
                    'status': 'error',
                    'reason': error
                })
                continue
            
            # Check if silent
            if is_silent(audio):
                skip_count += 1
                results.append({
                    'recording_id': recording_id,
                    'status': 'skipped',
                    'reason': 'silent audio'
                })
                continue
            
            # Normalize audio
            audio = normalize_audio(audio)
            
            # Save standardized audio
            success, error = save_audio(audio, output_path, TARGET_SAMPLE_RATE)
            
            if success:
                success_count += 1
                results.append({
                    'recording_id': recording_id,
                    'status': 'success',
                    'reason': None
                })
            else:
                error_count += 1
                results.append({
                    'recording_id': recording_id,
                    'status': 'error',
                    'reason': error
                })
                
        except Exception as e:
            error_count += 1
            results.append({
                'recording_id': recording_id,
                'status': 'error',
                'reason': str(e)
            })
    
    # Summary
    print("\n" + "-" * 65)
    print("📊 STANDARDIZATION SUMMARY")
    print("-" * 65)
    print(f"\n✅ Successfully processed: {success_count}")
    print(f"⏭️  Skipped: {skip_count}")
    print(f"❌ Errors: {error_count}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(OUTPUT_REPORTS, "standardization_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved: {results_path}")
    
    return results_df


# ============================================================
# CHUNKING FUNCTIONS
# ============================================================

def create_audio_chunks(input_folder, output_folder):
    """Split standardized audio into fixed-length chunks"""
    
    print("\n" + "=" * 65)
    print("✂️  PHASE 1.3: CREATING AUDIO CHUNKS")
    print("=" * 65)
    
    print(f"\n⚙️  Chunk Settings:")
    print(f"   Chunk Length: {CHUNK_LENGTH_SEC} seconds")
    print(f"   Overlap: {OVERLAP_PERCENT}% ({OVERLAP_SAMPLES / TARGET_SAMPLE_RATE:.2f} sec)")
    print(f"   Hop Size: {HOP_SAMPLES / TARGET_SAMPLE_RATE:.2f} seconds")
    
    # Get standardized files
    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    total_files = len(audio_files)
    
    if total_files == 0:
        print("\n❌ No standardized audio files found!")
        return None
    
    print(f"\n📁 Processing {total_files} standardized files...")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each file
    chunk_results = []
    total_chunks = 0
    
    for filename in tqdm(audio_files, desc="Chunking"):
        filepath = os.path.join(input_folder, filename)
        recording_id = filename.replace('.wav', '')
        
        try:
            # Load audio
            audio, sr = librosa.load(filepath, sr=TARGET_SAMPLE_RATE, mono=True)
            
            # Create chunks
            chunks = create_chunks(audio, sr, CHUNK_LENGTH_SAMPLES, HOP_SAMPLES)
            
            # Save each chunk
            for i, chunk_data in enumerate(chunks):
                chunk_filename = f"{recording_id}_chunk_{i+1:03d}.wav"
                chunk_path = os.path.join(output_folder, chunk_filename)
                
                # Save chunk
                save_audio(chunk_data['audio'], chunk_path, TARGET_SAMPLE_RATE)
                
                chunk_results.append({
                    'recording_id': recording_id,
                    'chunk_filename': chunk_filename,
                    'chunk_index': i + 1,
                    'start_time': round(chunk_data['start_time'], 3),
                    'end_time': round(chunk_data['end_time'], 3),
                    'is_padded': chunk_data['is_padded']
                })
                
                total_chunks += 1
                
        except Exception as e:
            tqdm.write(f"   ⚠️ Error processing {filename}: {str(e)[:50]}")
    
    # Summary
    print("\n" + "-" * 65)
    print("📊 CHUNKING SUMMARY")
    print("-" * 65)
    print(f"\n📁 Files processed: {total_files}")
    print(f"✂️  Total chunks created: {total_chunks}")
    print(f"📈 Average chunks per file: {total_chunks / total_files:.1f}")
    
    # Save chunk mapping
    chunks_df = pd.DataFrame(chunk_results)
    mapping_path = os.path.join(OUTPUT_REPORTS, "chunk_mapping.csv")
    chunks_df.to_csv(mapping_path, index=False)
    print(f"\n💾 Chunk mapping saved: {mapping_path}")
    
    return chunks_df


# ============================================================
# QUALITY CHECK FUNCTIONS
# ============================================================

def quality_check_chunks(chunks_folder, chunk_mapping_df):
    """Perform quality checks on generated chunks"""
    
    print("\n" + "=" * 65)
    print("🔍 PHASE 1.4: QUALITY CHECKING CHUNKS")
    print("=" * 65)
    
    chunk_files = [f for f in os.listdir(chunks_folder) if f.endswith('.wav')]
    total_chunks = len(chunk_files)
    
    print(f"\n📁 Checking {total_chunks} chunks...")
    
    quality_results = []
    silent_count = 0
    low_energy_count = 0
    good_count = 0
    
    for filename in tqdm(chunk_files, desc="Quality Check"):
        filepath = os.path.join(chunks_folder, filename)
        
        try:
            # Load chunk
            audio, sr = librosa.load(filepath, sr=TARGET_SAMPLE_RATE, mono=True)
            
            # Calculate metrics
            rms = calculate_rms(audio)
            peak = np.max(np.abs(audio))
            
            # Classify quality
            if rms < MIN_RMS_THRESHOLD:
                quality = 'silent'
                silent_count += 1
            elif rms < MIN_RMS_THRESHOLD * 5:
                quality = 'low_energy'
                low_energy_count += 1
            else:
                quality = 'good'
                good_count += 1
            
            quality_results.append({
                'chunk_filename': filename,
                'rms': round(rms, 6),
                'peak': round(peak, 4),
                'quality': quality
            })
            
        except Exception as e:
            quality_results.append({
                'chunk_filename': filename,
                'rms': 0,
                'peak': 0,
                'quality': 'error'
            })
    
    # Summary
    print("\n" + "-" * 65)
    print("📊 QUALITY CHECK SUMMARY")
    print("-" * 65)
    print(f"\n✅ Good quality: {good_count} ({100*good_count/total_chunks:.1f}%)")
    print(f"⚠️  Low energy: {low_energy_count} ({100*low_energy_count/total_chunks:.1f}%)")
    print(f"🔇 Silent: {silent_count} ({100*silent_count/total_chunks:.1f}%)")
    
    # Save quality report
    quality_df = pd.DataFrame(quality_results)
    quality_path = os.path.join(OUTPUT_REPORTS, "chunk_quality_report.csv")
    quality_df.to_csv(quality_path, index=False)
    print(f"\n💾 Quality report saved: {quality_path}")
    
    return quality_df


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    start_time = datetime.now()
    
    print("\n" + "🐦" * 25)
    print("\n   PHASE 1: AUDIO PREPROCESSING PIPELINE")
    print("\n" + "🐦" * 25)
    
    print(f"\n📅 Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📍 Project: {BASE_FOLDER}")
    
    # Create output directories
    os.makedirs(OUTPUT_STANDARDIZED, exist_ok=True)
    os.makedirs(OUTPUT_CHUNKS, exist_ok=True)
    os.makedirs(OUTPUT_REPORTS, exist_ok=True)
    
    # =========================================================
    # STEP 1: Analyze Audio Files
    # =========================================================
    analysis_df = analyze_all_audio_files(INPUT_AUDIO_FOLDER)
    
    if analysis_df is None or len(analysis_df) == 0:
        print("\n❌ No audio files to process. Exiting.")
        return
    
    # =========================================================
    # STEP 2: Standardize Audio Files
    # =========================================================
    print("\n")
    proceed = input("📋 Proceed with standardization? (Y/N): ").strip().upper()
    
    if proceed != 'Y':
        print("\n⏸️  Standardization skipped.")
        return
    
    standardization_df = standardize_audio_files(
        analysis_df, 
        INPUT_AUDIO_FOLDER, 
        OUTPUT_STANDARDIZED
    )
    
    # =========================================================
    # STEP 3: Create Audio Chunks
    # =========================================================
    print("\n")
    proceed = input("📋 Proceed with chunking? (Y/N): ").strip().upper()
    
    if proceed != 'Y':
        print("\n⏸️  Chunking skipped.")
        return
    
    chunk_mapping_df = create_audio_chunks(
        OUTPUT_STANDARDIZED, 
        OUTPUT_CHUNKS
    )
    
    # =========================================================
    # STEP 4: Quality Check
    # =========================================================
    print("\n")
    proceed = input("📋 Proceed with quality check? (Y/N): ").strip().upper()
    
    if proceed == 'Y':
        quality_df = quality_check_chunks(OUTPUT_CHUNKS, chunk_mapping_df)
    
    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 65)
    print("🎉 PHASE 1 COMPLETE!")
    print("=" * 65)
    
    print(f"\n📅 Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Duration: {duration}")
    
    # Count outputs
    standardized_count = len([f for f in os.listdir(OUTPUT_STANDARDIZED) if f.endswith('.wav')])
    chunk_count = len([f for f in os.listdir(OUTPUT_CHUNKS) if f.endswith('.wav')])
    
    print(f"\n📊 OUTPUT SUMMARY:")
    print(f"   📁 Standardized audio: {standardized_count} files")
    print(f"   ✂️  Audio chunks: {chunk_count} files")
    print(f"\n📍 Output locations:")
    print(f"   Standardized: {OUTPUT_STANDARDIZED}")
    print(f"   Chunks: {OUTPUT_CHUNKS}")
    print(f"   Reports: {OUTPUT_REPORTS}")
    
    # Save final config
    config = {
        'target_sample_rate': TARGET_SAMPLE_RATE,
        'chunk_length_sec': CHUNK_LENGTH_SEC,
        'overlap_percent': OVERLAP_PERCENT,
        'channels': CHANNELS,
        'processed_date': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'input_files': len(analysis_df),
        'standardized_files': standardized_count,
        'total_chunks': chunk_count
    }
    
    config_path = os.path.join(OUTPUT_REPORTS, "preprocessing_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"\n💾 Config saved: {config_path}")
    
    print("\n" + "=" * 65)
    print("📋 NEXT STEPS:")
    print("   1. Review quality reports in Quality_Reports/")
    print("   2. Proceed to Phase 2: Data Augmentation")
    print("   3. Or skip to Phase 3: Label Preparation (if metadata ready)")
    print("=" * 65)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()