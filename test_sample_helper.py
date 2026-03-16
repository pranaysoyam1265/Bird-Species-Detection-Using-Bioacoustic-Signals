"""
Test Helper Script: Easy tools for testing with long samples
"""

import os
import glob
import json
import csv
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

class TestSampleHelper:
    """Utility class for working with long test samples"""
    
    def __init__(self, test_dir="10_Outputs/Test_Samples_Long"):
        self.test_dir = test_dir
        self.samples = sorted(glob.glob(os.path.join(test_dir, "test_composite_*.wav")))
        self.metadata = self.load_metadata()
    
    def load_metadata(self) -> List[Dict]:
        """Load metadata JSON"""
        metadata_file = os.path.join(self.test_dir, "test_samples_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return []
    
    def load_sample(self, sample_idx: int, sr: int = 22050) -> Tuple[np.ndarray, int]:
        """Load a test sample by index (1-20)"""
        if not 1 <= sample_idx <= len(self.samples):
            raise ValueError(f"Sample index must be 1-{len(self.samples)}")
        
        sample_path = self.samples[sample_idx - 1]
        audio, sr = librosa.load(sample_path, sr=sr)
        return audio, sr
    
    def load_all_samples(self, sr: int = 22050) -> List[Tuple[np.ndarray, str]]:
        """Load all test samples"""
        samples = []
        for sample_path in self.samples:
            audio, sr = librosa.load(sample_path, sr=sr)
            filename = os.path.basename(sample_path)
            samples.append((audio, filename))
        return samples
    
    def get_sample_info(self, sample_idx: int) -> Dict:
        """Get metadata for a specific sample"""
        if not self.metadata or sample_idx > len(self.metadata):
            return {}
        return self.metadata[sample_idx - 1]
    
    def chunk_sample(self, audio: np.ndarray, sr: int, 
                    chunk_duration: float = 5.0, hop_duration: float = 1.0) -> List[np.ndarray]:
        """Split sample into overlapping chunks"""
        chunk_samples = int(chunk_duration * sr)
        hop_samples = int(hop_duration * sr)
        
        chunks = []
        for start in range(0, len(audio) - chunk_samples, hop_samples):
            chunk = audio[start:start + chunk_samples]
            chunks.append(chunk)
        
        return chunks
    
    def get_statistics(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Calculate audio statistics"""
        return {
            'duration_seconds': len(audio) / sr,
            'rms_energy': float(np.sqrt(np.mean(audio ** 2))),
            'mean_amplitude': float(np.mean(np.abs(audio))),
            'max_amplitude': float(np.max(np.abs(audio))),
            'min_amplitude': float(np.min(np.abs(audio))),
            'std_dev': float(np.std(audio)),
        }
    
    def extract_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract common audio features"""
        features = {}
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = mfcc.mean(axis=1).tolist()
        features['mfcc_std'] = mfcc.std(axis=1).tolist()
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['mel_spec_mean'] = mel_db.mean(axis=1).tolist()
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = float(zcr.mean())
        features['zcr_std'] = float(zcr.std())
        
        # Spectral centroid
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = float(spec_cent.mean())
        
        # Temporal features
        energy = np.sqrt(np.sum(mel_spec ** 2, axis=0))
        features['energy_mean'] = float(energy.mean())
        features['energy_std'] = float(energy.std())
        
        return features
    
    def get_sample_summary(self) -> pd.DataFrame:
        """Get summary of all samples as DataFrame"""
        data = []
        for i, sample_path in enumerate(self.samples, 1):
            audio, sr = librosa.load(sample_path, sr=22050)
            stats = self.get_statistics(audio, sr)
            data.append({
                'sample_id': i,
                'filename': os.path.basename(sample_path),
                'duration_seconds': stats['duration_seconds'],
                'rms_energy': stats['rms_energy'],
                'max_amplitude': stats['max_amplitude'],
                'num_recordings': self.metadata[i-1]['num_recordings_used'] if self.metadata else 'N/A'
            })
        return pd.DataFrame(data)
    
    def print_summary(self):
        """Print summary of all test samples"""
        print("\n" + "="*80)
        print("TEST SAMPLES SUMMARY")
        print("="*80 + "\n")
        
        df = self.get_sample_summary()
        print(df.to_string(index=False))
        
        print("\n" + "="*80)
        print(f"Total Samples: {len(self.samples)}")
        print(f"Total Duration: {df['duration_seconds'].sum()/60:.1f} minutes")
        print(f"Average RMS Energy: {df['rms_energy'].mean():.4f}")
        print(f"Average Recordings/Sample: {df['num_recordings'].apply(lambda x: x if isinstance(x, str) else x).mean():.1f}")
        print("="*80 + "\n")


def example_test_workflow():
    """Example: How to test with long samples"""
    print("🎯 EXAMPLE TEST WORKFLOW\n")
    
    # Initialize helper
    helper = TestSampleHelper("10_Outputs/Test_Samples_Long")
    
    print(f"✓ Initialized with {len(helper.samples)} test samples\n")
    
    # Load first sample
    print("Loading sample 1...")
    audio, sr = helper.load_sample(1)
    print(f"  • Loaded: {len(audio):,} samples at {sr} Hz\n")
    
    # Get sample info
    print("Sample Info:")
    info = helper.get_sample_info(1)
    print(f"  • Source recordings: {info['num_recordings_used']}")
    print(f"  • Duration: {info['actual_duration']:.1f} seconds\n")
    
    # Get statistics
    print("Audio Statistics:")
    stats = helper.get_statistics(audio, sr)
    for key, val in stats.items():
        print(f"  • {key}: {val:.6f}")
    print()
    
    # Extract features
    print("Extracting features (this may take a moment)...")
    features = helper.extract_features(audio, sr)
    print("  ✓ Features extracted:")
    print(f"    - MFCC: {len(features['mfcc_mean'])} coefficients")
    print(f"    - Mel Spectrogram: {len(features['mel_spec_mean'])} bins")
    print(f"    - Spectral Centroid: {features['spectral_centroid_mean']:.1f} Hz")
    print(f"    - ZCR Mean: {features['zcr_mean']:.4f}\n")
    
    # Create chunks for processing
    print("Creating 5-second chunks with 1-second hop...")
    chunks = helper.chunk_sample(audio, sr, chunk_duration=5.0, hop_duration=1.0)
    print(f"  ✓ Created {len(chunks)} overlapping chunks\n")
    
    # Print overall summary
    helper.print_summary()
    
    print("✓ Example workflow complete!\n")


# Main usage instructions
if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                   TEST SAMPLE HELPER SCRIPT                                ║
    ║                  Using Long Concatenated Test Samples                      ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    
    QUICK START:
    ────────────
    
    from test_sample_helper import TestSampleHelper
    
    # Initialize
    helper = TestSampleHelper("10_Outputs/Test_Samples_Long")
    
    # Load sample 1
    audio, sr = helper.load_sample(1)
    
    # Get statistics
    stats = helper.get_statistics(audio, sr)
    print(stats)
    
    # Extract features
    features = helper.extract_features(audio, sr)
    
    # Create chunks for processing
    chunks = helper.chunk_sample(audio, sr, chunk_duration=30)
    
    # Load all samples
    all_samples = helper.load_all_samples()
    
    # Get summary
    helper.print_summary()
    
    
    AVAILABLE METHODS:
    ──────────────────
    • load_sample(sample_idx, sr=22050) → (audio, sample_rate)
    • load_all_samples(sr=22050) → list of (audio, filename) tuples
    • get_sample_info(sample_idx) → dict with metadata
    • chunk_sample(audio, sr, chunk_duration, hop_duration) → list of chunks
    • get_statistics(audio, sr) → dict with audio stats
    • extract_features(audio, sr) → dict with MFCC, mel-spec, etc.
    • get_sample_summary() → pandas DataFrame
    • print_summary() → prints formatted table
    
    
    RUNNING EXAMPLE:
    ────────────────
    python test_sample_helper.py --example
    
    """)
    
    import sys
    if "--example" in sys.argv:
        example_test_workflow()
