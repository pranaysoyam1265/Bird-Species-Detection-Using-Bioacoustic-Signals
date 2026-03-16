"""
Concatenation Script: Build longer test samples from unused short recordings
Creates 5+ minute composite audio samples from unused Xeno-Canto recordings
"""

import os
import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import csv
from datetime import datetime

# Configuration
AUDIO_DIR = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\01_Raw_Data\Audio_Recordings"
UNUSED_IDS_FILE = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\unused_recording_ids.json"
OUTPUT_DIR = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\10_Outputs\Test_Samples_Long"
TARGET_DURATION = 300  # 5 minutes in seconds
MIN_SAMPLES = 20  # Create at least 20 long test samples
SAMPLE_RATE = 22050  # Standard sample rate

class LongTestSampleGenerator:
    def __init__(self, audio_dir, output_dir, target_duration=300, sample_rate=22050):
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.target_duration = target_duration
        self.sample_rate = sample_rate
        self.metadata = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_unused_ids(self, unused_ids_file):
        """Load list of unused recording IDs"""
        if os.path.exists(unused_ids_file):
            with open(unused_ids_file, 'r') as f:
                data = json.load(f)
                return data['unused_ids']
        return []
    
    def find_audio_file(self, xc_id):
        """Find audio file for a given XC ID (can be .mp3 or .wav)"""
        files = [f for f in os.listdir(self.audio_dir) 
                 if f.startswith(f'XC{xc_id}') and (f.endswith('.mp3') or f.endswith('.wav'))]
        return os.path.join(self.audio_dir, files[0]) if files else None
    
    def load_audio(self, filepath):
        """Load audio file safely"""
        try:
            audio, sr = librosa.load(filepath, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            print(f"⚠ Error loading {filepath}: {e}")
            return None, None
    
    def create_test_sample(self, xc_ids, sample_num):
        """Create a long test sample by concatenating multiple short recordings"""
        combined_audio = []
        used_files = []
        total_duration = 0
        failed_count = 0
        
        for xc_id in xc_ids:
            if total_duration >= self.target_duration:
                break
            
            filepath = self.find_audio_file(xc_id)
            if not filepath:
                failed_count += 1
                continue
            
            audio, sr = self.load_audio(filepath)
            if audio is None:
                failed_count += 1
                continue
            
            # Add silence gap between recordings (0.5 seconds)
            if combined_audio:
                silence = np.zeros(int(0.5 * self.sample_rate))
                combined_audio.append(silence)
            
            combined_audio.append(audio)
            total_duration += len(audio) / sr
            used_files.append({
                'xc_id': xc_id,
                'duration': len(audio) / sr,
                'filename': os.path.basename(filepath)
            })
        
        if not combined_audio or total_duration < self.target_duration * 0.8:  # At least 80% of target
            print(f"✗ Sample {sample_num}: Not enough valid audio ({total_duration:.1f}s < {self.target_duration*0.8:.1f}s)")
            return None
        
        # Concatenate and trim to exactly target duration
        full_audio = np.concatenate(combined_audio)
        final_audio = full_audio[:int(self.target_duration * self.sample_rate)]
        
        # Save audio file
        output_filename = f"test_composite_{sample_num:03d}.wav"
        output_path = os.path.join(self.output_dir, output_filename)
        
        sf.write(output_path, final_audio, self.sample_rate)
        
        # Store metadata
        metadata = {
            'sample_id': sample_num,
            'output_file': output_filename,
            'output_path': output_path,
            'target_duration': self.target_duration,
            'actual_duration': float(len(final_audio) / self.sample_rate),
            'num_recordings_used': len(used_files),
            'failed_to_load': failed_count,
            'source_recordings': used_files,
            'created_at': datetime.now().isoformat()
        }
        
        self.metadata.append(metadata)
        
        actual_min = metadata['actual_duration'] / 60
        print(f"✓ Sample {sample_num}: {len(used_files)} clips concatenated ({actual_min:.1f} min from {len(used_files)} recordings)")
        
        return metadata
    
    def generate_samples(self, unused_ids, min_samples=20):
        """Generate multiple long test samples"""
        print(f"\n{'='*70}")
        print(f"GENERATING {min_samples} TEST SAMPLES (Target: {self.target_duration}s each)")
        print(f"{'='*70}\n")
        
        # Shuffle IDs for variety
        np.random.shuffle(unused_ids)
        
        sample_count = 0
        batch_size = 15  # Recordings per composite sample
        
        for i in range(0, len(unused_ids), batch_size):
            if sample_count >= min_samples:
                break
            
            batch = unused_ids[i:i + batch_size]
            sample_count += 1
            
            self.create_test_sample(batch, sample_count)
        
        print(f"\n{'='*70}")
        print(f"SUMMARY:")
        print(f"  • Created: {len(self.metadata)} test samples")
        print(f"  • Location: {self.output_dir}")
        print(f"  • Total audio duration: {sum(m['actual_duration'] for m in self.metadata)/60:.1f} minutes")
        print(f"{'='*70}\n")
        
        return self.metadata
    
    def save_metadata(self):
        """Save metadata about created samples"""
        metadata_file = os.path.join(self.output_dir, "test_samples_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Also save as CSV for easier viewing
        csv_file = os.path.join(self.output_dir, "test_samples_manifest.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Sample ID', 'Filename', 'Duration (sec)', 'Num Recordings', 'Created'])
            for m in self.metadata:
                writer.writerow([
                    m['sample_id'],
                    m['output_file'],
                    f"{m['actual_duration']:.1f}",
                    m['num_recordings_used'],
                    m['created_at']
                ])
        
        print(f"✓ Metadata saved to:")
        print(f"  - {metadata_file}")
        print(f"  - {csv_file}\n")
    
    def generate_usage_report(self):
        """Generate a report summarizing the test samples"""
        report_file = os.path.join(self.output_dir, "GENERATION_REPORT.txt")
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("LONG TEST SAMPLES GENERATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            f.write("GENERATION PARAMETERS:\n")
            f.write(f"  • Target Duration: {self.target_duration} seconds ({self.target_duration/60:.1f} minutes)\n")
            f.write(f"  • Sample Rate: {self.sample_rate} Hz\n")
            f.write(f"  • Number of Samples: {len(self.metadata)}\n\n")
            
            f.write("SAMPLE DETAILS:\n")
            for m in self.metadata:
                f.write(f"\nSample {m['sample_id']:03d}: {m['output_file']}\n")
                f.write(f"  Duration: {m['actual_duration']:.1f} seconds ({m['actual_duration']/60:.2f} minutes)\n")
                f.write(f"  Created from: {m['num_recordings_used']} Xeno-Canto recordings\n")
                f.write(f"  Source Recordings:\n")
                for rec in m['source_recordings']:
                    f.write(f"    - {rec['filename']} (XC{rec['xc_id']}) - {rec['duration']:.1f}s\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("STATISTICS:\n")
            f.write("="*70 + "\n")
            f.write(f"Total Samples Created: {len(self.metadata)}\n")
            f.write(f"Total Audio Duration: {sum(m['actual_duration'] for m in self.metadata):.1f} seconds\n")
            f.write(f"Total Audio Duration: {sum(m['actual_duration'] for m in self.metadata)/60:.1f} minutes\n")
            f.write(f"Average Duration/Sample: {np.mean([m['actual_duration'] for m in self.metadata]):.1f} seconds\n")
            f.write(f"Total Unique Recordings Used: {sum(m['num_recordings_used'] for m in self.metadata)}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("NEXT STEPS:\n")
            f.write("="*70 + "\n")
            f.write("1. All test samples are in WAV format (22050 Hz sample rate)\n")
            f.write("2. Samples are named: test_composite_XYZ.wav\n")
            f.write("3. Use these for model testing/evaluation\n")
            f.write("4. Metadata available in: test_samples_metadata.json\n")
            f.write("5. Manifest CSV available in: test_samples_manifest.csv\n")
        
        print(f"✓ Report saved to: {report_file}\n")


def main():
    print("\n" + "="*70)
    print("LONG TEST SAMPLE GENERATOR")
    print("Concatenating unused XC recordings into 5+ minute test samples")
    print("="*70 + "\n")
    
    # Initialize generator
    generator = LongTestSampleGenerator(
        audio_dir=AUDIO_DIR,
        output_dir=OUTPUT_DIR,
        target_duration=TARGET_DURATION,
        sample_rate=SAMPLE_RATE
    )
    
    # Load unused IDs
    print("📂 Loading unused recording IDs...")
    unused_ids = generator.load_unused_ids(UNUSED_IDS_FILE)
    
    if not unused_ids:
        print("⚠ No unused IDs found. Run find_unused_recordings.py first.")
        return
    
    print(f"✓ Loaded {len(unused_ids)} unused recording IDs\n")
    
    # Generate samples
    metadata = generator.generate_samples(unused_ids, min_samples=MIN_SAMPLES)
    
    # Save metadata and reports
    generator.save_metadata()
    generator.generate_usage_report()
    
    print("\n" + "="*70)
    print("✓ TEST SAMPLE GENERATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Created {len(metadata)} long test samples in:")
    print(f"   {OUTPUT_DIR}\n")
    print("📄 Check GENERATION_REPORT.txt for detailed information\n")


if __name__ == "__main__":
    main()
