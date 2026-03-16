"""
Batch Test Script: Template for testing all 20 long samples with your model
"""

import os
import glob
import json
import csv
import numpy as np
import librosa
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Callable, Optional
import warnings
warnings.filterwarnings('ignore')


class BatchTestRunner:
    """Run batch evaluation on long test samples"""
    
    def __init__(self, test_dir: str = "10_Outputs/Test_Samples_Long",
                 output_dir: str = "10_Outputs/Test_Results"):
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.test_samples = sorted(glob.glob(os.path.join(test_dir, "test_composite_*.wav")))
        self.results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def run_test(self, model_predict_fn: Callable, 
                 model_name: str = "model",
                 use_chunks: bool = False,
                 chunk_duration: float = 5.0) -> List[Dict]:
        """
        Run batch test on all samples
        
        Args:
            model_predict_fn: Function that takes audio and returns prediction
                            Signature: func(audio, sr) -> prediction
            model_name: Name of model for logging
            use_chunks: If True, process sample in chunks and aggregate
            chunk_duration: Duration of chunks in seconds
        """
        
        print(f"\n{'='*80}")
        print(f"BATCH TEST RUNNER - {model_name}")
        print(f"{'='*80}\n")
        print(f"Testing {len(self.test_samples)} samples...")
        print(f"Output: {self.output_dir}\n")
        
        self.results = []
        
        for idx, sample_path in enumerate(self.test_samples, 1):
            print(f"[{idx:2d}/{len(self.test_samples)}] ", end="", flush=True)
            
            try:
                # Load audio
                audio, sr = librosa.load(sample_path, sr=22050)
                filename = os.path.basename(sample_path)
                
                # Get prediction
                if use_chunks:
                    prediction = self._predict_with_chunks(
                        audio, sr, model_predict_fn, chunk_duration
                    )
                else:
                    prediction = model_predict_fn(audio, sr)
                
                # Store result
                result = {
                    'sample_id': idx,
                    'filename': filename,
                    'duration': len(audio) / sr,
                    'prediction': prediction,
                    'status': 'success'
                }
                
                self.results.append(result)
                print(f"{filename} → {prediction}")
                
            except Exception as e:
                print(f"ERROR: {str(e)}")
                self.results.append({
                    'sample_id': idx,
                    'filename': filename,
                    'duration': 0,
                    'prediction': 'ERROR',
                    'status': 'failed',
                    'error': str(e)
                })
        
        self._save_results(model_name)
        return self.results
    
    def _predict_with_chunks(self, audio: np.ndarray, sr: int,
                            model_predict_fn: Callable,
                            chunk_duration: float) -> str:
        """Process audio in chunks and aggregate predictions"""
        chunk_samples = int(chunk_duration * sr)
        predictions = []
        
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start:start + chunk_samples]
            if len(chunk) < chunk_samples:
                break
            
            pred = model_predict_fn(chunk, sr)
            predictions.append(pred)
        
        # Return most common prediction
        from collections import Counter
        return Counter(predictions).most_common(1)[0][0]
    
    def _save_results(self, model_name: str):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON results
        json_file = os.path.join(self.output_dir, f"results_{model_name}_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved: {json_file}")
        
        # CSV results
        csv_file = os.path.join(self.output_dir, f"results_{model_name}_{timestamp}.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample_id', 'filename', 'duration', 'prediction', 'status'])
            writer.writeheader()
            writer.writerows(self.results)
        print(f"✓ CSV saved: {csv_file}")
        
        # Summary report
        self._generate_report(model_name, timestamp)
    
    def _generate_report(self, model_name: str, timestamp: str):
        """Generate test report"""
        report_file = os.path.join(self.output_dir, f"report_{model_name}_{timestamp}.txt")
        
        successful = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] == 'failed']
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"BATCH TEST REPORT - {model_name}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Test Set: 20 long samples (5 min each)\n\n")
            
            f.write("RESULTS SUMMARY:\n")
            f.write(f"  • Total Tests: {len(self.results)}\n")
            f.write(f"  • Successful: {len(successful)}\n")
            f.write(f"  • Failed: {len(failed)}\n")
            f.write(f"  • Success Rate: {len(successful)/len(self.results)*100:.1f}%\n\n")
            
            if successful:
                f.write("PREDICTIONS:\n")
                for r in successful:
                    f.write(f"  {r['filename']:<30} → {r['prediction']}\n")
            
            if failed:
                f.write("\nFAILURES:\n")
                for r in failed:
                    f.write(f"  {r['filename']:<30} → {r.get('error', 'Unknown error')}\n")
        
        print(f"✓ Report saved: {report_file}\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def dummy_model_predict(audio: np.ndarray, sr: int) -> str:
    """
    DUMMY MODEL - Replace with your actual model!
    
    This is a placeholder that returns random predictions.
    You need to implement your actual model prediction logic here.
    """
    
    # Extract a simple feature to simulate prediction
    rms = np.sqrt(np.mean(audio ** 2))
    
    # Dummy logic: return species based on RMS energy
    if rms > 0.08:
        return "Common_Blackbird"
    elif rms > 0.04:
        return "House_Sparrow"
    else:
        return "Quiet_Species"


def example_usage():
    """Example: How to use the batch test runner"""
    
    print("\n" + "="*80)
    print("BATCH TEST RUNNER - EXAMPLE USAGE")
    print("="*80 + "\n")
    
    # Initialize runner
    runner = BatchTestRunner(
        test_dir="10_Outputs/Test_Samples_Long",
        output_dir="10_Outputs/Test_Results"
    )
    
    # Run test with dummy model
    results = runner.run_test(
        model_predict_fn=dummy_model_predict,
        model_name="dummy_model",
        use_chunks=False  # Full sample prediction
    )
    
    print("\n✓ Batch test complete!")
    print(f"Results: {len(results)} samples processed")


# ============================================================================
# TEMPLATE FOR YOUR MODEL
# ============================================================================

"""
TEMPLATE: Replace the dummy model with your actual model

USE CASE 1: Run on full 5-minute sample
─────────────────────────────────────

def my_model_predict(audio, sr):
    # Your preprocessing here
    features = extract_features(audio, sr)
    
    # Your model inference here
    prediction = model.forward(features)
    
    return prediction.argmax()

runner = BatchTestRunner()
results = runner.run_test(
    model_predict_fn=my_model_predict,
    model_name="my_model_full_sample",
    use_chunks=False
)


USE CASE 2: Run on chunks and take majority vote
────────────────────────────────────────────────

def my_model_predict_chunk(chunk, sr):
    features = extract_features(chunk, sr)
    prediction = model.forward(features)
    return torch.argmax(prediction).item()

runner = BatchTestRunner()
results = runner.run_test(
    model_predict_fn=my_model_predict_chunk,
    model_name="my_model_chunked",
    use_chunks=True,
    chunk_duration=30  # 30-second chunks
)


USE CASE 3: Return confidence scores
─────────────────────────────────────

def my_model_predict_with_confidence(audio, sr):
    features = extract_features(audio, sr)
    logits = model.forward(features)
    pred_id = torch.argmax(logits).item()
    confidence = torch.softmax(logits, dim=0).max().item()
    
    return f"{class_names[pred_id]} ({confidence:.2%})"

runner = BatchTestRunner()
results = runner.run_test(
    model_predict_fn=my_model_predict_with_confidence,
    model_name="my_model_with_confidence"
)
"""


if __name__ == "__main__":
    # Run example
    example_usage()
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("""
1. Modify the 'dummy_model_predict' function OR
2. Create your own prediction function
3. Pass it to runner.run_test()

See the template above (TEMPLATE FOR YOUR MODEL) for examples!

Results will be saved as:
  • JSON: 10_Outputs/Test_Results/results_*.json
  • CSV:  10_Outputs/Test_Results/results_*.csv
  • TXT:  10_Outputs/Test_Results/report_*.txt
    """)
