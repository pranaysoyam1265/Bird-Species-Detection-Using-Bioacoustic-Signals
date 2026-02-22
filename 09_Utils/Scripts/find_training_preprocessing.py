# Save as: find_training_preprocessing.py

import os
import glob

os.chdir(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")

print("="*60)
print("🔍 FINDING TRAINING PREPROCESSING METHOD")
print("="*60)

# Find the training script
training_scripts = glob.glob('**/*train*.py', recursive=True)
print(f"\nFound {len(training_scripts)} training scripts:")
for s in training_scripts:
    print(f"  {s}")

# Read the v3 training script
v3_script = None
for s in training_scripts:
    if 'v3' in s.lower() or 'optimized' in s.lower():
        v3_script = s
        break

if v3_script is None:
    v3_script = training_scripts[0] if training_scripts else None

if v3_script:
    print(f"\n📄 Reading: {v3_script}")
    print("="*60)
    
    with open(v3_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the Dataset class and preprocessing
    print("\n🔍 Looking for Dataset class and preprocessing...")
    
    # Search for key patterns
    patterns = [
        'class.*Dataset',
        'def __getitem__',
        'transform',
        'Normalize',
        'np.load',
        'spectrogram',
        'ToTensor',
        'def load_spectrogram',
        'def preprocess',
    ]
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        for pattern in patterns:
            if pattern.lower() in line.lower():
                # Print context (5 lines before and after)
                start = max(0, i - 2)
                end = min(len(lines), i + 10)
                print(f"\n--- Found '{pattern}' at line {i+1} ---")
                for j in range(start, end):
                    marker = ">>>" if j == i else "   "
                    print(f"{marker} {j+1:4d}: {lines[j]}")
                break

print("\n" + "="*60)
print("🔍 SEARCHING FOR SPECTROGRAM LOADING CODE")
print("="*60)

# Search more specifically
search_terms = ['__getitem__', 'np.load', 'torch.from_numpy']

in_getitem = False
getitem_lines = []

for i, line in enumerate(lines):
    if 'def __getitem__' in line:
        in_getitem = True
        getitem_lines = []
    
    if in_getitem:
        getitem_lines.append((i+1, line))
        if line.strip().startswith('return'):
            in_getitem = False
            print("\n📋 __getitem__ method:")
            for ln, l in getitem_lines:
                print(f"  {ln:4d}: {l}")
            break

# Also look for normalization values
print("\n🔍 Looking for normalization values...")
for i, line in enumerate(lines):
    if 'mean' in line.lower() and ('=' in line or ':' in line):
        if 'running_mean' not in line.lower():
            print(f"  {i+1:4d}: {line.strip()}")
    if 'std' in line.lower() and ('=' in line or ':' in line):
        if 'running' not in line.lower():
            print(f"  {i+1:4d}: {line.strip()}")