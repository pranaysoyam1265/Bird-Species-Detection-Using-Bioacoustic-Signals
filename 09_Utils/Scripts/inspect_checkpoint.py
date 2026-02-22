# Save as: inspect_checkpoint.py

import torch
from pathlib import Path

print("="*60)
print("🔍 CHECKPOINT INSPECTION")
print("="*60)

# Load checkpoint - handle different working directories
from pathlib import Path
import os

# Try multiple path options
checkpoint_paths = [
    Path('05_Model/Saved_Models/best_model_v3.pth'),
    Path(__file__).parent.parent.parent / '05_Model/Saved_Models/best_model_v3.pth',
]

checkpoint_path = None
for path in checkpoint_paths:
    if path.exists():
        checkpoint_path = path
        break

if not checkpoint_path:
    print(f"❌ Checkpoint not found in any of these locations:")
    for path in checkpoint_paths:
        print(f"   {path.resolve()}")
    exit(1)

print(f"\n✅ Loading: {checkpoint_path.resolve()}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("\n📋 CHECKPOINT KEYS:")
print("-" * 40)
for key in checkpoint.keys():
    if key == 'model_state_dict':
        print(f"  {key}: (state dict with {len(checkpoint[key])} layers)")
    else:
        print(f"  {key}: {checkpoint[key]}")

print("\n📋 MODEL STATE DICT KEYS:")
print("-" * 40)

state_dict = checkpoint['model_state_dict']

# Group keys by prefix
from collections import defaultdict
prefixes = defaultdict(list)
for key in state_dict.keys():
    prefix = key.split('.')[0]
    prefixes[prefix].append(key)

for prefix, keys in prefixes.items():
    print(f"\n{prefix}/ ({len(keys)} keys)")
    # Show first few keys and shapes
    for key in keys[:3]:
        shape = state_dict[key].shape
        print(f"  {key}: {shape}")
    if len(keys) > 3:
        print(f"  ... and {len(keys)-3} more")

# Check classifier structure specifically
print("\n" + "="*60)
print("🎯 CLASSIFIER LAYER DETAILS:")
print("="*60)

classifier_keys = [k for k in state_dict.keys() if 'classifier' in k or 'fc' in k or 'head' in k]
for key in classifier_keys:
    print(f"  {key}: {state_dict[key].shape}")

# Check last layer output size (should be 87)
for key in state_dict.keys():
    if state_dict[key].dim() >= 1:
        if state_dict[key].shape[0] == 87:  # num_classes
            print(f"\n✅ Found output layer: {key} with shape {state_dict[key].shape}")