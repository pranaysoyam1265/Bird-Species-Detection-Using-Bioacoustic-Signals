"""
Quick Diagnostic: Check Quality Column
"""

import pandas as pd

# Load the ORIGINAL CSV (before fix)
CSV_PATH = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL\bird_metadata_batch1_checkpoint.csv"

# Load data
df = pd.read_csv(CSV_PATH)

print("="*60)
print("🔍 QUALITY COLUMN DIAGNOSTIC")
print("="*60)

# Check if quality column exists
if 'quality' in df.columns:
    print("\n✅ 'quality' column exists")
    
    # Show unique values
    print("\n📊 Unique values in 'quality' column:")
    print("-"*40)
    unique_vals = df['quality'].unique()
    for val in unique_vals[:20]:  # Show first 20
        count = (df['quality'] == val).sum()
        print(f"   '{val}' : {count} records")
    
    # Show value counts
    print("\n📊 Value counts:")
    print("-"*40)
    print(df['quality'].value_counts(dropna=False).head(20))
    
    # Show sample values
    print("\n📋 Sample values (first 10 non-empty):")
    print("-"*40)
    non_empty = df[df['quality'].notna() & (df['quality'] != '')]
    for idx, row in non_empty.head(10).iterrows():
        print(f"   {row['recording_id']}: quality = '{row['quality']}'")
    
    # Check data type
    print(f"\n📌 Data type: {df['quality'].dtype}")
    
else:
    print("\n❌ 'quality' column NOT found")
    print("\n📋 Available columns:")
    for col in df.columns:
        print(f"   - {col}")