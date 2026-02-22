"""
Rename BirdCLEF Folder for Clarity
==================================
"""

import os
import shutil
import time
import subprocess

# ============================================================
# CONFIGURATION
# ============================================================

BASE_FOLDER = r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL"

# Current folder name
OLD_PATH = os.path.join(BASE_FOLDER, "01_Raw_Data", "External_Data", "BirdCLEF_Spectrograms")

# New folder name (clearly marked as external/unused)
NEW_PATH = os.path.join(BASE_FOLDER, "01_Raw_Data", "External_Data", "BirdCLEF_Spectrograms_NOT_USED_28975_images")

# ============================================================
# RENAME
# ============================================================

def rename_folder():
    print("=" * 60)
    print("📁 RENAMING BIRDCLEF FOLDER")
    print("=" * 60)
    
    if os.path.exists(OLD_PATH):
        try:
            # Use PowerShell Move-Item for better OneDrive handling
            ps_command = f'Move-Item -Path "{OLD_PATH}" -Destination "{NEW_PATH}" -Force'
            result = subprocess.run(
                ["powershell", "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"\n✅ Renamed successfully!")
                print(f"\n   From: BirdCLEF_Spectrograms/")
                print(f"   To:   BirdCLEF_Spectrograms_NOT_USED_28975_images/")
                print(f"\n📍 Location: 01_Raw_Data/External_Data/")
            else:
                print(f"\n❌ PowerShell Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"\n❌ Operation timed out (folder may be very large)")
            print(f"   Try manually renaming the folder in Windows Explorer")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    elif os.path.exists(NEW_PATH):
        print(f"\n⚪ Already renamed to new name")
    
    else:
        print(f"\n❌ Folder not found at expected location")
        print(f"   Looking for: {OLD_PATH}")
    
    print("\n" + "=" * 60)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    rename_folder()