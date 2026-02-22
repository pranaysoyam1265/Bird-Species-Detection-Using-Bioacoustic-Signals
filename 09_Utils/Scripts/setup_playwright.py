"""
Setup script for Arena Export Tool
Installs all dependencies including Playwright browsers
"""

import subprocess
import sys

print("="*70)
print("🎬 SETTING UP PLAYWRIGHT BROWSERS")
print("="*70)
print("\nThis installs Chromium, Firefox, and WebKit browsers (~1 GB, takes 2-5 min)")
print()

try:
    # Install browsers
    print("📥 Installing browsers...")
    result = subprocess.run([sys.executable, "-m", "playwright", "install"], 
                          capture_output=False)
    
    if result.returncode == 0:
        print("\n✅ SETUP COMPLETE!")
        print("\nYou can now run:")
        print("   python arena_export.py")
    else:
        print("\n❌ Setup failed. Try running manually:")
        print(f"   {sys.executable} -m playwright install")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nManual setup:")
    print(f"   {sys.executable} -m playwright install")
