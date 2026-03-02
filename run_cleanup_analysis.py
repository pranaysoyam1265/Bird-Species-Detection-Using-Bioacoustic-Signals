"""
BirdSense Cleanup Analysis - Non-Interactive Version
Generates cleanup reports and scripts automatically.
"""

import sys
import os
import io

# Set UTF-8 encoding for output
if sys.stdout.encoding.lower() == 'cp1252':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from cleanup_scanner import scan_project, print_report, save_report, generate_cleanup_script

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  BIRDSENSE PROJECT CLEANUP ANALYSIS - AUTO MODE")
    print("="*70)
    
    try:
        # Scan project
        results = scan_project()
        
        # Print report
        print_report(results)
        
        # Save detailed report
        report_path = save_report(results)
        
        # Generate cleanup script
        script_path = generate_cleanup_script(results)
        
        print(f"\n{'='*70}")
        print(f"  CLEANUP ANALYSIS COMPLETE")
        print(f"{'='*70}\n")
        
        print(f"Generated Files:")
        print(f"   1. cleanup_report.json    - Detailed analysis report")
        print(f"   2. cleanup_script.py      - Interactive cleanup script\n")
        
        print(f"NEXT STEPS:")
        print(f"   1. Review cleanup_report.json")
        print(f"   2. Execute: python cleanup_script.py")
        print(f"   3. Type 'yes' to confirm deletions")
        print(f"   4. Test your application after cleanup\n")
        
        print(f"Warning: Always ensure you have backups before running cleanup!")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
