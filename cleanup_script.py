"""
BirdSense Project Cleanup Script - INTERACTIVE
Generated: 2026-02-25T11:13:23.190093

WARNING: This script PERMANENTLY DELETES files.
Review the list carefully before confirming.

SAFETY FEATURES:
- Requires explicit "yes" confirmation
- Never touches production files
- Reports all deletions
- Can be run multiple times safely
"""

import os
import shutil
from pathlib import Path
import json

PROJECT_ROOT = Path(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")

ITEMS_TO_DELETE = [
    # CACHE (6 items)
    r"__pycache__",  # 27.33 KB
    r"08_Deployment/__pycache__",  # 3.72 KB
    r"08_Deployment/Streamlit_Version/__pycache__",  # 3.77 KB
    r"09_Utils/Scripts/__pycache__",  # 73.78 KB
    r"09_Utils/Scripts/__pycache__/generate_report.cpython-311.pyc",  # 73.78 KB
    r"__pycache__/cleanup_scanner.cpython-311.pyc",  # 27.33 KB
    # OLD_VERSIONS (7 items)
    r"04_Labels/Train_Val_Test_Split/test_v2.csv",  # 856.70 KB
    r"04_Labels/Train_Val_Test_Split/train_v2.csv",  # 3.90 MB
    r"04_Labels/Train_Val_Test_Split/val_v2.csv",  # 856.68 KB
    r"05_Model/Saved_Models/best_model_v2.pth",  # 50.24 MB
    r"09_Utils/Scripts/metadata_diagnostic_v2.py",  # 13.11 KB
    r"09_Utils/Scripts/remaining_scraper_v2.py",  # 18.98 KB
    r"09_Utils/Scripts/train_with_augmentation_v2.py",  # 15.53 KB
    # DUPLICATE_MODELS (7 items)
    r"05_Model/Checkpoints/checkpoint_epoch_10.pth",  # 97.66 MB
    r"05_Model/Checkpoints/checkpoint_epoch_15.pth",  # 97.66 MB
    r"05_Model/Checkpoints/checkpoint_epoch_20.pth",  # 97.66 MB
    r"05_Model/Checkpoints/checkpoint_epoch_5.pth",  # 97.66 MB
    r"05_Model/Saved_Models/checkpoint_epoch_10.pth",  # 50.26 MB
    r"05_Model/Saved_Models/checkpoint_epoch_20.pth",  # 50.26 MB
    r"05_Model/Saved_Models/checkpoint_epoch_5.pth",  # 50.24 MB
]

def format_size(size_bytes):
    """Format bytes to human readable."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

def cleanup():
    """Delete unnecessary files with confirmation."""
    
    print("\n" + "="*70)
    print("  🧹 BIRDSENSE PROJECT CLEANUP")
    print("="*70 + "\n")
    
    print(f"Items to delete: {len(ITEMS_TO_DELETE)}")
    
    # Calculate total size
    total_size = 0
    for item in ITEMS_TO_DELETE:
        path = PROJECT_ROOT / item
        if path.exists():
            if path.is_dir():
                try:
                    total_size += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                except:
                    pass
            else:
                try:
                    total_size += path.stat().st_size
                except:
                    pass
    
    print(f"Total space to free: {format_size(total_size)}\n")
    
    # Confirmation
    print("Items that will be deleted:")
    for item in sorted(ITEMS_TO_DELETE)[:20]:
        print(f"  - {item}")
    if len(ITEMS_TO_DELETE) > 20:
        print(f"  ... and {len(ITEMS_TO_DELETE) - 20} more")
    
    print("\nWARNING: This action CANNOT be undone!")
    response = input("\nType 'yes' to confirm deletion (or anything to cancel): ")
    
    if response.lower() != "yes":
        print("\n[CANCELLED] Cleanup cancelled. No files deleted.")
        return
    
    # Perform cleanup
    print("\n[DELETING] Starting deletion...")
    deleted_count = 0
    deleted_size = 0
    errors = []
    
    for item in ITEMS_TO_DELETE:
        path = PROJECT_ROOT / item
        if path.exists():
            try:
                if path.is_dir():
                    try:
                        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                    except:
                        size = 0
                    shutil.rmtree(path)
                    print(f"[OK] Deleted: {item}")
                else:
                    size = path.stat().st_size
                    path.unlink()
                    print(f"[OK] Deleted: {item}")
                deleted_count += 1
                deleted_size += size
            except Exception as e:
                errors.append((item, str(e)))
                print(f"[ERROR] Failed: {item} - {e}")
    
    # Report results
    line_sep = "=" * 70
    print(f"\n{{line_sep}}")
    print(f"  [OK] CLEANUP COMPLETE")
    print(f"{{line_sep}}\n")
    print(f"Items deleted: {deleted_count}")
    print(f"Space freed: {format_size(deleted_size)}")
    if errors:
        print(f"Errors: {len(errors)}")
    
    # Save cleanup log
    log_data = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "items_deleted": deleted_count,
        "space_freed": deleted_size,
        "errors": errors,
    }
    
    log_path = PROJECT_ROOT / "cleanup_log.json"
    with open(log_path, "w") as f:
        __import__("json").dump(log_data, f, indent=2)
    print(f"[OK] Log saved: cleanup_log.json")

if __name__ == "__main__":
    try:
        cleanup()
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Cleanup cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] Error during cleanup: {e}")