"""
BirdSense Project Cleanup Scanner & Organizer
Scans project directory, categorizes files, and generates safe cleanup recommendations.

SAFETY FEATURES:
- Never touches production-critical files
- Generates interactive cleanup script with confirmation
- Provides detailed before/after analysis
- Creates backup report for reference

Generated: 2026-02-25
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import fnmatch

# ==================== CONFIGURATION ====================

PROJECT_ROOT = Path(r"C:\Users\prana\OneDrive\Desktop\ML Conf-BioFSL")

# File patterns to identify for each category
PATTERNS = {
    "cache": [
        "__pycache__",
        ".ipynb_checkpoints",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".scannerwork",
        "*.pyc",
        "*.pyo",
        ".next",
        ".nuxt",
        ".cache",
        "node_modules/.cache",
    ],
    "logs": [
        "*.log",
        "*.logs",
        "npm-debug.log*",
        "yarn-debug.log*",
        "yarn-error.log*",
        "lerna-debug.log*",
        "poetry.lock.log",
    ],
    "temp": [
        "*.tmp",
        "*.temp",
        "*.swp",
        "*.swo",
        "*~",
        ".tern-port",
        ".python-version",
        "*.tempfile",
    ],
    "os_files": [
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
        "ehthumbs.db",
        ".Spotlight-V100",
        ".Trashes",
        "$RECYCLE.BIN",
    ],
    "build_artifacts": [
        "node_modules",
        "dist",
        "build",
        ".next",
        "out",
        ".nuxt",
        ".cache",
        "*.egg-info",
        ".egg",
        ".eggs",
        "*.egg",
        "__pycache__",
        ".pytest_cache",
        ".tox",
        ".coverage",
        "htmlcov",
        ".mypy_cache",
        ".dmypy.json",
        "dmypy.json",
    ],
    "old_versions": [
        "*_old.*",
        "*_old",
        "*_backup.*",
        "*_backup",
        "*_bak.*",
        "*_bak",
        "*.bak",
        "*_v1.*",
        "*_v1",
        "*_v2.*",
        "*_v2",
        "*_copy.*",
        "*_copy",
        "*_archive.*",
        "Copy of *",
        "archive_*",
        "deprecated_*",
    ],
    "duplicate_models": [
        "model_v1.pth",
        "model_v2.pth",
        "best_model_v1.pth",
        "best_model_v2.pth",
        "checkpoint_*.pth",
        "*.ckpt",
        "*_old.pth",
        "*_backup.pth",
    ],
}

# Essential files - NEVER DELETE
ESSENTIAL_FILES = {
    # Production models
    "best_model_v3.pth",
    "best_model.pth",
    "final_model.pth",
    
    # Labels and mappings
    "label_mapping_v3.json",
    "label_mapping.json",
    "class_mapping.json",
    
    # Data splits
    "train_v3.csv",
    "val_v3.csv",
    "test_v3.csv",
    "train.csv",
    "val.csv",
    "test.csv",
    
    # Frontend config
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "tsconfig.json",
    "next.config.js",
    "next.config.mjs",
    "next.config.ts",
    "tailwind.config.js",
    "tailwind.config.ts",
    "tailwind.config.cjs",
    "postcss.config.js",
    "postcss.config.mjs",
    "postcss.config.ts",
    "vercel.json",
    
    # Environment and config
    ".env",
    ".env.local",
    ".env.production",
    ".env.production.local",
    ".env.development.local",
    ".gitignore",
    ".gitattributes",
    
    # Documentation
    "README.md",
    "README.txt",
    "LICENSE",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    
    # Important configs
    "jest.config.js",
    ".eslintrc.json",
    ".prettierrc",
    "babel.config.js",
    "webpack.config.js",
}

# Essential directories - NEVER DELETE
ESSENTIAL_DIRS = {
    "08_Deployment/Frontend",
    "08_Deployment/Backend",
    "08_Deployment/API",
    "08_Deployment/Streamlit_Version",
    "app",
    "components",
    "lib",
    "utils",
    "public",
    "styles",
    "src",
    "api",
    "hooks",
    "contexts",
    "config",
    "pages",
    "scripts",
}

# Large directories - Ask about deletion
LARGE_DIRS_TO_REVIEW = {
    "01_Raw_Data/Audio_Recordings": "Original source audio files",
    "02_Preprocessed/Standardized_Audio": "Processed audio (regenerable)",
    "02_Preprocessed/Audio_Chunks": "Audio chunks (regenerable)",
    "03_Features/Spectrograms": "Spectrograms (regenerable)",
    "05_Model/Checkpoints": "Training checkpoints",
    "05_Model/Training_Logs": "Training logs",
}

# ==================== UTILITY FUNCTIONS ====================

def get_size(path):
    """Get size of file or directory in bytes."""
    try:
        if path.is_file():
            return path.stat().st_size
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except (OSError, IOError):
                    pass
        return total
    except (OSError, IOError):
        return 0

def format_size(size_bytes):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

def matches_pattern(name, patterns):
    """Check if name matches any pattern (case-insensitive)."""
    name_lower = name.lower()
    for pattern in patterns:
        pattern_lower = pattern.lower()
        if fnmatch.fnmatch(name_lower, pattern_lower):
            return True
        if fnmatch.fnmatch(name, pattern):
            return True
        if name == pattern or name_lower == pattern_lower:
            return True
    return False

def is_essential(path):
    """Check if path is essential and should never be deleted."""
    name = path.name
    
    # Check essential files
    if name in ESSENTIAL_FILES:
        return True
    
    try:
        rel_path = str(path.relative_to(PROJECT_ROOT))
        
        # Check if starts with any essential directory
        for edir in ESSENTIAL_DIRS:
            if rel_path.startswith(edir) or edir in rel_path:
                return True
        
        # Special case for important subdirectories
        if any(x in rel_path for x in ['Frontend', 'Backend', 'API', 'components', 'pages', 'api']):
            return True
            
    except ValueError:
        pass
    
    return False

def is_in_large_directory(path, large_dirs):
    """Check if path is in a large directory that needs review."""
    try:
        rel_path = str(path.relative_to(PROJECT_ROOT))
        for large_dir in large_dirs.keys():
            if rel_path.startswith(large_dir):
                return large_dir
    except ValueError:
        pass
    return None

def count_files(path):
    """Count files in a directory."""
    try:
        if path.is_dir():
            return len(list(path.rglob("*")))
    except (OSError, IOError):
        pass
    return 0

# ==================== MAIN SCANNER ====================

def scan_project():
    """Scan project and categorize files for cleanup."""
    
    results = {
        "scan_date": datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "categories": defaultdict(list),
        "large_dirs": {},
        "summary": {
            "total_files": 0,
            "total_size": 0,
            "total_size_formatted": "",
            "deletable_files": 0,
            "deletable_size": 0,
            "deletable_size_formatted": "",
            "protected_files": 0,
            "protected_size": 0,
        }
    }
    
    print(f"\n{'='*70}")
    print(f"  BIRDSENSE PROJECT CLEANUP SCANNER")
    print(f"{'='*70}")
    print(f"\nScanning: {PROJECT_ROOT}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    scanned_dirs = 0
    scanned_files = 0
    
    for root, dirs, files in os.walk(PROJECT_ROOT):
        root_path = Path(root)
        
        # Skip git and common ignored directories
        dirs_to_remove = [d for d in dirs if d in {'.git', '.github', 'venv', '.venv', '.env'}]
        for d in dirs_to_remove:
            dirs.remove(d)
        
        scanned_dirs += 1
        
        # Check directories
        for d in dirs[:]:  # Copy to modify during iteration
            dir_path = root_path / d
            
            try:
                rel_path = str(dir_path.relative_to(PROJECT_ROOT))
            except ValueError:
                continue
            
            # Check each category
            for category, patterns in PATTERNS.items():
                if matches_pattern(d, patterns):
                    if not is_essential(dir_path):
                        size = get_size(dir_path)
                        file_count = count_files(dir_path)
                        
                        results["categories"][category].append({
                            "path": rel_path,
                            "type": "directory",
                            "size": size,
                            "size_formatted": format_size(size),
                            "file_count": file_count,
                        })
                        
                        results["summary"]["deletable_files"] += 1
                        results["summary"]["deletable_size"] += size
                    break
        
        # Check files
        for f in files:
            scanned_files += 1
            file_path = root_path / f
            
            try:
                rel_path = str(file_path.relative_to(PROJECT_ROOT))
            except ValueError:
                continue
            
            try:
                size = file_path.stat().st_size
                results["summary"]["total_size"] += size
            except (OSError, IOError):
                size = 0
            
            results["summary"]["total_files"] += 1
            
            # Skip essential files
            if is_essential(file_path):
                results["summary"]["protected_files"] += 1
                results["summary"]["protected_size"] += size
                continue
            
            # Check if in large directory needing review
            large_dir = is_in_large_directory(file_path, LARGE_DIRS_TO_REVIEW)
            if large_dir and large_dir not in results["large_dirs"]:
                size = get_size(Path(PROJECT_ROOT) / large_dir)
                file_count = count_files(Path(PROJECT_ROOT) / large_dir)
                results["large_dirs"][large_dir] = {
                    "size": size,
                    "size_formatted": format_size(size),
                    "file_count": file_count,
                    "description": LARGE_DIRS_TO_REVIEW[large_dir],
                }
            
            # Check each category
            for category, patterns in PATTERNS.items():
                if matches_pattern(f, patterns):
                    results["categories"][category].append({
                        "path": rel_path,
                        "type": "file",
                        "size": size,
                        "size_formatted": format_size(size),
                    })
                    
                    results["summary"]["deletable_files"] += 1
                    results["summary"]["deletable_size"] += size
                    break
    
    results["summary"]["total_size_formatted"] = format_size(results["summary"]["total_size"])
    results["summary"]["deletable_size_formatted"] = format_size(results["summary"]["deletable_size"])
    
    print(f"[OK] Scan complete!")
    print(f"   Directories scanned: {scanned_dirs:,}")
    print(f"   Files scanned: {scanned_files:,}")
    
    return results

# ==================== REPORT GENERATION ====================

def print_report(results):
    """Print detailed cleanup report to console."""
    
    print(f"\n{'='*70}")
    print(f"  📊 CLEANUP ANALYSIS REPORT")
    print(f"{'='*70}\n")
    
    # Summary statistics
    print(f"PROJECT STATISTICS")
    print(f"{'-'*70}")
    print(f"  Total files: {results['summary']['total_files']:,}")
    print(f"  Total size: {results['summary']['total_size_formatted']}")
    print(f"  Protected files (production): {results['summary']['protected_files']:,}")
    print(f"  Protected size: {format_size(results['summary']['protected_size'])}")
    
    print(f"\nCLEANUP OPPORTUNITIES")
    print(f"{'-'*70}")
    print(f"  Deletable items: {results['summary']['deletable_files']:,}")
    print(f"  Potential savings: {results['summary']['deletable_size_formatted']}")
    
    if results['summary']['deletable_size'] > 0:
        percentage = (results['summary']['deletable_size'] / results['summary']['total_size'] * 100) if results['summary']['total_size'] > 0 else 0
        print(f"  Percentage of total: {percentage:.1f}%")
    
    # Detailed breakdown by category
    print(f"\n{'='*70}")
    print(f"  FILES BY CATEGORY")
    print(f"{'='*70}\n")
    
    category_order = ['cache', 'logs', 'temp', 'os_files', 'build_artifacts', 'old_versions', 'duplicate_models']
    
    for category in category_order:
        items = results["categories"].get(category, [])
        if items:
            total_size = sum(item["size"] for item in items)
            category_display = category.upper().replace('_', ' ')
            
            print(f"\n[DELETE] {category_display}")
            print(f"   Count: {len(items)} items | Size: {format_size(total_size)}")
            print(f"   {'-'*65}")
            
            # Sort by size, show top N
            sorted_items = sorted(items, key=lambda x: x['size'], reverse=True)
            max_show = 8
            
            for item in sorted_items[:max_show]:
                icon = "[D]" if item["type"] == "directory" else "[F]"
                path_display = item['path']
                if len(path_display) > 50:
                    path_display = "..." + path_display[-47:]
                
                print(f"   {icon} {path_display}")
                
                if item["type"] == "directory" and item.get("file_count"):
                    print(f"      └─ {item['size_formatted']} ({item['file_count']} files)")
                else:
                    print(f"      └─ {item['size_formatted']}")
            
            if len(items) > max_show:
                print(f"   ... and {len(items) - max_show} more items")
    
    # Large directories review
    if results['large_dirs']:
        print(f"\n{'='*70}")
        print(f"  LARGE DIRECTORIES (Review Before Deletion)")
        print(f"{'='*70}\n")
        
        for dir_name, dir_info in sorted(results['large_dirs'].items(), 
                                        key=lambda x: x[1]['size'], 
                                        reverse=True):
            print(f"[DIR] {dir_name}")
            print(f"   Size: {dir_info['size_formatted']} | Files: {dir_info['file_count']:,}")
            print(f"   Purpose: {dir_info['description']}")
            print(f"   Status: [REVIEW] Can be regenerated if needed")
            print()
    
    # Safety notice
    print(f"\n{'='*70}")
    print(f"  SAFETY INFORMATION")
    print(f"{'='*70}\n")
    print(f"[PROTECTED] Never delete:")
    print(f"   • Production models (best_model_v3.pth)")
    print(f"   • Label mappings (label_mapping_v3.json)")
    print(f"   • Frontend code (08_Deployment/Frontend)")
    print(f"   • Backend code (08_Deployment/Backend)")
    print(f"   • Configuration files (.env, package.json, etc.)")
    print(f"\n[REVIEW] Before deleting:")
    print(f"   • Source audio data (01_Raw_Data)")
    print(f"   • Training scripts and notebooks")
    print(f"   • Data split files (train_v3.csv, etc.)")
    print(f"\n[OK] Safe to delete:")
    print(f"   • Cache and build artifacts")
    print(f"   • Log files")
    print(f"   • Old versions and backups")
    print(f"   • OS-specific files")

def save_report(results):
    """Save detailed report to JSON file."""
    report_path = PROJECT_ROOT / "cleanup_report.json"
    
    # Convert defaultdict to regular dict for JSON serialization
    report_data = {
        "scan_date": results["scan_date"],
        "project_root": results["project_root"],
        "categories": dict(results["categories"]),
        "large_dirs": results["large_dirs"],
        "summary": results["summary"],
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    print(f"\n[FILE] Report saved: {report_path}")
    return report_path

def generate_cleanup_script(results):
    """Generate interactive cleanup script."""
    
    deletable_items = []
    for category, items in results["categories"].items():
        deletable_items.extend(items)
    
    script_lines = [
        '"""',
        'BirdSense Project Cleanup Script - INTERACTIVE',
        'Generated: ' + datetime.now().isoformat(),
        '',
        'WARNING: This script PERMANENTLY DELETES files.',
        'Review the list carefully before confirming.',
        '',
        'SAFETY FEATURES:',
        '- Requires explicit "yes" confirmation',
        '- Never touches production files',
        '- Reports all deletions',
        '- Can be run multiple times safely',
        '"""',
        '',
        'import os',
        'import shutil',
        'from pathlib import Path',
        'import json',
        '',
        f'PROJECT_ROOT = Path(r"{PROJECT_ROOT}")',
        '',
        'ITEMS_TO_DELETE = [',
    ]
    
    # Group items by category
    for category in ['cache', 'logs', 'temp', 'os_files', 'build_artifacts', 'old_versions', 'duplicate_models']:
        items = results["categories"].get(category, [])
        if items:
            script_lines.append(f'    # {category.upper()} ({len(items)} items)')
            for item in items:
                safe_path = str(item['path']).replace('\\', '/')
                script_lines.append(f'    r"{safe_path}",  # {item["size_formatted"]}')
    
    script_lines.extend([
        ']',
        '',
        'def format_size(size_bytes):',
        '    """Format bytes to human readable."""',
        '    for unit in ["B", "KB", "MB", "GB", "TB"]:',
        '        if size_bytes < 1024:',
        '            return f"{size_bytes:.2f} {unit}"',
        '        size_bytes /= 1024',
        '    return f"{size_bytes:.2f} PB"',
        '',
        'def cleanup():',
        '    """Delete unnecessary files with confirmation."""',
        '    ',
        '    print("\\n" + "="*70)',
        '    print("  🧹 BIRDSENSE PROJECT CLEANUP")',
        '    print("="*70 + "\\n")',
        '    ',
        '    print(f"Items to delete: {len(ITEMS_TO_DELETE)}")',
        '    ',
        '    # Calculate total size',
        '    total_size = 0',
        '    for item in ITEMS_TO_DELETE:',
        '        path = PROJECT_ROOT / item',
        '        if path.exists():',
        '            if path.is_dir():',
        '                try:',
        '                    total_size += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())',
        '                except:',
        '                    pass',
        '            else:',
        '                try:',
        '                    total_size += path.stat().st_size',
        '                except:',
        '                    pass',
        '    ',
        '    print(f"Total space to free: {format_size(total_size)}\\n")',
        '    ',
        '    # Confirmation',
        '    print("Items that will be deleted:")',
        '    for item in sorted(ITEMS_TO_DELETE)[:20]:',
        '        print(f"  - {item}")',
        '    if len(ITEMS_TO_DELETE) > 20:',
        '        print(f"  ... and {len(ITEMS_TO_DELETE) - 20} more")',
        '    ',
        '    print("\\n⚠️  This action CANNOT be undone!")',
        '    response = input("\\nType \"yes\" to confirm deletion (or anything to cancel): ")',
        '    ',
        '    if response.lower() != "yes":',
        '        print("\\n❌ Cleanup cancelled. No files deleted.")',
        '        return',
        '    ',
        '    # Perform cleanup',
        '    print("\\n🗑️  Deleting files...")',
        '    deleted_count = 0',
        '    deleted_size = 0',
        '    errors = []',
        '    ',
        '    for item in ITEMS_TO_DELETE:',
        '        path = PROJECT_ROOT / item',
        '        if path.exists():',
        '            try:',
        '                if path.is_dir():',
        '                    try:',
        '                        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())',
        '                    except:',
        '                        size = 0',
        '                    shutil.rmtree(path)',
        '                    print(f"✅ Deleted: {{item}}")',
        '                else:',
        '                    size = path.stat().st_size',
        '                    path.unlink()',
        '                    print(f"✅ Deleted: {{item}}")',
        '                deleted_count += 1',
        '                deleted_size += size',
        '            except Exception as e:',
        '                errors.append((item, str(e)))',
        '                print(f"❌ Failed: {{item}} - {{e}}")',
        '    ',
        '    # Report results',
        '    line_sep = "=" * 70',
        '    print(f"\\n{{line_sep}}")',
        '    print(f"  ✅ CLEANUP COMPLETE")',
        '    print(f"{{line_sep}}\\n")',
        '    print(f"Items deleted: {deleted_count}")',
        '    print(f"Space freed: {format_size(deleted_size)}")',
        '    if errors:',
        '        print(f"Errors: {len(errors)}")',
        '    ',
        '    # Save cleanup log',
        '    log_data = {',
        '        "timestamp": __import__("datetime").datetime.now().isoformat(),',
        '        "items_deleted": deleted_count,',
        '        "space_freed": deleted_size,',
        '        "errors": errors,',
        '    }',
        '    ',
        '    log_path = PROJECT_ROOT / "cleanup_log.json"',
        '    with open(log_path, "w") as f:',
        '        __import__("json").dump(log_data, f, indent=2)',
        '    print(f"✅ Log saved: cleanup_log.json")',
        '',
        'if __name__ == "__main__":',
        '    try:',
        '        cleanup()',
        '    except KeyboardInterrupt:',
        '        print("\\n\\n⚠️  Cleanup cancelled by user.")',
        '    except Exception as e:',
        '        print(f"\\n❌ Error during cleanup: {e}")',
    ])
    
    script_path = PROJECT_ROOT / "cleanup_script.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(script_lines))
    
    print(f"[OK] Cleanup script generated: cleanup_script.py")
    return script_path

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    try:
        # Scan project
        results = scan_project()
        
        # Print report
        print_report(results)
        
        # Save detailed report
        save_report(results)
        
        # Generate cleanup script
        print(f"\n{'='*70}")
        response = input("\n✨ Generate interactive cleanup script? (yes/no): ").strip().lower()
        
        if response == 'yes':
            generate_cleanup_script(results)
            print(f"\n{'='*70}")
            print(f"  📝 NEXT STEPS:")
            print(f"{'='*70}")
            print(f"\n1. Review the cleanup_report.json file")
            print(f"2. Run the cleanup script: python cleanup_script.py")
            print(f"3. Review items and confirm deletion")
            print(f"\n⚠️  Important:")
            print(f"   • Always backup before cleanup")
            print(f"   • Review the deletion list carefully")
            print(f"   • Test your application after cleanup")
        else:
            print(f"\n✅ Report generated. Run script generation later if needed.")
        
        print(f"\n✅ Scan & analysis complete!")
        print(f"   Report: cleanup_report.json")
        print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Scan cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error during scan: {e}")
        import traceback
        traceback.print_exc()
