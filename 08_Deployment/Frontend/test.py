#!/usr/bin/env python3
"""
Template Directory Scanner & Code Documenter
Scans entire project directory, maps structure, 
and captures all code files into one reference document.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# ============================================
# CONFIGURATION - CHANGE THIS PATH
# ============================================
# Set this to your template's root directory
TEMPLATE_DIR = r"."  # Current directory, or paste your path like:
# TEMPLATE_DIR = r"C:\Users\YourName\Projects\brutalist-template"
# TEMPLATE_DIR = r"/home/user/projects/brutalist-template"

# Output file name
OUTPUT_FILE = "template_reference_document.md"

# File extensions to capture code from
CODE_EXTENSIONS = {
    # Web/Frontend
    '.tsx', '.ts', '.jsx', '.js',
    '.css', '.scss', '.sass', '.less',
    '.html', '.htm',
    # Config
    '.json', '.yaml', '.yml', '.toml',
    '.env', '.env.local', '.env.example',
    # Next.js / React specific
    '.mjs', '.cjs',
    # Markdown
    '.md', '.mdx',
    # Other
    '.svg', '.xml',
    '.prisma', '.graphql', '.gql',
    '.sh', '.bash',
    '.py',
    '.txt',
}

# Directories to SKIP
SKIP_DIRS = {
    'node_modules',
    '.next',
    '.git',
    'dist',
    'build',
    '.cache',
    '.vercel',
    '.turbo',
    '__pycache__',
    '.svn',
    'coverage',
    '.nyc_output',
    '.parcel-cache',
    'out',
    '.contentlayer',
    '.swc',
}

# Files to SKIP
SKIP_FILES = {
    'package-lock.json',
    'yarn.lock',
    'pnpm-lock.yaml',
    'bun.lockb',
    '.DS_Store',
    'Thumbs.db',
    '.gitkeep',
}

# Max file size to read (skip huge files)
MAX_FILE_SIZE_KB = 500  # Skip files larger than 500KB


def get_file_icon(ext):
    """Return emoji icon based on file extension."""
    icons = {
        '.tsx': '⚛️',
        '.ts': '🔷',
        '.jsx': '⚛️',
        '.js': '🟨',
        '.css': '🎨',
        '.scss': '🎨',
        '.html': '🌐',
        '.json': '📋',
        '.yaml': '📋',
        '.yml': '📋',
        '.md': '📝',
        '.mdx': '📝',
        '.svg': '🖼️',
        '.env': '🔒',
        '.toml': '⚙️',
        '.prisma': '🗄️',
        '.graphql': '◼️',
        '.py': '🐍',
        '.sh': '🐚',
        '.txt': '📄',
        '.mjs': '🟨',
        '.cjs': '🟨',
    }
    return icons.get(ext, '📄')


def get_language(ext):
    """Return language identifier for markdown code blocks."""
    languages = {
        '.tsx': 'tsx',
        '.ts': 'typescript',
        '.jsx': 'jsx',
        '.js': 'javascript',
        '.css': 'css',
        '.scss': 'scss',
        '.html': 'html',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.md': 'markdown',
        '.mdx': 'mdx',
        '.svg': 'xml',
        '.xml': 'xml',
        '.env': 'bash',
        '.toml': 'toml',
        '.prisma': 'prisma',
        '.graphql': 'graphql',
        '.py': 'python',
        '.sh': 'bash',
        '.txt': 'text',
        '.mjs': 'javascript',
        '.cjs': 'javascript',
    }
    return languages.get(ext, 'text')


def format_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def scan_directory(root_path):
    """
    Scan directory and return structure and file contents.
    Returns: (tree_lines, file_data_list)
    """
    tree_lines = []
    file_data = []
    file_count = 0
    dir_count = 0
    total_size = 0
    skipped_files = []

    root = Path(root_path).resolve()

    if not root.exists():
        print(f"❌ ERROR: Directory '{root}' does not exist!")
        print(f"   Please update TEMPLATE_DIR in the script.")
        sys.exit(1)

    print(f"📂 Scanning: {root}")
    print(f"{'=' * 60}")

    for dirpath, dirnames, filenames in os.walk(root):
        # Remove skipped directories (modifying in-place affects os.walk)
        dirnames[:] = [
            d for d in sorted(dirnames)
            if d not in SKIP_DIRS and not d.startswith('.')
        ]

        rel_dir = Path(dirpath).relative_to(root)
        depth = len(rel_dir.parts)

        # Add directory to tree
        if depth == 0:
            tree_lines.append(f"📦 {root.name}/")
        else:
            indent = "│   " * (depth - 1) + "├── "
            tree_lines.append(f"{indent}📁 {rel_dir.name}/")
            dir_count += 1

        # Process files
        sorted_files = sorted(filenames)
        for i, filename in enumerate(sorted_files):
            if filename in SKIP_FILES:
                continue
            if filename.startswith('.') and filename not in ('.env', '.env.local', '.env.example'):
                continue

            filepath = Path(dirpath) / filename
            rel_path = filepath.relative_to(root)
            ext = filepath.suffix.lower()
            file_size = filepath.stat().st_size

            # Tree line
            is_last = (i == len(sorted_files) - 1)
            indent = "│   " * depth
            connector = "└── " if is_last else "├── "
            icon = get_file_icon(ext)
            size_str = format_size(file_size)
            tree_lines.append(f"{indent}{connector}{icon} {filename} ({size_str})")

            file_count += 1
            total_size += file_size

            # Read file content if it's a code file
            if ext in CODE_EXTENSIONS:
                if file_size > MAX_FILE_SIZE_KB * 1024:
                    skipped_files.append((str(rel_path), "Too large"))
                    file_data.append({
                        'path': str(rel_path),
                        'ext': ext,
                        'size': file_size,
                        'content': f"[FILE TOO LARGE - {size_str}]",
                        'language': get_language(ext),
                        'skipped': True
                    })
                else:
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                        file_data.append({
                            'path': str(rel_path),
                            'ext': ext,
                            'size': file_size,
                            'content': content,
                            'language': get_language(ext),
                            'skipped': False
                        })
                        print(f"  ✅ Read: {rel_path}")
                    except Exception as e:
                        skipped_files.append((str(rel_path), str(e)))
                        file_data.append({
                            'path': str(rel_path),
                            'ext': ext,
                            'size': file_size,
                            'content': f"[ERROR READING FILE: {e}]",
                            'language': get_language(ext),
                            'skipped': True
                        })
                        print(f"  ⚠️ Error: {rel_path} - {e}")

    stats = {
        'total_files': file_count,
        'total_dirs': dir_count,
        'total_size': total_size,
        'code_files': len(file_data),
        'skipped': skipped_files
    }

    return tree_lines, file_data, stats


def categorize_files(file_data):
    """Group files by category for organized output."""
    categories = {
        '⚙️ Configuration': [],
        '📐 Layout & Pages': [],
        '🧩 Components': [],
        '🎨 Styles': [],
        '🔧 Utilities & Libs': [],
        '🪝 Hooks': [],
        '📊 Types & Interfaces': [],
        '🌐 API Routes': [],
        '📄 Content & Docs': [],
        '🖼️ Assets': [],
        '📦 Other': [],
    }

    for fd in file_data:
        path = fd['path'].replace('\\', '/')
        ext = fd['ext']

        if any(cfg in path for cfg in [
            'package.json', 'tsconfig', 'next.config',
            'tailwind.config', 'postcss', '.eslint',
            'prettier', 'vite.config', '.env',
            'components.json'
        ]):
            categories['⚙️ Configuration'].append(fd)
        elif '/app/' in path or 'pages/' in path or 'layout' in path.lower() or 'page' in path.lower():
            categories['📐 Layout & Pages'].append(fd)
        elif '/components/' in path or '/ui/' in path:
            categories['🧩 Components'].append(fd)
        elif ext in ('.css', '.scss', '.sass', '.less'):
            categories['🎨 Styles'].append(fd)
        elif '/lib/' in path or '/utils/' in path or '/helpers/' in path:
            categories['🔧 Utilities & Libs'].append(fd)
        elif '/hooks/' in path:
            categories['🪝 Hooks'].append(fd)
        elif '/types/' in path or '.d.ts' in path:
            categories['📊 Types & Interfaces'].append(fd)
        elif '/api/' in path:
            categories['🌐 API Routes'].append(fd)
        elif ext in ('.md', '.mdx', '.txt'):
            categories['📄 Content & Docs'].append(fd)
        elif ext in ('.svg', '.xml'):
            categories['🖼️ Assets'].append(fd)
        else:
            categories['📦 Other'].append(fd)

    return categories


def generate_document(tree_lines, file_data, stats, root_path):
    """Generate the complete reference document."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    categories = categorize_files(file_data)

    doc = []

    # ==========================================
    # HEADER
    # ==========================================
    doc.append("# 📂 Template Reference Document")
    doc.append("")
    doc.append("## Brutalist AI SaaS Landing Page - Complete Code Reference")
    doc.append("")
    doc.append(f"**Generated:** {timestamp}")
    doc.append(f"**Source:** `{root_path}`")
    doc.append(f"**Purpose:** Reference for converting to BirdCALL_AI application")
    doc.append("")
    doc.append("---")
    doc.append("")

    # ==========================================
    # STATISTICS
    # ==========================================
    doc.append("## 📊 Project Statistics")
    doc.append("")
    doc.append(f"| Metric | Value |")
    doc.append(f"|--------|-------|")
    doc.append(f"| Total Files | {stats['total_files']} |")
    doc.append(f"| Total Directories | {stats['total_dirs']} |")
    doc.append(f"| Code Files Captured | {stats['code_files']} |")
    doc.append(f"| Total Size | {format_size(stats['total_size'])} |")
    doc.append(f"| Skipped Files | {len(stats['skipped'])} |")
    doc.append("")

    # File type breakdown
    ext_count = {}
    for fd in file_data:
        ext = fd['ext']
        ext_count[ext] = ext_count.get(ext, 0) + 1

    if ext_count:
        doc.append("### File Type Breakdown")
        doc.append("")
        doc.append("| Extension | Count | Icon |")
        doc.append("|-----------|-------|------|")
        for ext, count in sorted(ext_count.items(), key=lambda x: -x[1]):
            doc.append(f"| `{ext}` | {count} | {get_file_icon(ext)} |")
        doc.append("")

    doc.append("---")
    doc.append("")

    # ==========================================
    # DIRECTORY TREE
    # ==========================================
    doc.append("## 🌳 Directory Tree")
    doc.append("")
    doc.append("```")
    for line in tree_lines:
        doc.append(line)
    doc.append("```")
    doc.append("")
    doc.append("---")
    doc.append("")

    # ==========================================
    # FILE CONTENTS BY CATEGORY
    # ==========================================
    doc.append("## 📝 Complete File Contents")
    doc.append("")
    doc.append("### Table of Contents")
    doc.append("")

    toc_index = 1
    for category, files in categories.items():
        if files:
            doc.append(f"- **{category}**")
            for fd in files:
                doc.append(f"  - [{fd['path']}](#{fd['path'].replace('/', '-').replace('.', '-').replace(' ', '-').lower()})")
            toc_index += 1
    doc.append("")
    doc.append("---")
    doc.append("")

    # Write each category
    for category, files in categories.items():
        if not files:
            continue

        doc.append(f"## {category}")
        doc.append("")

        for fd in files:
            path = fd['path']
            ext = fd['ext']
            size = format_size(fd['size'])
            language = fd['language']
            icon = get_file_icon(ext)
            content = fd['content']
            line_count = content.count('\n') + 1 if content else 0

            doc.append(f"### {icon} `{path}`")
            doc.append("")
            doc.append(f"- **Size:** {size}")
            doc.append(f"- **Lines:** {line_count}")
            doc.append(f"- **Type:** `{ext}`")
            doc.append("")

            if fd.get('skipped'):
                doc.append(f"> ⚠️ {content}")
            else:
                doc.append(f"```{language}")
                doc.append(content)
                if not content.endswith('\n'):
                    doc.append("")
                doc.append("```")

            doc.append("")
            doc.append("---")
            doc.append("")

    # ==========================================
    # SKIPPED FILES
    # ==========================================
    if stats['skipped']:
        doc.append("## ⚠️ Skipped Files")
        doc.append("")
        doc.append("| File | Reason |")
        doc.append("|------|--------|")
        for path, reason in stats['skipped']:
            doc.append(f"| `{path}` | {reason} |")
        doc.append("")

    # ==========================================
    # KEY OBSERVATIONS
    # ==========================================
    doc.append("## 🔍 Key Observations for Conversion")
    doc.append("")
    doc.append("### Important Files to Modify")
    doc.append("")
    doc.append("| File | What to Change |")
    doc.append("|------|---------------|")

    for fd in file_data:
        path = fd['path'].replace('\\', '/')
        if 'layout' in path.lower() and fd['ext'] in ('.tsx', '.jsx'):
            doc.append(f"| `{path}` | Main layout - add sidebar, modify header |")
        elif 'page' in path.lower() and fd['ext'] in ('.tsx', '.jsx'):
            doc.append(f"| `{path}` | Page content - replace with app content |")
        elif 'globals' in path.lower() and fd['ext'] in ('.css', '.scss'):
            doc.append(f"| `{path}` | Global styles - keep brutalist theme vars |")
        elif 'tailwind' in path.lower():
            doc.append(f"| `{path}` | Tailwind config - keep theme, add new utilities |")
        elif 'components.json' in path:
            doc.append(f"| `{path}` | shadcn/ui config - keep as is |")

    doc.append("")
    doc.append("### Brutalist Design Tokens to Preserve")
    doc.append("")
    doc.append("```")
    doc.append("Look for these patterns in the code above:")
    doc.append("- Border widths (border-2, border-4, etc.)")
    doc.append("- Font families (font-mono, monospace)")
    doc.append("- Shadow styles (shadow-brutal, offset shadows)")
    doc.append("- Color scheme (black, white, accent colors)")
    doc.append("- Corner radius (rounded-none, sharp corners)")
    doc.append("- Typography styles (uppercase, tracking, etc.)")
    doc.append("```")
    doc.append("")

    # ==========================================
    # FOOTER
    # ==========================================
    doc.append("---")
    doc.append("")
    doc.append(f"*Document generated on {timestamp}*")
    doc.append(f"*Scanner version: 1.0*")
    doc.append(f"*Total code files documented: {stats['code_files']}*")

    return '\n'.join(doc)


def main():
    """Main execution."""
    print("=" * 60)
    print("🔍 TEMPLATE DIRECTORY SCANNER")
    print("=" * 60)
    print()

    root_path = Path(TEMPLATE_DIR).resolve()

    print(f"📁 Target: {root_path}")
    print(f"📄 Output: {OUTPUT_FILE}")
    print(f"📋 Extensions: {len(CODE_EXTENSIONS)} types")
    print(f"🚫 Skipping dirs: {', '.join(sorted(SKIP_DIRS))}")
    print()

    # Scan
    tree_lines, file_data, stats = scan_directory(root_path)

    print()
    print(f"{'=' * 60}")
    print(f"📊 SCAN COMPLETE")
    print(f"{'=' * 60}")
    print(f"  📁 Directories: {stats['total_dirs']}")
    print(f"  📄 Total files: {stats['total_files']}")
    print(f"  📝 Code files:  {stats['code_files']}")
    print(f"  💾 Total size:  {format_size(stats['total_size'])}")
    print(f"  ⚠️  Skipped:    {len(stats['skipped'])}")
    print()

    # Generate document
    print("📝 Generating reference document...")
    document = generate_document(tree_lines, file_data, stats, root_path)

    # Write output
    output_path = Path(OUTPUT_FILE)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(document)

    output_size = output_path.stat().st_size
    print(f"✅ Document saved: {output_path.resolve()}")
    print(f"📄 Document size: {format_size(output_size)}")
    print()

    # Quick summary of what was captured
    print("📋 FILES CAPTURED:")
    print("-" * 40)
    for fd in file_data:
        status = "⚠️ SKIPPED" if fd.get('skipped') else "✅"
        print(f"  {status} {fd['path']}")

    print()
    print("🎯 NEXT STEPS:")
    print(f"  1. Open '{OUTPUT_FILE}' to review the template code")
    print(f"  2. Identify brutalist design patterns to keep")
    print(f"  3. Use the conversion prompts to transform the template")
    print()
    print("=" * 60)
    print("✨ DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()