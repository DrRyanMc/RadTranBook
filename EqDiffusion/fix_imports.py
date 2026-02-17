#!/usr/bin/env python3
"""
Fix imports in all Python files to account for new directory structure.
Adds sys.path modification to find utils modules.
"""

import os
import re
from pathlib import Path

# Directories that need fixing
DIRS_TO_FIX = ['tests', 'debug', 'analysis', 'verification', 'problems']

# Pattern to match import statements
IMPORT_PATTERN = re.compile(r'^(from (oneDFV|twoDFV|plotfuncs) import)', re.MULTILINE)

# Header to add (will be inserted after docstring/shebang but before imports)
PATH_FIX = """import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

"""

def needs_fixing(content):
    """Check if file needs path fix"""
    # Check if it imports from our modules
    if not IMPORT_PATTERN.search(content):
        return False
    # Check if it already has the fix
    if 'sys.path.insert' in content and 'utils' in content:
        return False
    return True

def find_import_position(lines):
    """Find the position to insert the path fix (before first import)"""
    in_docstring = False
    docstring_char = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Handle docstrings
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                if stripped.count(docstring_char) == 1:  # Opening docstring
                    in_docstring = True
                # else it's a one-line docstring, skip
            elif stripped.startswith('#!'):  # Shebang
                continue
            elif stripped.startswith('#'):  # Comment
                continue
            elif not stripped:  # Empty line
                continue
            elif stripped.startswith('import ') or stripped.startswith('from '):
                # Found first import, insert before it
                return i
        else:
            # Inside docstring, look for closing
            if docstring_char in line:
                in_docstring = False
    
    return 0  # Fallback to beginning

def fix_file(filepath):
    """Add path fix to a single file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    if not needs_fixing(content):
        return False
    
    lines = content.split('\n')
    insert_pos = find_import_position(lines)
    
    # Insert the path fix
    lines.insert(insert_pos, PATH_FIX.rstrip())
    
    # Write back
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Fixed: {filepath}")
    return True

def main():
    base_dir = Path(__file__).parent
    fixed_count = 0
    
    for dirname in DIRS_TO_FIX:
        dir_path = base_dir / dirname
        if not dir_path.exists():
            continue
        
        for py_file in dir_path.glob('*.py'):
            if fix_file(py_file):
                fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")

if __name__ == '__main__':
    main()
