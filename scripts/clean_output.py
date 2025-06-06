#!/usr/bin/env python3
"""
Clean decorative output from Python files
Removes emojis, box characters, and decorative elements
"""

import re
import sys

def clean_file(filename):
    """Clean decorative output from a Python file"""
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Replace unicode box characters
    box_chars = {
        '╔': '+', '╗': '+', '╚': '+', '╝': '+',
        '║': '|', '═': '-', '├': '|', '┤': '|',
        '└': '|', '┘': '|', '┌': '|', '┐': '|',
        '│': '|', '─': '-', '┼': '+', '┬': '+',
        '┴': '+', '├──': '  ', '└──': '  '
    }
    for old, new in box_chars.items():
        content = content.replace(old, new)
    
    # Replace emoji
    emoji_patterns = [
        (r'🔍', ''),
        (r'✓', '[OK]'),
        (r'✗', '[ERROR]'),
        (r'⚠️', '[WARNING]'),
        (r'🎯', ''),
        (r'💡', '[TIP]'),
        (r'📊', '[STATS]'),
        (r'🖥️', '[SYSTEM]'),
    ]
    for pattern, replacement in emoji_patterns:
        content = re.sub(pattern, replacement, content)
    
    # Replace bullet points
    content = content.replace('•', '-')
    
    # Replace special characters
    content = content.replace('→', ' to ')
    content = content.replace('Å', 'Angstroms')
    content = content.replace('π', 'pi')
    
    # Replace Korean text
    content = content.replace('합벡터', 'combined vector')
    
    # Replace status markers with cleaner format
    status_replacements = [
        (r'\[OK\]', ''),
        (r'\[ERROR\]', 'Error:'),
        (r'\[WARNING\]', 'Warning:'),
        (r'\[TIP\]', 'Tip:'),
        (r'\[INFO\]', 'Info:'),
        (r'\[STATS\]', 'Stats:'),
        (r'\[TIME\]', 'Time:'),
        (r'\[DONE\]', 'Done:'),
        (r'\[VIZ\]', 'Visualization:'),
    ]
    for pattern, replacement in status_replacements:
        content = re.sub(pattern, replacement, content)
    
    # Clean up colorama usage
    colorama_patterns = [
        r'from colorama import.*\n',
        r'init\(autoreset=True\)\n',
        r'{?Fore\.\w+}?',
        r'{?Style\.\w+}?',
    ]
    for pattern in colorama_patterns:
        content = re.sub(pattern, '', content)
    
    # Save if changed
    if content != original_content:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Cleaned: {filename}")
        return True
    else:
        print(f"No changes needed: {filename}")
        return False

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python clean_output.py <file1.py> [file2.py ...]")
        sys.exit(1)
    
    changed = 0
    for filename in sys.argv[1:]:
        if clean_file(filename):
            changed += 1
    
    print(f"\nCleaned {changed}/{len(sys.argv)-1} files")

if __name__ == "__main__":
    main()