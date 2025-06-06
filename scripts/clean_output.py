#!/usr/bin/env python3
"""
Script to clean up hardcoded messages and emoji from FluxMD output
"""

import re
import sys

def clean_fluxmd_output(file_path):
    """Remove emoji and clean up output messages"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace emoji with clean markers
    replacements = {
        '⏱️': '[TIME]',
        '📊': '[STATS]',
        '✓': '[OK]',
        '❌': '[ERROR]',
        '💡': '[TIP]',
        '🔧': '[INFO]',
        '🎨': '[VIZ]',
        '💫': '',
        '🚀': '',
        '🌀': '',
        '⚡️': '',
        '✅': '[DONE]',
        '⚠️': '[WARNING]',
    }
    
    for emoji, replacement in replacements.items():
        content = content.replace(emoji, replacement)
    
    # Remove any decorative banners
    content = re.sub(r'═{40,}', '-' * 60, content)
    content = re.sub(r'#{40,}', '-' * 60, content)
    
    # Write cleaned content
    output_path = file_path.replace('.py', '_clean.py')
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"Cleaned file written to: {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        clean_fluxmd_output(sys.argv[1])
    else:
        # Clean main files
        for file in ['fluxmd.py', 'fluxmd_uma.py']:
            try:
                clean_fluxmd_output(file)
            except FileNotFoundError:
                print(f"File not found: {file}")