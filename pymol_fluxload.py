#!/usr/bin/env python
"""
pymol_fluxload.py - Simple flux loading command for PyMOL

This is a simplified version that works better with PyMOL's command parsing.

Usage:
    run pymol_fluxload.py
    fload pdb_path csv_path [label]
"""

from pymol import cmd
import os
import csv

# Berlin color palette
BERLIN_HEX = [
    '#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
    '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f'
]

def fload(*args):
    """
    Simple flux loader that works with PyMOL's parsing
    
    Usage: fload pdb_path csv_path [label]
    """
    if len(args) < 2:
        print("Usage: fload pdb_path csv_path [label]")
        print("Example: fload /path/to/protein.pdb /path/to/flux.csv WT")
        return
    
    pdb_file = args[0]
    csv_file = args[1]
    label = args[2] if len(args) > 2 else os.path.basename(pdb_file).replace('.pdb', '')
    
    # Check files
    if not os.path.exists(pdb_file):
        print(f"Error: PDB not found: {pdb_file}")
        return
    if not os.path.exists(csv_file):
        print(f"Error: CSV not found: {csv_file}")
        return
    
    # Load PDB
    cmd.load(pdb_file, label)
    
    # Hide non-protein
    cmd.hide('everything', f'{label} and (solvent or inorganic or not polymer)')
    
    # Register colors
    colors = []
    for i, hx in enumerate(BERLIN_HEX):
        rgb = tuple(int(hx[j:j+2], 16)/255.0 for j in (1, 3, 5))
        cname = f"flux_{i}"
        cmd.set_color(cname, rgb)
        colors.append(cname)
    
    # Load flux data
    flux_data = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'residue_index' in row and 'average_flux' in row:
                try:
                    res_id = int(float(row['residue_index']))
                    flux_val = float(row['average_flux'])
                    flux_data[res_id] = flux_val
                except:
                    pass
    
    if not flux_data:
        print("No flux data found!")
        return
    
    # Color by flux
    v_min = min(flux_data.values())
    v_max = max(flux_data.values())
    span = v_max - v_min if v_max != v_min else 1.0
    
    for res_id, flux_val in flux_data.items():
        norm = (flux_val - v_min) / span
        idx = int(norm * 9.999)
        idx = max(0, min(9, idx))
        cmd.color(colors[idx], f"{label} and resi {res_id}")
    
    # Show cartoon
    cmd.show('cartoon', f'{label} and polymer')
    cmd.bg_color('white')
    
    print(f"Loaded {label}: flux range {v_min:.3f} to {v_max:.3f}")

cmd.extend("fload", fload)

print("""
Simple flux loader ready!
Usage: fload pdb_path csv_path [label]

Example:
fload /Users/myunghyun/Desktop/proteins/GPX4.pdb /Users/myunghyun/Desktop/flux.csv WT
""")