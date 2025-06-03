#!/usr/bin/env python
"""
pymol_colorflux.py - Color already loaded proteins by flux values

This script works with proteins already loaded in PyMOL.

Usage:
    run pymol_colorflux.py
    colorflux              # Interactive - asks for CSV for each loaded protein
    colorflux object_name csv_path  # Direct coloring of specific object
    colorflux all          # Color all loaded proteins (asks for CSVs)
"""

from pymol import cmd
import os
import csv

# Berlin color palette
BERLIN_HEX = [
    '#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
    '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f'
]

def _register_colors():
    """Register Berlin color palette"""
    colors = []
    for i, hx in enumerate(BERLIN_HEX):
        rgb = tuple(int(hx[j:j+2], 16)/255.0 for j in (1, 3, 5))
        cname = f"flux_{i}"
        cmd.set_color(cname, rgb)
        colors.append(cname)
    return colors

def _get_protein_objects():
    """Get list of loaded protein objects"""
    objects = []
    for obj in cmd.get_names('objects'):
        if cmd.count_atoms(f"{obj} and polymer") > 0:
            objects.append(obj)
    return objects

def _load_flux_data(csv_path):
    """Load flux data from CSV"""
    flux_data = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'residue_index' in row and 'average_flux' in row:
                    try:
                        res_id = int(float(row['residue_index']))
                        flux_val = float(row['average_flux'])
                        flux_data[res_id] = flux_val
                    except:
                        pass
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    return flux_data

def _color_by_flux(obj_name, flux_data, colors):
    """Apply flux coloring to an object"""
    if not flux_data:
        return False
    
    v_min = min(flux_data.values())
    v_max = max(flux_data.values())
    span = v_max - v_min if v_max != v_min else 1.0
    
    # Hide non-protein elements
    cmd.hide('everything', f'{obj_name} and (solvent or inorganic or not polymer)')
    
    # Color each residue
    for res_id, flux_val in flux_data.items():
        norm = (flux_val - v_min) / span
        idx = int(norm * 9.999)
        idx = max(0, min(9, idx))
        cmd.color(colors[idx], f"{obj_name} and resi {res_id}")
    
    # Show as cartoon
    cmd.show('cartoon', f'{obj_name} and polymer')
    
    print(f"[colorflux] {obj_name}: flux range {v_min:.3f} to {v_max:.3f}")
    return True

def colorflux(selection="all", csv_path=None):
    """
    Color loaded proteins by flux values
    
    Parameters:
    -----------
    selection : str
        "all" - color all loaded proteins (default)
        object_name - color specific object
    csv_path : str
        Path to CSV file (if coloring single object)
    """
    colors = _register_colors()
    
    # Get protein objects
    if selection == "all":
        objects = _get_protein_objects()
        if not objects:
            print("[colorflux] No protein objects found in PyMOL")
            return
        
        print(f"[colorflux] Found {len(objects)} protein(s): {', '.join(objects)}")
        print("\n[colorflux] Enter CSV file for each protein (or press Enter to skip):\n")
        
        colored_count = 0
        
        for obj in objects:
            # Ask for CSV file
            try:
                csv_file = cmd.get_raw_input(f"CSV file for {obj}: ")
            except:
                # Fallback to Python input if dialog fails
                csv_file = input(f"CSV file for {obj}: ")
            
            csv_file = csv_file.strip()
            
            if csv_file and os.path.exists(csv_file):
                flux_data = _load_flux_data(csv_file)
                if flux_data and _color_by_flux(obj, flux_data, colors):
                    colored_count += 1
            elif csv_file:
                print(f"  File not found: {csv_file}")
        
        # Set up grid view if multiple proteins
        if colored_count > 1:
            cmd.set('grid_mode', 1)
            print(f"\n[colorflux] Grid view enabled for {colored_count} proteins")
        
        # Set white background
        cmd.bg_color('white')
        cmd.zoom('visible')
        
    else:
        # Color specific object
        if csv_path is None:
            try:
                csv_path = cmd.get_raw_input(f"CSV file for {selection}: ")
            except:
                csv_path = input(f"CSV file for {selection}: ")
            csv_path = csv_path.strip()
        
        if not os.path.exists(csv_path):
            print(f"[colorflux] Error: CSV file not found: {csv_path}")
            return
        
        flux_data = _load_flux_data(csv_path)
        if flux_data:
            _color_by_flux(selection, flux_data, colors)
            cmd.bg_color('white')

# Register command
cmd.extend("colorflux", colorflux)

# Simpler shortcuts
def cflux():
    """Shortcut for colorflux all"""
    colorflux("all")

cmd.extend("cflux", cflux)

print("""
PyMOL ColorFlux loaded! Works with already loaded proteins.

Commands:
  colorflux              # Color all loaded proteins (asks for CSV files)
  colorflux object csv   # Color specific object with CSV
  cflux                  # Shortcut for colorflux

Example workflow:
  1. Load your proteins in PyMOL (File > Open or 'load' command)
  2. Run: colorflux
  3. Enter CSV file paths when prompted
  
The script will automatically set up grid view for multiple proteins.
""")