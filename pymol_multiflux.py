#!/usr/bin/env python
"""
pymol_multiflux.py - Color multiple proteins by flux values in grid view

Usage in PyMOL:
    run /path/to/pymol_multiflux.py
    multiflux              # Interactively select CSV for each loaded protein
    multiflux GPX4-wt=wt.csv,GPX4-single=single.csv,GPX4-double=double.csv
"""

from pymol import cmd
import csv
import os
import math

# 10-color flux palette (blue to red)
PALETTE_HEX = [
    '#000080', '#0000ff', '#0080ff', '#00ffff', '#00ff80',
    '#80ff00', '#ffff00', '#ff8000', '#ff0000', '#800000'
]


def _register_palette(prefix="flux"):
    """Create PyMOL colors flux_0 through flux_9"""
    names = []
    for i, hx in enumerate(PALETTE_HEX):
        rgb = tuple(int(hx[j:j+2], 16)/255.0 for j in (1, 3, 5))
        cname = f"{prefix}_{i}"
        cmd.set_color(cname, rgb)
        names.append(cname)
    return names


def _load_flux(csv_path, column="average_flux"):
    """Load flux data from CSV file"""
    data = {}
    try:
        with open(csv_path, newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                res = row.get("residue_index")
                if res is None or row.get(column) in (None, ""):
                    continue
                try:
                    data[str(int(float(res)))] = float(row[column])
                except ValueError:
                    pass
    except Exception as e:
        print(f"[multiflux] Error reading {csv_path}: {e}")
        return None
    
    if not data:
        print(f"[multiflux] Warning: No usable '{column}' data found in {csv_path}")
        return None
    return data


def _color_by_flux(selection, flux_data, palette):
    """Apply flux coloring to a selection"""
    if not flux_data:
        return
    
    v_min = min(flux_data.values())
    v_max = max(flux_data.values())
    span = v_max - v_min if v_max != v_min else 1.0
    ncol = len(palette)
    
    # Color each residue
    for resi, val in flux_data.items():
        bin_idx = int((val - v_min) / span * (ncol - 1) + 1e-9)
        bin_idx = max(0, min(ncol - 1, bin_idx))
        cmd.color(palette[bin_idx], f"({selection}) and resi {resi}")
    
    return v_min, v_max


def _setup_grid(n_objects):
    """Calculate grid dimensions for n objects"""
    if n_objects <= 1:
        return 1, 1
    elif n_objects <= 2:
        return 1, 2
    elif n_objects <= 4:
        return 2, 2
    elif n_objects <= 6:
        return 2, 3
    elif n_objects <= 9:
        return 3, 3
    else:
        cols = math.ceil(math.sqrt(n_objects))
        rows = math.ceil(n_objects / cols)
        return rows, cols


def _get_loaded_objects():
    """Get list of loaded protein objects (excluding measurements, selections, etc.)"""
    objects = []
    for obj in cmd.get_names('objects'):
        # Skip if it's a selection or measurement
        if cmd.get_type(obj) not in ['object:molecule', 'object:map', 'object:mesh']:
            continue
        # Check if it has atoms (is a molecule)
        if cmd.count_atoms(obj) > 0:
            objects.append(obj)
    return objects


def multiflux(mappings=None):
    """
    Color multiple proteins by their flux values and display in grid
    
    Parameters
    ----------
    mappings : str or None
        Comma-separated object=csv pairs, e.g. "obj1=file1.csv,obj2=file2.csv"
        If None, will prompt for each loaded object
    """
    # Register the color palette
    palette = _register_palette()
    
    # Get loaded objects
    objects = _get_loaded_objects()
    
    if not objects:
        print("[multiflux] No protein objects found. Load proteins first.")
        return
    
    print(f"[multiflux] Found {len(objects)} protein object(s): {', '.join(objects)}")
    
    # Parse mappings or get them interactively
    obj_csv_map = {}
    
    if mappings:
        # Parse provided mappings
        for mapping in mappings.split(','):
            if '=' in mapping:
                obj, csv_file = mapping.split('=', 1)
                obj = obj.strip()
                csv_file = csv_file.strip()
                if obj in objects and os.path.isfile(csv_file):
                    obj_csv_map[obj] = csv_file
                else:
                    print(f"[multiflux] Warning: Invalid mapping {mapping}")
    else:
        # Interactive mode
        print("\n[multiflux] Enter CSV file for each protein (or press Enter to skip):")
        for obj in objects:
            csv_file = input(f"  {obj}: ").strip()
            if csv_file and os.path.isfile(csv_file):
                obj_csv_map[obj] = csv_file
            elif csv_file:
                print(f"    File not found: {csv_file}")
    
    if not obj_csv_map:
        print("[multiflux] No valid CSV files provided.")
        return
    
    # Set up grid view
    n_mapped = len(obj_csv_map)
    rows, cols = _setup_grid(n_mapped)
    
    print(f"\n[multiflux] Setting up {rows}x{cols} grid for {n_mapped} protein(s)")
    
    # Hide all objects first
    cmd.hide('everything', 'all')
    
    # Process each protein
    grid_pos = 0
    stats = []
    
    for obj, csv_file in obj_csv_map.items():
        print(f"\n[multiflux] Processing {obj} with {csv_file}")
        
        # Load flux data
        flux_data = _load_flux(csv_file)
        if flux_data is None:
            continue
        
        # Apply coloring
        result = _color_by_flux(obj, flux_data, palette)
        if result:
            v_min, v_max = result
            stats.append(f"{obj}: flux range {v_min:.3g} to {v_max:.3g}")
        
        # Show as cartoon
        cmd.show('cartoon', obj)
        
        # Set grid position
        row = grid_pos // cols
        col = grid_pos % cols
        grid_slot = row * cols + col + 1
        
        cmd.set('grid_slot', grid_slot, obj)
        grid_pos += 1
    
    # Configure grid mode
    cmd.set('grid_mode', 1)
    cmd.set('grid_max', rows * cols)
    
    # Adjust grid spacing for better visibility
    if n_mapped > 1:
        cmd.set('grid_spacing', 40)  # Adjust spacing between grid cells
    
    # Reset view for all objects
    cmd.zoom('visible')
    cmd.orient()
    
    # Print summary
    print("\n[multiflux] Coloring complete!")
    print(f"[multiflux] Grid: {rows}x{cols}")
    for stat in stats:
        print(f"[multiflux] {stat}")
    print(f"[multiflux] Color palette: blue (low flux) to red (high flux)")
    
    # Add labels to identify proteins
    if n_mapped > 1:
        print("\n[multiflux] Tip: Use 'label all, object' to show protein names")


# Register as PyMOL command
cmd.extend("multiflux", multiflux)

# Optional: Add simpler command for common case
def fluxgrid():
    """Interactive version of multiflux"""
    multiflux()

cmd.extend("fluxgrid", fluxgrid)