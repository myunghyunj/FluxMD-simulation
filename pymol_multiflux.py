#!/usr/bin/env python
"""
pymol_multiflux.py - Color multiple proteins by flux values in PyMOL grid view

This script is designed specifically for PyMOL's native visualization system.
For matplotlib-based ribbon visualization, use visualize_multiflux.py instead.

Usage in PyMOL:
    run /path/to/pymol_multiflux.py
    multiflux              # Interactively select CSV for each loaded protein
    multiflux GPX4-wt=wt.csv,GPX4-single=single.csv,GPX4-double=double.csv

Note: This script must be run from within PyMOL.
"""

from pymol import cmd
import csv
import os
import math

# Berlin color palette - professional blue-white-red diverging colormap
PALETTE_HEX = [
    '#053061',  # Deep blue
    '#2166ac',  # Blue
    '#4393c3',  # Light blue
    '#92c5de',  # Pale blue
    '#d1e5f0',  # Very pale blue
    '#fddbc7',  # Very pale red
    '#f4a582',  # Pale red
    '#d6604d',  # Light red
    '#b2182b',  # Red
    '#67001f'   # Deep red
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
        # Interactive mode - use PyMOL's askstring dialog
        print("\n[multiflux] Interactive mode - dialog boxes will appear for each protein")
        for obj in objects:
            try:
                # Use PyMOL's GUI dialog
                csv_file = cmd.get_raw_input(f"Enter CSV file for {obj} (or Cancel to skip):")
                if csv_file:
                    csv_file = csv_file.strip()
                    if os.path.isfile(csv_file):
                        obj_csv_map[obj] = csv_file
                        print(f"[multiflux] {obj} -> {csv_file}")
                    else:
                        print(f"[multiflux] File not found: {csv_file}")
            except:
                # If dialog fails, try looking for common locations
                common_paths = [
                    f"{obj}_flux_analysis/processed_flux_data.csv",
                    f"flux_analysis_{obj}/processed_flux_data.csv",
                    f"{obj}/flux_analysis/processed_flux_data.csv",
                    "flux_analysis/processed_flux_data.csv",
                    "processed_flux_data.csv"
                ]
                
                for path in common_paths:
                    if os.path.isfile(path):
                        obj_csv_map[obj] = path
                        print(f"[multiflux] Auto-found for {obj}: {path}")
                        break
    
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
        
        # Hide ions, waters, and other non-protein elements
        cmd.hide('everything', f'{obj} and (resn HOH,WAT,SOL,NA,CL,K,CA,MG,ZN,FE,CU,MN,CO,NI)')
        cmd.hide('everything', f'{obj} and solvent')
        cmd.hide('everything', f'{obj} and inorganic')
        cmd.hide('everything', f'{obj} and not polymer')
        
        # Ensure only polymer (protein) is shown
        cmd.show('cartoon', f'{obj} and polymer')
        
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
    
    # Set white background for better contrast with Berlin palette
    cmd.bg_color('white')
    
    # Reset view for all objects
    cmd.zoom('visible')
    cmd.orient()
    
    # Print summary
    print("\n[multiflux] Coloring complete!")
    print(f"[multiflux] Grid: {rows}x{cols}")
    for stat in stats:
        print(f"[multiflux] {stat}")
    print(f"[multiflux] Color palette: Berlin (blue-white-red diverging)")
    
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

# Direct loading command - better handling for PyMOL
def fluxload(pdb_file, csv_file=None, label=None):
    """
    Load PDB and color by flux in one command
    
    Usage: 
        fluxload pdb_file, csv_file [, label]
        fluxload "pdb_file", "csv_file" [, "label"]  # For paths with spaces
    
    Example:
        fluxload protein.pdb, flux.csv, WT
        fluxload "/path with spaces/protein.pdb", "/path with spaces/flux.csv", "My Protein"
    """
    # Handle if all args come as single string (PyMOL quirk)
    if csv_file is None and ',' in str(pdb_file):
        # Parse as comma-separated string
        parts = pdb_file.split(',')
        if len(parts) >= 2:
            pdb_file = parts[0].strip().strip('"\'')
            csv_file = parts[1].strip().strip('"\'')
            label = parts[2].strip().strip('"\'') if len(parts) > 2 else None
        else:
            print("[fluxload] Error: Usage: fluxload pdb_file, csv_file [, label]")
            return
    
    # Clean up paths
    pdb_file = str(pdb_file).strip().strip('"\'')
    csv_file = str(csv_file).strip().strip('"\'') if csv_file else None
    label = str(label).strip().strip('"\'') if label else None
    
    if not os.path.exists(pdb_file):
        print(f"[fluxload] Error: PDB file not found: {pdb_file}")
        return
    
    if not os.path.exists(csv_file):
        print(f"[fluxload] Error: CSV file not found: {csv_file}")
        return
    
    # Determine label
    if label is None:
        label = os.path.basename(pdb_file).replace('.pdb', '')
    
    # Load PDB
    cmd.load(pdb_file, label)
    
    # Hide non-protein elements
    cmd.hide('everything', f'{label} and (resn HOH,WAT,SOL,NA,CL,K,CA,MG,ZN,FE,CU,MN,CO,NI)')
    cmd.hide('everything', f'{label} and solvent')
    cmd.hide('everything', f'{label} and inorganic')
    cmd.hide('everything', f'{label} and not polymer')
    
    # Color by flux
    palette = _register_palette()
    flux_data = _load_flux(csv_file)
    
    if flux_data:
        result = _color_by_flux(label, flux_data, palette)
        if result:
            v_min, v_max = result
            print(f"[fluxload] {label}: flux range {v_min:.3g} to {v_max:.3g}")
        
        cmd.show('cartoon', f'{label} and polymer')
        cmd.bg_color('white')
        print(f"[fluxload] Loaded and colored {label}")
    else:
        print(f"[fluxload] Failed to load flux data from {csv_file}")

cmd.extend("fluxload", fluxload)

print("""
PyMOL MultiFlux loaded! Commands available:
  multiflux - Interactive mode for multiple loaded proteins
  fluxload pdb_file, csv_file [, label] - Direct loading and coloring
  
Example: fluxload /path/to/GPX4-wt.pdb, /path/to/flux.csv, WT
Note: Use commas to separate arguments (required for paths with spaces)
""")