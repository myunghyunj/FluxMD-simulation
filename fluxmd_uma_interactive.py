#!/usr/bin/env python3
"""
Interactive wrapper for FluxMD UMA version
Provides the same interactive workflow as the standard version
"""

import os
import sys
import subprocess

def parse_simulation_parameters(params_file):
    """Parse simulation parameters from existing file"""
    params = {}
    
    try:
        with open(params_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if ': ' in line:
                key, value = line.split(': ', 1)
                
                # Extract trajectory parameters
                if key == "Steps per approach":
                    params['n_steps'] = int(value)
                elif key == "Number of iterations":
                    params['n_iterations'] = int(value)
                elif key == "Number of approaches":
                    params['n_approaches'] = int(value)
                elif key == "Approach distance" or key == "Initial approach distance":
                    clean_value = value.replace(' Angstroms', '').replace(' Å', '').strip()
                    params['approach_distance'] = float(clean_value)
                elif key == "Starting distance":
                    clean_value = value.replace(' Angstroms', '').replace(' Å', '').strip()
                    params['starting_distance'] = float(clean_value)
                elif key == "Rotations per position":
                    params['n_rotations'] = int(value)
                elif key == "pH":
                    params['physiological_pH'] = float(value)
                elif key == "Protein":
                    params['protein_file'] = value
                elif key == "Ligand":
                    params['ligand_file'] = value
                elif key == "Protein name":
                    params['protein_name'] = value
                elif key == "Mode":
                    params['mode'] = value
        
        return params
    
    except Exception as e:
        print(f"Error parsing parameters file: {e}")
        return None


def main():
    """Interactive workflow for UMA version"""
    print("=" * 80)
    print(" " * 29 + "UMA-OPTIMIZED WORKFLOW" + " " * 29)
    print("=" * 80)
    print("\nThis uses zero-copy GPU processing for maximum performance.")
    print("Best for Apple Silicon Macs and systems with unified memory.")
    
    # Ask if user wants to load existing parameters
    use_existing = input("\nLoad parameters from existing simulation? (y/n): ").strip().lower()
    
    protein_file = None
    ligand_file = None
    params = {}
    
    if use_existing == 'y':
        params_file = input("Enter path to simulation_parameters.txt: ").strip()
        if os.path.exists(params_file):
            loaded_params = parse_simulation_parameters(params_file)
            if loaded_params:
                print("\nLoaded parameters from file:")
                for key, value in loaded_params.items():
                    print(f"  {key}: {value}")
                
                # Ask about protein and ligand files FIRST
                if 'protein_file' in loaded_params and 'ligand_file' in loaded_params:
                    print(f"\nLoaded protein: {loaded_params['protein_file']}")
                    print(f"Loaded ligand: {loaded_params['ligand_file']}")
                    use_same = input("\nUse the same protein and ligand files? (y/n): ").strip().lower()
                    
                    if use_same == 'y':
                        protein_file = loaded_params['protein_file']
                        ligand_file = loaded_params['ligand_file']
                    else:
                        # Ask for new files
                        protein_file = input("Enter path to protein PDB file: ").strip()
                        ligand_file = input("Enter path to ligand PDB file: ").strip()
                
                # Copy other parameters
                params = loaded_params.copy()
        else:
            print(f"File not found: {params_file}")
            use_existing = 'n'
    
    # Get protein/ligand files if not already set
    if not protein_file:
        protein_file = input("\nEnter path to protein PDB file: ").strip()
    if not ligand_file:
        ligand_file = input("Enter path to ligand PDB file: ").strip()
    
    # Get output directory
    output_dir = input("Output directory (default 'flux_analysis_uma'): ").strip() or "flux_analysis_uma"
    
    # If not using existing parameters, ask for them
    if use_existing != 'y' or not params:
        print("\nEnter parameters manually (press Enter for defaults):")
        params['n_steps'] = int(input("Steps per approach (default 200): ") or "200")
        params['n_iterations'] = int(input("Number of iterations (default 10): ") or "10")
        params['n_approaches'] = int(input("Number of approaches (default 10): ") or "10")
        params['approach_distance'] = float(input("Approach distance in Angstroms (default 2.5): ") or "2.5")
        params['starting_distance'] = float(input("Starting distance in Angstroms (default 20): ") or "20")
        params['n_rotations'] = int(input("Rotations per position (default 36): ") or "36")
        params['physiological_pH'] = float(input("pH for protonation state (default 7.4): ") or "7.4")
    
    # Ask about additional options
    save_trajectories = input("\nSave trajectory visualizations? (y/n): ").strip().lower() == 'y'
    interaction_details = input("Show detailed interaction breakdown? (y/n): ").strip().lower() == 'y'
    
    # Build command
    cmd = ['python', '-m', 'fluxmd_uma', protein_file, ligand_file]
    cmd.extend(['-o', output_dir])
    
    # Add parameters
    if 'n_steps' in params:
        cmd.extend(['-s', str(params['n_steps'])])
    if 'n_iterations' in params:
        cmd.extend(['-i', str(params['n_iterations'])])
    if 'n_approaches' in params:
        cmd.extend(['-a', str(params['n_approaches'])])
    if 'starting_distance' in params:
        cmd.extend(['-d', str(params['starting_distance'])])
    if 'n_rotations' in params:
        cmd.extend(['-r', str(params['n_rotations'])])
    if 'approach_distance' in params:
        cmd.extend(['--approach-distance', str(params['approach_distance'])])
    if 'physiological_pH' in params:
        cmd.extend(['--ph', str(params['physiological_pH'])])
    
    if save_trajectories:
        cmd.append('--save-trajectories')
    if interaction_details:
        cmd.append('--interaction-details')
    
    # Run the UMA version
    print("\nStarting UMA-optimized analysis...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError running UMA analysis: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())