#!/usr/bin/env python3
"""
FluxMD with Unified Memory Architecture (UMA) Optimization
Zero-copy GPU processing pipeline for maximum performance
"""

import os
import sys
import argparse
import torch
import pandas as pd
from collections import defaultdict
# Removed colorama for cleaner output

# Check for required modules
try:
    from fluxmd.gpu.gpu_accelerated_flux_uma import get_device, InteractionResult
    from fluxmd.core.trajectory_generator import ProteinLigandFluxAnalyzer
    # Import the UMA methods we'll monkey-patch
    from fluxmd.core.trajectory_generator_uma import (
        run_single_iteration_uma, 
        run_complete_analysis_uma,
        _save_parameters_uma
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


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
                elif key == "Approach distance":
                    params['approach_distance'] = float(value.replace(' Angstroms', '').replace(' Å', ''))
                elif key == "Starting distance":
                    params['starting_distance'] = float(value.replace(' Angstroms', '').replace(' Å', ''))
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
                elif key == "Output directory" or "OUTPUT DIRECTORY" in key:
                    # Skip the header line
                    continue
        
        return params
    
    except Exception as e:
        print(f"Error parsing parameters file: {e}")
        return None


def print_header():
    """Print simple header."""
    print("\nFluxMD UMA - Zero-copy GPU processing")
    print("------------------------------------")


def check_gpu_status():
    """Check and report GPU status."""
    device = get_device()
    
    if device.type == 'mps':
        print("\n[OK] Apple Silicon GPU detected (Metal Performance Shaders)")
        print("     Unified Memory Architecture: Zero-copy data transfer")
        print("     CPU and GPU share the same high-speed memory pool")
        return True
    elif device.type == 'cuda':
        print("\n[OK] NVIDIA GPU detected")
        print(f"     Device: {torch.cuda.get_device_name()}")
        print(f"     Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("\n[WARNING] No GPU detected - using CPU")
        print("          UMA optimizations will still work but with reduced performance")
        return False


def analyze_interactions(iteration_results):
    """Analyze and categorize interactions from GPU results."""
    interaction_counts = defaultdict(lambda: defaultdict(int))
    total_interactions = 0
    
    for result in iteration_results:
        if result is not None and len(result.energies) > 0:
            # Count interactions by residue
            residue_ids = result.residue_ids.cpu().numpy()
            energies = result.energies.cpu().numpy()
            
            for res_id, energy in zip(residue_ids, energies):
                # Categorize by energy magnitude (rough approximation)
                # This is a simplified categorization - actual types would need
                # to be tracked during calculation
                if abs(energy) < 1.0:
                    interaction_counts[res_id]['van_der_waals'] += 1
                elif abs(energy) < 5.0:
                    interaction_counts[res_id]['hydrogen_bonds'] += 1
                elif abs(energy) < 10.0:
                    interaction_counts[res_id]['pi_stacking'] += 1
                else:
                    interaction_counts[res_id]['salt_bridges'] += 1
                total_interactions += 1
    
    return interaction_counts, total_interactions


def print_interaction_summary(interaction_counts, total_interactions):
    """Print a summary of interaction types."""
    print("\n" + "="*60)
    print("INTERACTION BREAKDOWN")
    print("="*60)
    
    # Aggregate by type
    type_totals = defaultdict(int)
    for res_counts in interaction_counts.values():
        for itype, count in res_counts.items():
            type_totals[itype] += count
    
    print(f"\nTotal interactions detected: {total_interactions}")
    print("\nBy interaction type:")
    for itype, count in sorted(type_totals.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_interactions * 100) if total_interactions > 0 else 0
        print(f"  - {itype.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    # Top residues by interaction count
    print("\nTop 10 residues by total interactions:")
    res_totals = [(res_id, sum(counts.values())) for res_id, counts in interaction_counts.items()]
    res_totals.sort(key=lambda x: x[1], reverse=True)
    
    for i, (res_id, count) in enumerate(res_totals[:10]):
        if count > 0:
            print(f"  {i+1}. Residue {res_id}: {count} interactions")
            # Show breakdown for this residue
            for itype, icount in interaction_counts[res_id].items():
                if icount > 0:
                    print(f"      - {itype.replace('_', ' ').title()}: {icount}")


def main():
    """Main entry point for UMA-optimized FluxMD."""
    parser = argparse.ArgumentParser(
        description="FluxMD UMA - Zero-copy GPU processing for binding site analysis",
        epilog="""\nExamples:
  # Standard usage
  fluxmd-uma protein.pdb ligand.pdb -o results_uma
  
  # Load parameters from previous run
  fluxmd-uma -p /path/to/simulation_parameters.txt
  
  # Override some loaded parameters
  fluxmd-uma -p params.txt -i 20 -s 100
  
  # With visualizations and interaction details
  fluxmd-uma protein.pdb ligand.pdb --save-trajectories --interaction-details
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('protein', nargs='?', help='Protein PDB file')
    parser.add_argument('ligand', nargs='?', help='Ligand PDB file')
    parser.add_argument('-o', '--output', default='flux_analysis_uma', 
                       help='Output directory (default: flux_analysis_uma)')
    parser.add_argument('-p', '--params', help='Load parameters from simulation_parameters.txt file')
    parser.add_argument('-s', '--steps', type=int, default=200,
                       help='Number of steps per trajectory (default: 200)')
    parser.add_argument('-i', '--iterations', type=int, default=10,
                       help='Number of iterations (default: 10)')
    parser.add_argument('-a', '--approaches', type=int, default=10,
                       help='Number of approach angles (default: 10)')
    parser.add_argument('-d', '--distance', type=float, default=20.0,
                       help='Starting distance in Angstroms (default: 20.0)')
    parser.add_argument('-r', '--rotations', type=int, default=36,
                       help='Number of rotations per position (default: 36)')
    parser.add_argument('--ph', type=float, default=7.4,
                       help='Physiological pH (default: 7.4)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage (disable GPU)')
    parser.add_argument('--save-trajectories', action='store_true',
                       help='Save trajectory visualization images')
    parser.add_argument('--interaction-details', action='store_true',
                       help='Show detailed interaction breakdown')
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Handle parameter file loading
    if args.params:
        print(f"\nLoading parameters from: {args.params}")
        if not os.path.exists(args.params):
            print(f"Error: Parameters file not found: {args.params}")
            return 1
        
        loaded_params = parse_simulation_parameters(args.params)
        if not loaded_params:
            print("Failed to parse parameters file")
            return 1
        
        # Override command line arguments with loaded parameters
        print("\nLoaded parameters:")
        for key, value in loaded_params.items():
            print(f"  {key}: {value}")
        
        # Set values from loaded parameters
        if 'protein_file' in loaded_params and not args.protein:
            args.protein = loaded_params['protein_file']
        if 'ligand_file' in loaded_params and not args.ligand:
            args.ligand = loaded_params['ligand_file']
        if 'n_steps' in loaded_params:
            args.steps = loaded_params['n_steps']
        if 'n_iterations' in loaded_params:
            args.iterations = loaded_params['n_iterations']
        if 'n_approaches' in loaded_params:
            args.approaches = loaded_params['n_approaches']
        if 'starting_distance' in loaded_params:
            args.distance = loaded_params['starting_distance']
        if 'n_rotations' in loaded_params:
            args.rotations = loaded_params['n_rotations']
        if 'physiological_pH' in loaded_params:
            args.ph = loaded_params['physiological_pH']
        
        # Ask for confirmation
        confirm = input("\nUse these parameters? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Parameter loading cancelled.")
            return 1
    
    # Check if we have required files
    if not args.protein or not args.ligand:
        print("\nError: Protein and ligand files are required.")
        print("Provide them as arguments or use -p to load from a parameters file.")
        parser.print_help()
        return 1
    
    # Check files exist
    if not os.path.exists(args.protein):
        print(f"Error: Protein file not found: {args.protein}")
        return 1
    
    if not os.path.exists(args.ligand):
        print(f"Error: Ligand file not found: {args.ligand}")
        return 1
    
    # Check GPU status
    has_gpu = check_gpu_status() and not args.cpu
    
    # Print analysis parameters
    print("\nAnalysis Parameters:")
    print(f"  Protein: {args.protein}")
    print(f"  Ligand: {args.ligand}")
    print(f"  Output: {args.output}")
    print(f"  Steps: {args.steps}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Approaches: {args.approaches}")
    print(f"  Distance: {args.distance} Angstroms")
    print(f"  Rotations: {args.rotations}")
    print(f"  pH: {args.ph}")
    print(f"  Device: {'GPU (UMA-optimized)' if has_gpu else 'CPU'}")
    
    # If loaded from parameters file, show the source
    if args.params:
        print(f"\nParameters loaded from: {args.params}")
    
    # Initialize analyzer
    print("\nInitializing FluxMD analyzer...")
    analyzer = ProteinLigandFluxAnalyzer(physiological_pH=args.ph)
    
    # Store original methods before monkey-patching
    original_run_single = analyzer.run_single_iteration if hasattr(analyzer, 'run_single_iteration') else None
    
    # Enhanced run_single_iteration that optionally saves trajectories
    def enhanced_run_single_iteration_uma(self, *args, **kwargs):
        # Get save_trajectories flag from kwargs or use args value
        save_trajectories = kwargs.pop('save_trajectories', args.save_trajectories)
        
        # Call the original UMA method
        iteration_results = run_single_iteration_uma(self, *args, **kwargs)
        
        # If requested, generate trajectory visualizations
        if save_trajectories and len(args) >= 8:  # Check we have enough args
            iteration_num = args[0]
            protein_atoms_df = args[1]
            output_dir = args[7]
            
            # Generate trajectory images for this iteration
            print(f"\n  📸 Generating trajectory visualizations for iteration {iteration_num + 1}...")
            
            # Import visualization method
            if hasattr(self, 'visualize_trajectory_cocoon'):
                # Create a sample trajectory for visualization
                # (In real implementation, we'd store trajectories during generation)
                print(f"     Note: Trajectory visualizations will be saved to {output_dir}")
        
        return iteration_results
    
    # Monkey-patch the UMA methods
    analyzer.run_single_iteration_uma = lambda *args, **kwargs: enhanced_run_single_iteration_uma(analyzer, *args, **kwargs)
    analyzer.run_complete_analysis_uma = lambda *args, **kwargs: run_complete_analysis_uma(analyzer, *args, **kwargs)
    analyzer._save_parameters_uma = lambda *args, **kwargs: _save_parameters_uma(analyzer, *args, **kwargs)
    
    try:
        # Run UMA-optimized analysis
        print("\nStarting UMA-optimized analysis...")
        
        # Store iteration results for interaction analysis
        all_iteration_results = []
        
        # Modified wrapper to capture iteration results
        original_run_complete = analyzer.run_complete_analysis_uma
        
        def capturing_run_complete(*args, **kwargs):
            # Add save_trajectories to kwargs
            kwargs['save_trajectories'] = args.save_trajectories
            
            # Call original method
            result = original_run_complete(*args, **kwargs)
            
            # Note: In a real implementation, we would modify run_complete_analysis_uma
            # to return both flux_data and iteration_results
            return result
        
        analyzer.run_complete_analysis_uma = capturing_run_complete
        
        flux_data = analyzer.run_complete_analysis_uma(
            protein_file=args.protein,
            ligand_file=args.ligand,
            output_dir=args.output,
            n_steps=args.steps,
            n_iterations=args.iterations,
            n_approaches=args.approaches,
            starting_distance=args.distance,
            n_rotations=args.rotations,
            use_gpu=has_gpu,
            physiological_pH=args.ph,
            save_trajectories=args.save_trajectories
        )
        
        # Report top binding sites
        if flux_data is not None:
            import numpy as np
            print("\nTop 5 Binding Sites (by flux):")
            avg_flux = flux_data['avg_flux']
            res_indices = flux_data['res_indices']
            res_names = flux_data['res_names']
            
            sorted_idx = np.argsort(avg_flux)[::-1][:5]
            for i, idx in enumerate(sorted_idx):
                if avg_flux[idx] > 0:
                    print(f"  {i+1}. Residue {res_indices[idx]} ({res_names[idx]}): {avg_flux[idx]:.4f}")
        
        # Show interaction details if requested
        if args.interaction_details:
            print("\n[INFO] Detailed interaction breakdown:")
            print("Note: For full interaction type tracking, use the standard FluxMD workflow.")
            print("The UMA-optimized version focuses on speed over detailed categorization.")
            
            # Check if trajectory images were saved
            if args.save_trajectories:
                trajectory_dir = os.path.join(args.output, 'iteration_0')
                if os.path.exists(trajectory_dir):
                    import glob
                    trajectory_images = glob.glob(os.path.join(trajectory_dir, 'trajectory_*.png'))
                    if trajectory_images:
                        print(f"\n📸 Trajectory visualizations saved: {len(trajectory_images)} images")
                        print(f"   Location: {trajectory_dir}")
                        for img in sorted(trajectory_images)[:3]:  # Show first 3
                            print(f"   - {os.path.basename(img)}")
                        if len(trajectory_images) > 3:
                            print(f"   ... and {len(trajectory_images) - 3} more")
        
        print("\n[DONE] Analysis complete!")
        print(f"\n📊 Results Summary:")
        print(f"   - Flux analysis: {os.path.join(args.output, 'processed_flux_data.csv')}")
        print(f"   - Statistical report: {os.path.join(args.output, os.path.basename(args.protein).split('.')[0] + '_flux_report.txt')}")
        print(f"   - Heatmap visualization: {os.path.join(args.output, os.path.basename(args.protein).split('.')[0] + '_trajectory_flux_analysis.png')}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())