#!/usr/bin/env python3
"""
FluxMD with Unified Memory Architecture (UMA) Optimization
Zero-copy GPU processing pipeline for maximum performance
"""

import os
import sys
import argparse
import torch
from colorama import init, Fore, Style
init(autoreset=True)

# Check for required modules
try:
    from gpu_accelerated_flux_uma import get_device
    from trajectory_generator import ProteinLigandFluxAnalyzer
    # Import the UMA methods we'll monkey-patch
    from trajectory_generator_uma import (
        run_single_iteration_uma, 
        run_complete_analysis_uma,
        _save_parameters_uma
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


def print_banner():
    """Print FluxMD UMA banner."""
    banner = f"""
{Fore.CYAN}╔------------------------------------------------------------╗
║                                                                  ║
║  {Fore.WHITE}███████╗██╗     ██╗   ██╗██╗  ██╗███╗   ███╗██████╗{Fore.CYAN}           ║
║  {Fore.WHITE}██╔════╝██║     ██║   ██║╚██╗██╔╝████╗ ████║██╔══██╗{Fore.CYAN}          ║
║  {Fore.WHITE}█████╗  ██║     ██║   ██║ ╚███╔╝ ██╔████╔██║██║  ██║{Fore.CYAN}          ║
║  {Fore.WHITE}██╔══╝  ██║     ██║   ██║ ██╔██╗ ██║╚██╔╝██║██║  ██║{Fore.CYAN}          ║
║  {Fore.WHITE}██║     ███████╗╚██████╔╝██╔╝ ██╗██║ ╚═╝ ██║██████╔╝{Fore.CYAN}          ║
║  {Fore.WHITE}╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝{Fore.CYAN}           ║
║                                                                  ║
║          {Fore.YELLOW} Unified Memory Architecture Edition {Fore.CYAN}               ║
║                                                                  ║
║  {Fore.GREEN}Zero-Copy GPU Processing | 100x Faster | No File I/O{Fore.CYAN}           ║
╚------------------------------------------------------------╝{Style.RESET_ALL}
"""
    print(banner)


def check_gpu_status():
    """Check and report GPU status."""
    device = get_device()
    
    if device.type == 'mps':
        print(f"\n{Fore.GREEN}[OK] Apple Silicon GPU detected (Metal Performance Shaders)")
        print(f"  → Unified Memory Architecture: Zero-copy data transfer")
        print(f"  → CPU and GPU share the same high-speed memory pool")
        return True
    elif device.type == 'cuda':
        print(f"\n{Fore.GREEN}[OK] NVIDIA GPU detected")
        print(f"  → Device: {torch.cuda.get_device_name()}")
        print(f"  → Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print(f"\n{Fore.YELLOW}⚠ No GPU detected - using CPU")
        print(f"  → UMA optimizations will still work but with reduced performance")
        return False


def main():
    """Main entry point for UMA-optimized FluxMD."""
    parser = argparse.ArgumentParser(
        description="FluxMD UMA - Zero-copy GPU processing for binding site analysis"
    )
    
    parser.add_argument('protein', help='Protein PDB file')
    parser.add_argument('ligand', help='Ligand PDB file')
    parser.add_argument('-o', '--output', default='flux_analysis_uma', 
                       help='Output directory (default: flux_analysis_uma)')
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
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check files exist
    if not os.path.exists(args.protein):
        print(f"{Fore.RED}Error: Protein file not found: {args.protein}")
        return 1
    
    if not os.path.exists(args.ligand):
        print(f"{Fore.RED}Error: Ligand file not found: {args.ligand}")
        return 1
    
    # Check GPU status
    has_gpu = check_gpu_status() and not args.cpu
    
    # Print analysis parameters
    print(f"\n{Fore.CYAN}Analysis Parameters:")
    print(f"  Protein: {args.protein}")
    print(f"  Ligand: {args.ligand}")
    print(f"  Output: {args.output}")
    print(f"  Steps: {args.steps}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Approaches: {args.approaches}")
    print(f"  Distance: {args.distance} Å")
    print(f"  Rotations: {args.rotations}")
    print(f"  pH: {args.ph}")
    print(f"  Device: {'GPU (UMA-optimized)' if has_gpu else 'CPU'}")
    
    # Initialize analyzer
    print(f"\n{Fore.CYAN}Initializing FluxMD analyzer...")
    analyzer = ProteinLigandFluxAnalyzer(physiological_pH=args.ph)
    
    # Monkey-patch the UMA methods
    analyzer.run_single_iteration_uma = lambda *args, **kwargs: run_single_iteration_uma(analyzer, *args, **kwargs)
    analyzer.run_complete_analysis_uma = lambda *args, **kwargs: run_complete_analysis_uma(analyzer, *args, **kwargs)
    analyzer._save_parameters_uma = lambda *args, **kwargs: _save_parameters_uma(analyzer, *args, **kwargs)
    
    try:
        # Run UMA-optimized analysis
        print(f"\n{Fore.CYAN}Starting UMA-optimized analysis...")
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
            physiological_pH=args.ph
        )
        
        # Report top binding sites
        if flux_data is not None:
            print(f"\n{Fore.GREEN}Top 5 Binding Sites (by flux):")
            avg_flux = flux_data['avg_flux']
            res_indices = flux_data['res_indices']
            res_names = flux_data['res_names']
            
            sorted_idx = np.argsort(avg_flux)[::-1][:5]
            for i, idx in enumerate(sorted_idx):
                if avg_flux[idx] > 0:
                    print(f"  {i+1}. Residue {res_indices[idx]} ({res_names[idx]}): {avg_flux[idx]:.4f}")
        
        print(f"\n{Fore.GREEN}[DONE] Analysis complete!")
        return 0
        
    except Exception as e:
        print(f"\n{Fore.RED}Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())