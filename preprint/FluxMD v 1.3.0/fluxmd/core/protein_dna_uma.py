#!/usr/bin/env python3
"""
FluxMD Protein-DNA UMA: Command-line tool for Protein-DNA interaction analysis.
"""
import argparse
from fluxmd.core.protein_dna_workflow import run_protein_dna_workflow

def main():
    parser = argparse.ArgumentParser(
        description='Run GPU-accelerated Protein-DNA interaction analysis.',
        epilog='Example: fluxmd-protein-dna-uma dna.pdb protein.pdb -o results/'
    )
    parser.add_argument('dna_file', help='Input DNA structure file (PDB format)')
    parser.add_argument('protein_file', help='Input Protein structure file (PDB format)')
    parser.add_argument('-o', '--output', default='flux_results',
                       help='Output directory for results (default: flux_results)')
    # Add other relevant arguments from fluxmd-uma if needed
    parser.add_argument('-s', '--steps', type=int, default=200, help='Steps per trajectory')
    parser.add_argument('-i', '--iterations', type=int, default=10, help='Number of iterations')
    parser.add_argument('-a', '--approaches', type=int, default=10, help='Number of approach angles')
    parser.add_argument('-d', '--distance', type=float, default=20.0, 
                       help='Starting distance in Å (default: 20.0, recommended: 30.0 for large proteins)')
    parser.add_argument('-r', '--rotations', type=int, default=36, help='Rotations per position')
    parser.add_argument('--step-size', type=float, default=None,
                       help='Trajectory step size in Angstroms (default: auto-calculated)')
    parser.add_argument('--ph', type=float, default=7.4, help='Physiological pH for protonation')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--no-collision', action='store_true', 
                       help='Disable collision detection (for debugging)')
    parser.add_argument('--approach-distance', type=float, default=2.5,
                       help='Distance step between approaches (default: 2.5 Å, recommended: 1.5 for large proteins)')

    args = parser.parse_args()

    run_protein_dna_workflow(
        dna_file=args.dna_file,
        protein_file=args.protein_file,
        output_dir=args.output,
        n_steps=args.steps,
        n_iterations=args.iterations,
        n_approaches=args.approaches,
        starting_distance=args.distance,
        approach_distance=args.approach_distance,
        n_rotations=args.rotations,
        trajectory_step_size=args.step_size,
        physiological_pH=args.ph,
        force_cpu=args.cpu,
        disable_collision_detection=args.no_collision
    )

if __name__ == '__main__':
    main() 