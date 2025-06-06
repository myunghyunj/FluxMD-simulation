#!/usr/bin/env python3
"""
Basic usage example for FluxMD
"""

import os
import sys

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fluxmd import ProteinLigandFluxAnalyzer, get_device


def run_basic_analysis():
    """Run a basic FluxMD analysis."""
    
    # File paths - replace with your own
    protein_file = "examples/data/protein.pdb"
    ligand_file = "examples/data/ligand.pdb"
    output_dir = "results/basic_example"
    
    # Check if files exist
    if not os.path.exists(protein_file):
        print(f"Error: Protein file not found: {protein_file}")
        print("Please provide your own protein PDB file")
        return
        
    if not os.path.exists(ligand_file):
        print(f"Error: Ligand file not found: {ligand_file}")
        print("Please provide your own ligand PDB file")
        return
    
    # Initialize analyzer
    print("Initializing FluxMD analyzer...")
    analyzer = ProteinLigandFluxAnalyzer(physiological_pH=7.4)
    
    # Check GPU availability
    device = get_device()
    use_gpu = device.type != 'cpu'
    
    # Run analysis with moderate parameters for testing
    print(f"Running analysis (GPU: {use_gpu})...")
    analyzer.run_complete_analysis(
        protein_file=protein_file,
        ligand_file=ligand_file,
        output_dir=output_dir,
        n_steps=100,         # Reduced for example
        n_iterations=5,      # Reduced for example
        n_approaches=5,      # Reduced for example
        starting_distance=20.0,
        n_rotations=12,      # Reduced for example
        use_gpu=use_gpu
    )
    
    print(f"Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    run_basic_analysis()