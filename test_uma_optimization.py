#!/usr/bin/env python3
"""
Quick test of UMA optimization to verify it works correctly
"""

import torch
import numpy as np
import pandas as pd
from gpu_accelerated_flux_uma import GPUAcceleratedInteractionCalculator, get_device
from flux_analyzer_uma import TrajectoryFluxAnalyzer

def test_uma_pipeline():
    """Test the UMA-optimized pipeline with synthetic data."""
    print("Testing UMA-Optimized Pipeline")
    print("="*60)
    
    # Get device
    device = get_device()
    
    # Create synthetic protein (10 residues, 3 atoms each)
    n_residues = 10
    atoms_per_residue = 3
    protein_data = []
    
    for res_id in range(n_residues):
        for atom_idx in range(atoms_per_residue):
            protein_data.append({
                'x': res_id * 3.5 + atom_idx * 1.2,
                'y': np.sin(res_id * 0.5) * 5,
                'z': np.cos(res_id * 0.5) * 5,
                'resname': 'ALA',
                'name': f'CA{atom_idx}',
                'element': 'C',
                'residue_id': res_id,
                'chain': 'A',
                'resSeq': res_id
            })
    
    protein_df = pd.DataFrame(protein_data)
    print(f"Created synthetic protein: {len(protein_df)} atoms, {n_residues} residues")
    
    # Create synthetic ligand (benzene-like, 6 atoms)
    ligand_data = []
    for i in range(6):
        angle = i * np.pi / 3
        ligand_data.append({
            'x': np.cos(angle) * 1.4,
            'y': np.sin(angle) * 1.4,
            'z': 0,
            'element': 'C',
            'name': f'C{i+1}'
        })
    
    ligand_df = pd.DataFrame(ligand_data)
    print(f"Created synthetic ligand: {len(ligand_df)} atoms")
    
    # Initialize GPU calculator
    print(f"\nInitializing GPU calculator on {device}...")
    calc = GPUAcceleratedInteractionCalculator(device=device)
    
    # Pre-compute properties
    calc.precompute_protein_properties(protein_df)
    ligand_props = calc.precompute_ligand_properties(ligand_df)
    
    # Create mock intra-protein vectors
    print("\nCreating mock intra-protein vectors...")
    intra_vectors = {}
    for res_id in range(n_residues):
        # Random small vectors
        intra_vectors[f'A:{res_id}'] = np.random.randn(3) * 0.1
    calc.set_intra_protein_vectors(intra_vectors)
    
    # Generate mock trajectory (5 positions)
    print("\nGenerating mock trajectory...")
    trajectory = []
    for i in range(5):
        # Move ligand around protein
        angle = i * 2 * np.pi / 5
        position = [
            np.cos(angle) * 15,
            np.sin(angle) * 15,
            0
        ]
        trajectory.append(position)
    
    # Process trajectory
    print("\nProcessing trajectory on GPU...")
    ligand_coords = ligand_df[['x', 'y', 'z']].values
    results = calc.process_trajectory_batch(np.array(trajectory), ligand_coords, n_rotations=4)
    
    print(f"Processed {len(results)} frames with interactions")
    total_interactions = sum(len(r.energies) for r in results if r is not None)
    print(f"Total interactions found: {total_interactions}")
    
    # Test flux analysis
    print("\nTesting flux analysis...")
    analyzer = TrajectoryFluxAnalyzer(device=device)
    
    # Create mock protein structure info
    analyzer.n_residues = n_residues
    analyzer.residue_indices = list(range(n_residues))
    analyzer.residue_names = ['ALA'] * n_residues
    
    # Process iterations (mock 2 iterations)
    all_iterations = [results, results]  # Use same results twice
    
    try:
        flux_data = analyzer.process_iterations_and_calculate_flux(
            all_iterations,
            calc.intra_protein_vectors_gpu
        )
        
        print("\nFlux analysis successful!")
        print(f"Average flux shape: {flux_data['avg_flux'].shape}")
        print(f"Max flux value: {flux_data['avg_flux'].max():.4f}")
        print(f"Non-zero residues: {np.sum(flux_data['avg_flux'] > 0)}")
        
        # Show top residues
        top_indices = np.argsort(flux_data['avg_flux'])[::-1][:3]
        print("\nTop 3 residues by flux:")
        for i, idx in enumerate(top_indices):
            if flux_data['avg_flux'][idx] > 0:
                print(f"  {i+1}. Residue {idx}: {flux_data['avg_flux'][idx]:.4f}")
        
        print("\n✅ UMA optimization test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during flux analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scatter_operations():
    """Test that scatter operations work correctly."""
    print("\n\nTesting Scatter Operations")
    print("="*60)
    
    device = get_device()
    
    # Test scatter_add
    n_residues = 10
    n_interactions = 100
    
    # Random residue assignments
    residue_ids = torch.randint(0, n_residues, (n_interactions,), device=device)
    energies = torch.randn(n_interactions, device=device)
    
    # Scatter add
    energy_sum = torch.zeros(n_residues, device=device)
    energy_sum.scatter_add_(0, residue_ids, energies)
    
    # Verify
    for res_id in range(n_residues):
        mask = residue_ids == res_id
        expected = energies[mask].sum()
        actual = energy_sum[res_id]
        assert torch.allclose(expected, actual), f"Scatter add failed for residue {res_id}"
    
    print("✅ Scatter operations test PASSED!")
    
    # Show performance
    print(f"\nProcessed {n_interactions} interactions")
    print(f"Residue energy sums: {energy_sum[:5].cpu().numpy()}")


if __name__ == "__main__":
    # Run tests
    test_uma_pipeline()
    test_scatter_operations()
    
    print("\n" + "="*60)
    print("All UMA optimization tests completed!")
    print("="*60)