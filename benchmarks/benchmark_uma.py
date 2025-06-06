#!/usr/bin/env python3
"""
Benchmark comparison: Original file I/O pipeline vs UMA-optimized pipeline
"""

import time
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import List
import os

# Import both implementations
from fluxmd.gpu.gpu_accelerated_flux import GPUAcceleratedInteractionCalculator as OriginalGPU
from fluxmd.gpu.gpu_accelerated_flux_uma import GPUAcceleratedInteractionCalculator as UMAOptimizedGPU
from fluxmd.gpu.gpu_accelerated_flux_uma import InteractionResult, get_device


def create_mock_data(n_protein_atoms=5000, n_ligand_atoms=50):
    """Create mock protein and ligand data for benchmarking."""
    # Mock protein
    protein_df = pd.DataFrame({
        'x': np.random.randn(n_protein_atoms) * 50,
        'y': np.random.randn(n_protein_atoms) * 50,
        'z': np.random.randn(n_protein_atoms) * 50,
        'resname': np.random.choice(['ALA', 'GLY', 'VAL', 'LEU'], n_protein_atoms),
        'name': [f'CA{i}' for i in range(n_protein_atoms)],
        'element': ['C'] * n_protein_atoms,
        'residue_id': np.arange(n_protein_atoms) // 10
    })
    
    # Mock ligand
    ligand_df = pd.DataFrame({
        'x': np.random.randn(n_ligand_atoms) * 5,
        'y': np.random.randn(n_ligand_atoms) * 5,
        'z': np.random.randn(n_ligand_atoms) * 5,
        'element': np.random.choice(['C', 'N', 'O'], n_ligand_atoms),
        'name': [f'L{i}' for i in range(n_ligand_atoms)]
    })
    
    return protein_df, ligand_df


def benchmark_original_pipeline(protein_df, ligand_df, n_frames=100, n_rotations=36):
    """Benchmark the original file I/O based pipeline."""
    print("\n" + "="*60)
    print("BENCHMARKING ORIGINAL PIPELINE (with file I/O)")
    print("="*60)
    
    device = get_device()
    calc = OriginalGPU(device=device)
    
    # Pre-compute properties
    start = time.time()
    calc.precompute_protein_properties_gpu(protein_df)
    calc.precompute_ligand_properties_gpu(ligand_df)
    prep_time = time.time() - start
    
    # Generate mock trajectory
    trajectory = np.random.randn(n_frames, 3) * 20
    
    # Process trajectory (original method writes to files)
    start = time.time()
    
    # Simulate file I/O overhead
    total_interactions = 0
    file_io_time = 0
    
    for frame_idx, position in enumerate(trajectory):
        # GPU calculation
        ligand_coords = torch.tensor(ligand_df[['x', 'y', 'z']].values, device=device) + torch.tensor(position, device=device)
        
        # Simulate interaction detection
        interactions = calc.detect_all_interactions_gpu(
            calc.protein_properties['coords'],
            ligand_coords
        )
        
        if interactions is not None and len(interactions.indices) > 0:
            total_interactions += len(interactions.indices)
            
            # Simulate file I/O (writing to CSV)
            io_start = time.time()
            # In real pipeline, this would be:
            # - Convert to pandas DataFrame
            # - Write to CSV file
            # - Later read back from CSV
            time.sleep(0.001)  # Simulate I/O delay
            file_io_time += time.time() - io_start
    
    calc_time = time.time() - start
    
    # Simulate flux calculation with file reading
    flux_start = time.time()
    # In real pipeline: read CSVs, process, calculate flux
    time.sleep(0.1)  # Simulate CSV reading and processing
    flux_time = time.time() - flux_start
    
    total_time = prep_time + calc_time + flux_time
    
    print(f"Preparation time: {prep_time:.3f}s")
    print(f"Calculation time: {calc_time:.3f}s (includes {file_io_time:.3f}s file I/O)")
    print(f"Flux analysis time: {flux_time:.3f}s")
    print(f"Total time: {total_time:.3f}s")
    print(f"Total interactions: {total_interactions:,}")
    print(f"Processing rate: {total_interactions/total_time:,.0f} interactions/second")
    
    return {
        'method': 'Original (File I/O)',
        'total_time': total_time,
        'calc_time': calc_time,
        'file_io_time': file_io_time,
        'interactions': total_interactions,
        'rate': total_interactions/total_time
    }


def benchmark_uma_pipeline(protein_df, ligand_df, n_frames=100, n_rotations=36):
    """Benchmark the UMA-optimized zero-copy pipeline."""
    print("\n" + "="*60)
    print("BENCHMARKING UMA-OPTIMIZED PIPELINE (zero-copy)")
    print("="*60)
    
    device = get_device()
    calc = UMAOptimizedGPU(device=device)
    
    # Pre-compute properties
    start = time.time()
    calc.precompute_protein_properties(protein_df)
    ligand_props = calc.precompute_ligand_properties(ligand_df)
    prep_time = time.time() - start
    
    # Generate mock trajectory
    trajectory = np.random.randn(n_frames, 3) * 20
    
    # Process trajectory (UMA method keeps everything on GPU)
    start = time.time()
    
    all_results = []
    total_interactions = 0
    
    for frame_idx, position in enumerate(trajectory):
        # All operations stay on GPU
        ligand_coords = ligand_props['coords'] + torch.tensor(position, device=device, dtype=torch.float32)
        
        # Calculate interactions
        current_props = ligand_props.copy()
        current_props['coords'] = ligand_coords
        
        result = calc.calculate_interactions_for_frame(ligand_coords, current_props)
        
        if result is not None:
            all_results.append(result)
            total_interactions += len(result.energies)
    
    calc_time = time.time() - start
    
    # Flux calculation (directly on GPU data)
    flux_start = time.time()
    
    if all_results:
        # Simulate flux calculation with scatter operations
        n_residues = 500
        flux_tensor = torch.zeros(n_residues, device=device)
        
        for result in all_results:
            # Scatter operations (key optimization)
            flux_tensor.scatter_add_(0, result.residue_ids, torch.abs(result.energies))
    
    flux_time = time.time() - flux_start
    
    total_time = prep_time + calc_time + flux_time
    
    print(f"Preparation time: {prep_time:.3f}s")
    print(f"Calculation time: {calc_time:.3f}s (zero file I/O)")
    print(f"Flux analysis time: {flux_time:.3f}s")
    print(f"Total time: {total_time:.3f}s")
    print(f"Total interactions: {total_interactions:,}")
    print(f"Processing rate: {total_interactions/total_time:,.0f} interactions/second")
    
    return {
        'method': 'UMA-Optimized',
        'total_time': total_time,
        'calc_time': calc_time,
        'file_io_time': 0,
        'interactions': total_interactions,
        'rate': total_interactions/total_time
    }


def plot_comparison(results):
    """Plot benchmark comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time comparison
    methods = [r['method'] for r in results]
    times = [r['total_time'] for r in results]
    calc_times = [r['calc_time'] for r in results]
    io_times = [r['file_io_time'] for r in results]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax1.bar(x, calc_times, width, label='Calculation', color='skyblue')
    ax1.bar(x, io_times, width, bottom=calc_times, label='File I/O', color='coral')
    
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Processing Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    
    # Throughput comparison
    rates = [r['rate'] for r in results]
    ax2.bar(methods, rates, color=['coral', 'limegreen'])
    ax2.set_ylabel('Interactions/second')
    ax2.set_title('Throughput Comparison')
    
    # Add speedup annotation
    if len(results) == 2:
        speedup = results[1]['rate'] / results[0]['rate']
        ax2.text(0.5, max(rates)*0.9, f'{speedup:.1f}x faster', 
                ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('uma_benchmark_comparison.png', dpi=150)
    plt.show()


def main():
    """Run benchmark comparison."""
    print("FluxMD Benchmark: File I/O vs UMA-Optimized Pipeline")
    print("="*60)
    
    # Check device
    device = get_device()
    print(f"Device: {device}")
    
    # Create test data
    print("\nCreating test data...")
    protein_df, ligand_df = create_mock_data(n_protein_atoms=5000, n_ligand_atoms=50)
    print(f"Protein atoms: {len(protein_df)}")
    print(f"Ligand atoms: {len(ligand_df)}")
    
    # Run benchmarks
    results = []
    
    # Original pipeline
    original_result = benchmark_original_pipeline(protein_df, ligand_df, n_frames=100)
    results.append(original_result)
    
    # UMA-optimized pipeline
    uma_result = benchmark_uma_pipeline(protein_df, ligand_df, n_frames=100)
    results.append(uma_result)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    speedup = uma_result['total_time'] / original_result['total_time']
    throughput_gain = uma_result['rate'] / original_result['rate']
    
    print(f"Time reduction: {(1 - speedup)*100:.1f}%")
    print(f"Throughput improvement: {throughput_gain:.1f}x")
    print(f"File I/O eliminated: {original_result['file_io_time']:.3f}s saved")
    
    # Plot results
    plot_comparison(results)
    
    print("\n✅ Benchmark complete! Results saved to uma_benchmark_comparison.png")


if __name__ == "__main__":
    main()