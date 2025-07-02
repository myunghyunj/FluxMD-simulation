# FluxMD Analysis Module

## Overview
Advanced flux analysis with GPU acceleration and statistical validation.

## Key Features

### GPU-Accelerated Bootstrap Analysis
- 1000 iterations for confidence intervals
- P-value calculation for significance testing
- Scatter operations for massive parallelism

### Adaptive Algorithms
- Direct computation (<1M atom pairs)
- Spatial hashing (1M-100M pairs)  
- Hierarchical filtering (>100M pairs)

### Flux Calculation
Implements the signed flux metric: **Φᵢ = ⟨E̅ᵢ⟩ · Cᵢ · (1 + τᵢ)**

- **Signed Energy (⟨E̅ᵢ⟩)**: Mean energy-weighted vector preserving sign
  - Negative: attractive/stabilizing interactions
  - Positive: repulsive/destabilizing interactions
- **Consistency (Cᵢ)**: Directional alignment (entropic ordering)
- **Temporal (τᵢ)**: Dynamic fluctuation (stability indicator)

### Implementation Differences
- **CPU Version**: Uses Savitzky-Golay derivative for temporal term
- **GPU Version**: Uses normalized variance for temporal term

## Performance Optimizations

### Memory Architecture
- Zero-copy tensor operations on UMA systems
- Pre-computed rotation matrices
- Structure-of-arrays format for vectorized loads

### GPU Kernels
- Warp-aligned computations (multiples of 32)
- Texture memory for lookup tables
- Mixed precision with Tensor Cores (NVIDIA)

## Statistical Methods
- Bootstrap resampling with GPU acceleration
- Gaussian smoothing for spatial coherence
- Collision detection via KD-tree indexing