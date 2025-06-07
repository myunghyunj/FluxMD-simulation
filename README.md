# FluxMD

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
## TL;DR 
FluxMD postulates protein-ligand binding sites as energy sinkholes—regions where molecular forces converge on the protein surface. Unlike traditional docking methods that sample static conformations or giving local residue freedom (scope subject to heuristic designation), FluxMD analyzes the overall dynamic flow of interaction energy through continuous trajectories, revealing how proteins channel ligand binding through their inherent force fields.

Mind that uma versions indicate FluxMD for unified memory architecture (UMA) hardwares—i.e. Apple Silicon. FluxMD recognizes and leverages to facilitate heavy simulations.


## The Physics of Molecular Recognition

Imagine a protein surface as a dynamic energy landscape, shaped by countless atomic forces that create valleys, ridges, and—most importantly—sinkholes where molecular interactions naturally converge. These energy sinkholes are nature's binding sites, sculpted by evolution to capture specific molecular partners with exquisite precision.

**FluxMD** reveals these hidden binding sites by following the flow of energy itself. Rather than searching for static pockets or sampling discrete conformations, FluxMD traces how interaction forces propagate across the protein surface, identifying regions where energy flux converges—much like finding whirlpools by watching the flow of water.

### A Cartography for Binding Site Discovery

Traditional docking methods ask: *"Where does the ligand fit?"*  
FluxMD asks: *"Where does the energy flow?"*

By combining dynamic trajectory exploration with physics-based force calculations, FluxMD captures the true energetic character of protein-ligand recognition. The method tracks how proteins channel interaction forces through their three-dimensional structure, revealing not just where ligands bind, but *why* they bind there—through the convergence of electrostatic attractions, hydrogen bonds, van der Waals forces, and aromatic interactions into focused energy sinkholes.

### Key Innovation: The Unified Force Field

FluxMD's breakthrough lies in treating proteins as pre-stressed mechanical systems. Just as architectural structures distribute loads through tension and compression, proteins maintain an internal force network that guides molecular recognition. FluxMD combines:

- **Static intra-protein forces** that maintain native conformation
- **Dynamic protein-ligand interactions** sampled through winding trajectories
- **Unified vector analysis** that reveals how these forces converge

This unified approach captures phenomena invisible to conventional methods—allosteric sites, cryptic pockets, and conformational selection mechanisms emerge naturally from the physics of force propagation.

### Beyond Geometry: Understanding Binding Through Physics

While pocket-based methods rely on shape complementarity, FluxMD operates on fundamental physical principles:
Φᵢ = ⟨|E̅ᵢ|⟩ · Cᵢ · (1 + τᵢ)
Where binding sites emerge not from geometric cavities, but from the convergence of force vectors—regions where the protein's internal stress field aligns with favorable ligand interactions to create thermodynamic sinkholes.

## Method

FluxMD combines static intra-protein force fields with dynamic protein-ligand interactions to identify binding sites. The method:

1. Pre-calculates internal protein forces that define native conformation (one-time computation)
2. Generates winding trajectories that spiral around the protein like thread (we here call this cocoon_trajectory)
3. Samples multiple ligand orientations (36 rotations) at each trajectory position
4. Calculates combined force vectors at each residue
5. Identifies binding sites where forces converge
6. Validates results using bootstrap statistical analysis


## Features

- **Winding trajectories**: Ligand winds around protein like thread, exploring entire surface geometry
  - Free distance variation from 5Å to 2.5×target distance (e.g., 5-50Å)
  - Thread-like motion using spherical coordinates with angular momentum
  - Natural oscillatory in/out motion for organic exploration
  - Samples 36 rotations at each trajectory position
  - Principal axes alignment for protein-shape-aware winding
- **Physics-based motion**: Molecular weight-dependent diffusion (40 fs time step)
  - Corrected molecular radius calculation: r ≈ 0.66 × MW^(1/3) Å
  - True Brownian dynamics with distance constraints
- **Protonation-aware interactions**: pH-dependent H-bond donor/acceptor assignment and charge states
- **Complete interaction detection**: H-bonds (3.5 Å), salt bridges (4.0 Å), π-π stacking (4.5 Å), π-cation (6.0 Å), VDW (1-5 Å)
- **Intra-protein force field**: Static internal protein forces combined with ligand interactions
- **Intelligent GPU/CPU selection**: Automatically chooses optimal processing based on:
  - System size and complexity
  - Available hardware (Apple Silicon MPS, NVIDIA CUDA, or CPU)
  - Optional performance benchmarking for accurate selection
  - Memory constraints and workload characteristics
- **True parallel processing**:
  - GPU: Processes multiple rotations simultaneously in batches
  - CPU: Uses all available cores with joblib parallelization
- **Statistical validation**: Bootstrap confidence intervals and p-values
- **Parameter reuse**: Load trajectory parameters from previous simulations for consistent comparisons

## Installation

```bash
git clone https://github.com/yourusername/FluxMD.git
cd FluxMD
conda create -n fluxmd python=3.8
conda activate fluxmd
pip install -e .  # Install FluxMD package in development mode
conda install -c conda-forge openbabel
```

### Test Installation
```bash
# Test Apple Silicon GPU detection
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Test FluxMD installation
fluxmd --help

# Full system check
fluxmd
# The startup will show GPU detection status
```

### Requirements

- Python 3.8+
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- BioPython >= 1.79
- PyTorch >= 2.0.0 (for GPU acceleration)
- Joblib >= 1.0.0
- NetworkX (for aromatic ring detection)
- OpenBabel (for file conversions)

## Recent Improvements

### Winding Trajectory Implementation (v2.1)
- **Thread-like motion**: Complete redesign - ligand now winds around protein like thread
- **Free distance variation**: 5Å to 2.5×target distance range (e.g., 5-50Å)
- **Spherical coordinates**: Angular momentum system for natural spiraling motion
- **Oscillatory dynamics**: Natural in/out breathing motion during winding
- **Principal axes alignment**: Uses PCA to align winding with protein shape
- **Soft boundaries**: Gentle forces instead of hard constraints

### Improved Aromatic Detection (v2.1)
- **SMILES conversion via NCI CACTUS**: Preserves aromatic bonds and generates proper 3D structures
  - Creates SDF files with explicit aromatic bond information (bond type 4)
  - Ensures planar geometry for aromatic rings
  - Generates all atoms including hydrogens (e.g., C6H6 for benzene)
- **NetworkX integration**: Graph-based aromatic ring detection for accurate π-π stacking
- **PDBQT parser fix**: Recognizes 'A' element as aromatic carbon
- **Simplified OpenBabel**: Removed complex workarounds, now just a basic fallback

### Fixed GPU/CPU Performance (v2.1)
- **Apple Silicon GPU detection fixed**: Now properly detects MPS on all macOS versions
- **GPU now truly parallel**: Processes rotations in batches instead of sequentially
- **CPU parallelization**: Uses all cores with joblib for rotation sampling
- **Smart selection**: Benchmarks actual performance instead of arbitrary rules
- **Memory aware**: Considers GPU memory limits before selection

### Integrated GPU Optimization (v2.2)
- **Scatter operations**: GPU-native scatter_add operations replace Python loops
- **240x speedup**: Process 12M interactions/second vs 50K originally
- **Zero CPU sync**: Eliminates .item() calls that stall GPU pipeline
- **Seamlessly integrated**: Optimizations built directly into core modules
- **Memory efficient**: Pre-allocates tensors to avoid repeated allocations
- **Automatic activation**: Uses optimized path automatically when GPU is enabled

### Unified Memory Architecture (UMA) Optimization (v2.3)
- **Zero-copy processing**: Leverages shared CPU/GPU memory on Apple Silicon
- **No file I/O**: Entire pipeline stays in GPU memory from start to finish
- **Direct tensor passing**: Modules communicate via GPU tensors, not files
- **100x+ speedup**: Eliminates all data transfer bottlenecks
- **Automatic on M-series**: Detects and optimizes for Apple Silicon automatically
- **Backward compatible**: Original pipeline still available when needed

## Code Structure

```
FluxMD/
├── fluxmd.py                    # Main entry point with interactive CLI
├── fluxmd_uma.py                # UMA-optimized zero-copy entry point
├── pyproject.toml               # Package configuration and dependencies
├── requirements.txt             # Core dependencies
├── requirements-dev.txt         # Development dependencies
├── setup.py                     # Legacy setup script
│
├── fluxmd/                      # Main package directory
│   ├── __init__.py             # Package initialization
│   ├── __version__.py          # Version information
│   ├── cli.py                  # Command-line interface definitions
│   │
│   ├── core/                   # Core computational modules
│   │   ├── __init__.py
│   │   ├── trajectory_generator.py          # Winding trajectory engine
│   │   ├── trajectory_generator_uma.py      # UMA-optimized trajectories
│   │   ├── intra_protein_interactions.py    # Static force field calculator
│   │   └── protonation_aware_interactions.py # pH-dependent interactions
│   │
│   ├── gpu/                    # GPU acceleration modules
│   │   ├── __init__.py
│   │   ├── gpu_accelerated_flux.py     # Standard GPU acceleration
│   │   └── gpu_accelerated_flux_uma.py # UMA zero-copy GPU pipeline
│   │
│   ├── analysis/               # Analysis and statistics
│   │   ├── __init__.py
│   │   ├── flux_analyzer.py          # Statistical analysis & visualization
│   │   └── flux_analyzer_uma.py      # UMA-optimized flux analysis
│   │
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   └── dna_to_pdb.py            # DNA structure generator
│   │
│   └── visualization/          # Visualization tools
│       ├── __init__.py
│       └── visualize_multiflux.py   # Multi-protein comparison plots
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_flux_analyzer_example.py
│   └── test_uma_optimization.py
│
├── examples/                   # Usage examples
│   ├── basic_usage.py         # Simple example script
│   └── README.md              # Examples documentation
│
├── benchmarks/                 # Performance benchmarks
│   ├── benchmark_uma.py       # UMA vs standard comparison
│   └── README.md              # Benchmark documentation
│
└── scripts/                    # Utility scripts
    ├── continue_analysis.py   # Resume interrupted runs
    ├── process_completed_iterations.py
    └── README.md              # Scripts documentation
```

### Core Modules

#### Main Entry Points
- **`fluxmd.py`** - Interactive command-line interface
  - File format conversions (CIF→PDB, SMILES→PDB, DNA→PDB)
  - Parameter configuration with smart GPU/CPU selection
  - Coordinates standard analysis pipeline

- **`fluxmd_uma.py`** - UMA-optimized entry point
  - Zero file I/O for 100x+ speedup
  - Direct GPU memory pipeline
  - Command-line arguments for automation

#### Core Package (`fluxmd/core/`)
- **`trajectory_generator.py`** - Winding trajectory simulation
  - Thread-like motion spiraling around protein
  - Free distance variation (5Å to 2.5×target)
  - Collision detection using VDW radii
  - pH-aware interaction detection

- **`intra_protein_interactions.py`** - Static force field
  - Pre-computes n×n residue interaction matrix
  - One-time O(n²) calculation
  - pH-dependent H-bonds and salt bridges

#### GPU Acceleration (`fluxmd/gpu/`)
- **`gpu_accelerated_flux.py`** - Standard GPU module
  - Auto-detects Apple Silicon (MPS) or NVIDIA CUDA
  - Integrated scatter operations for speedup
  - Memory-aware algorithm selection

- **`gpu_accelerated_flux_uma.py`** - UMA optimization
  - Zero-copy operations on unified memory
  - Direct tensor passing between modules
  - Eliminates all file I/O bottlenecks

#### Analysis (`fluxmd/analysis/`)
- **`flux_analyzer.py`** - Statistical analysis
  - Energy flux calculation: Φᵢ = ⟨|E̅ᵢ|⟩ · Cᵢ · (1 + τᵢ)
  - Bootstrap validation (1000 iterations)
  - Heatmap visualization generation

#### Utilities (`fluxmd/utils/`)
- **`dna_to_pdb.py`** - DNA structure generator
  - Creates B-DNA double helix from sequence
  - Automatic Watson-Crick pairing
  - Full atomic detail with backbone

### Key Features by Module

| Module | Primary Function | Key Features |
|--------|-----------------|--------------|
| fluxmd.py | Workflow control | Interactive CLI, file conversions, smart GPU detection |
| fluxmd_uma.py | UMA entry point | Zero file I/O, command-line interface, 100x+ speedup |
| core/trajectory_generator.py | Winding trajectories | Thread-like motion, collision detection, pH-aware |
| core/intra_protein_interactions.py | Static forces | n×n residue matrix, pH-dependent interactions |
| core/protonation_aware_interactions.py | pH calculations | Henderson-Hasselbalch, dynamic protonation states |
| gpu/gpu_accelerated_flux.py | GPU acceleration | MPS/CUDA detection, scatter ops, memory-aware |
| gpu/gpu_accelerated_flux_uma.py | UMA optimization | Zero-copy tensors, unified memory, no file I/O |
| analysis/flux_analyzer.py | Statistical analysis | Bootstrap validation, heatmaps, p-values |
| analysis/flux_analyzer_uma.py | UMA analysis | Direct GPU processing, scatter operations |
| utils/dna_to_pdb.py | DNA generator | B-DNA helix, Watson-Crick pairing, full atoms |
| visualization/visualize_multiflux.py | Multi-protein plots | Publication figures, comparative analysis |

## Usage

### Quick Start - Test with Benzene
```bash
# Use the main program
fluxmd
# Choose option 2, enter "c1ccccc1" for benzene
```

### Generate DNA Structure
```bash
# From command line - generates double helix with automatic complement
fluxmd-dna ATCGATCG -o my_dna.pdb

# Example: input ATCG generates:
# Strand 1 (5'→3'): ATCG
# Strand 2 (3'→5'): CGAT (automatic complement)

# Or through main menu
fluxmd
# Choose option 3, enter DNA sequence
```

### Run Complete Workflow
```bash
# Standard workflow (with file I/O)
fluxmd
# Choose option 1 for full analysis

# UMA-optimized workflow (zero file I/O, 100x faster)
fluxmd-uma protein.pdb ligand.pdb -o results_uma
```

### Run Individual Components
```bash
# Generate trajectories only
python -m fluxmd.core.trajectory_generator

# Analyze existing trajectory data
python -m fluxmd.analysis.flux_analyzer
```

Input files:
- Protein: PDB, CIF, or mmCIF
- Ligand: PDB, PDBQT, SMILES, or DNA sequence

SMILES conversion:
- Primary: NCI CACTUS web service (preserves aromaticity, generates proper 3D coordinates)
  - Creates both PDB and SDF files (SDF contains aromatic bond information)
  - Correctly handles aromatic systems with planar geometry
  - For benzene: generates all 12 atoms (6C + 6H) in hexagonal arrangement
- Fallback: OpenBabel (simplified local method, may have aromatic issues)

DNA sequence conversion:
- Direct generation of B-DNA double helix from sequence (e.g., ATCGATCG)
- Automatic complementary strand generation (A↔T, G↔C)
- Pure Python implementation with proper Watson-Crick base pairing
- Canonical B-DNA parameters: 3.38 Å rise, 34.3° twist per base
- Creates full atomic structure with bases, sugars, and phosphates
- Proper intertwined double helix with antiparallel strands
- CONECT records for backbone connectivity
- Use as "ligand" input for protein-DNA binding site analysis

The program guides you through parameter selection, including:
- pH for protonation state calculations (default 7.4)
- Number of iterations and approach distances
- Rotations per position (default 36)
- GPU acceleration options (auto-detects Apple Silicon MPS or NVIDIA CUDA)
- All parameters are saved to `simulation_parameters.txt`

### Reusing Parameters for Comparisons
When comparing different ligands or proteins, you can reuse trajectory parameters from a previous simulation:
```
Load parameters from existing simulation? (y/n): y
Enter path to simulation_parameters.txt: /path/to/previous/simulation_parameters.txt
```
This ensures consistent trajectory conditions (steps, approaches, rotations, pH) across different runs, enabling fair comparisons while using different protein/ligand files.

## Output

```
flux_analysis/
├── simulation_parameters.txt              # Complete record of simulation settings
├── processed_flux_data.csv               # Binding site rankings with statistics
├── {protein}_flux_report.txt             # Detailed statistical analysis
├── {protein}_trajectory_flux_analysis.png # Flux heatmap visualization
├── all_iterations_flux.csv               # Raw flux data from all iterations
└── iteration_*/                          # Per-iteration trajectory data
    ├── trajectory_iteration_*_approach_*.png # Cocoon trajectory visualizations
    ├── trajectory_iteration_*_approach_*.csv # Trajectory coordinates
    ├── interactions_approach_*.csv        # Detailed interactions with rotations
    └── flux_iteration_*_output_vectors.csv # Combined interaction vectors
```

### Output Data Columns

**processed_flux_data.csv** contains:
- `residue_index`, `residue_name`: Residue identification
- `average_flux`, `std_flux`: Flux statistics across iterations
- `inter_protein_flux`: Contribution from protein-ligand interactions
- `intra_protein_flux`: Contribution from internal protein forces
- `inter_intra_ratio`: Ratio indicating dominant force type
- `p_value`, `is_significant`: Statistical significance (p < 0.05)
- `ci_lower_95`, `ci_upper_95`: Bootstrap confidence intervals
- `is_aromatic`: Marks residues capable of π-π stacking
- `analysis_pH`: pH used for protonation state calculations

Red regions in visualizations indicate high-flux binding sites with statistical significance.

## Visualization

FluxMD provides matplotlib visualization for creating publication-quality figures:

### From Terminal
```bash
# Interactive mode
python visualize_multiflux.py

# Command-line mode
python visualize_multiflux.py \
  --proteins GPX4-wt.pdb GPX4-single.pdb GPX4-double.pdb \
  --fluxes wt_flux.csv single_flux.csv double_flux.csv \
  --labels "GPX4-WT" "GPX4-Single" "GPX4-Double" \
  --output comparison.png
```

**Features:**
- Publication-ready PNG output
- Smooth spline-interpolated ribbons using BioPython
- Grid layout for multiple proteins
- Consistent viewing angles
- Berlin color palette (blue-white-red diverging colormap)
- Best for figures and presentations

### File Requirements

- **PDB files**: Your protein structures
- **CSV files**: The `processed_flux_data.csv` file from FluxMD analysis (typically in `flux_analysis/` directory)

## Performance

### Dynamic GPU/CPU Selection
FluxMD automatically selects the optimal processing method:

1. **Initial estimation** based on:
   - System size (protein/ligand atoms)
   - Number of rotations and frames
   - Available hardware and memory

2. **Optional benchmarking**:
   - Run actual performance test on your system
   - Measures real GPU vs CPU speed
   - Makes data-driven decision

### Performance Characteristics
- **GPU excels**: Large proteins (>10K atoms) with moderate rotations (<24)
- **CPU excels**: Small systems or many rotations (>36)
- **Parallel processing**: Both GPU and CPU use parallel algorithms
- **Integrated optimization**: GPU mode automatically uses scatter operations for 100x+ speedup

### GPU Performance

FluxMD includes two levels of GPU optimization:

#### 1. Integrated GPU Optimization (v2.2)
Built-in scatter operations eliminate Python loops and CPU-GPU synchronization:

- **Original**: ~50K interactions/second (bottlenecked by Python loops)
- **Optimized**: ~12M interactions/second (240x speedup)
- **Automatic**: No configuration needed, just use GPU mode

#### 2. UMA Optimization (v2.3) - NEW!
Zero-copy pipeline for Unified Memory Architecture (Apple Silicon, etc.):

```bash
# Use UMA-optimized version for maximum performance
fluxmd-uma protein.pdb ligand.pdb -o results_uma
```

**Benefits of UMA optimization:**
- **Zero file I/O**: No CSV files written or read
- **Direct GPU pipeline**: Data flows GPU → GPU → GPU
- **Shared memory**: CPU and GPU access same memory (no copying)
- **100x+ faster**: Eliminates all transfer bottlenecks

**Performance comparison (Apple M1 Pro):**
| Pipeline | 100K interactions | 5M interactions | File I/O |
|----------|------------------|-----------------|----------|
| Original | 2.3 sec | 115 sec | Yes (CSV) |
| GPU Optimized | 0.08 sec | 0.9 sec | Yes (CSV) |
| UMA Optimized | 0.03 sec | 0.4 sec | None |

**When to use each version:**
- `fluxmd.py`: General use, compatibility, smaller datasets
- `fluxmd_uma.py`: Maximum performance, large datasets, Apple Silicon

## Theory

FluxMD calculates energy flux differential Φᵢ for each residue using combined force analysis:

**Φᵢ = ⟨|E̅ᵢ|⟩ · Cᵢ · (1 + τᵢ)**

where:
- **E̅ᵢ = E_inter + E_intra** (combined force vector)
- **⟨|E̅ᵢ|⟩** = mean magnitude of combined energy vectors
- **Cᵢ** = directional consistency (0-1)
- **τᵢ** = temporal fluctuation rate

The method uniquely considers both:
1. **Inter-protein forces** (E_inter): Dynamic protein-ligand interactions
2. **Intra-protein forces** (E_intra): Static internal protein stress field

High Φᵢ indicates energy convergence where internal protein forces align with ligand interactions, revealing true binding sites. This combined approach captures how proteins are "pre-stressed" to recognize specific ligands.

### Non-Covalent Interaction Detection

FluxMD detects and quantifies the following interactions with specific cutoffs and energy calculations:

#### 1. Hydrogen Bonds (pH-Dependent)
- **Distance cutoff**: 3.5 Å (heavy atom to heavy atom)
- **Angle cutoff**: >120° (D-H-A angle)
- **Energy**: E = -2.0 · cos²(θ) · exp(-r/2.0) kcal/mol
- **pH-aware criteria**: 
  - Donor/acceptor roles determined by protonation state
  - HIS, ASP, GLU can be donors when protonated
  - LYS, ARG can be acceptors when deprotonated
  - Bidirectional checking for all potential H-bonds

#### 2. Salt Bridges (pH-Dependent Ionic Interactions)
- **Distance cutoff**: 4.0 Å (between charged groups)
- **Energy**: E = 332.0 · q₁ · q₂ / (ε · r) kcal/mol
- **pH-dependent charges**: 
  - ASP (pKa 3.9): Negative at pH > 4.9
  - GLU (pKa 4.2): Negative at pH > 5.2
  - HIS (pKa 6.0): Positive at pH < 5.0, neutral/positive near pH 7.4
  - LYS (pKa 10.5): Positive at pH < 11.5
  - ARG (pKa 12.5): Always positive at physiological pH
  - CYS (pKa 8.3): Can be negative at pH > 9.3

#### 3. π-π Stacking
- **Distance cutoff**: 4.5 Å (ring centroid to centroid)
- **Angle ranges**:
  - Parallel: 0-30° → E = -5.0 kcal/mol
  - T-shaped: 60-120° → E = -4.0 kcal/mol
  - Offset/Angled: 30-60° → E = -3.5 to -2.5 kcal/mol
- **Aromatic residues**: PHE, TYR, TRP, HIS
- **Energy**: Smooth interpolation based on geometry

#### 4. π-Cation Interactions
- **Distance cutoff**: 6.0 Å (cation to ring centroid)
- **Energy**: E = -2.0 to -5.0 kcal/mol (distance-dependent)
- **Cations**: ARG (guanidinium), LYS (NH3+), charged ligand atoms
- **Aromatic systems**: PHE, TYR, TRP rings

#### 5. Van der Waals Forces
- **Distance range**: 1.0-5.0 Å
- **Energy**: Lennard-Jones 6-12 potential
  - E = 4ε[(σ/r)¹² - (σ/r)⁶]
  - Attractive at optimal distance (σ)
  - Repulsive at very short distances
- **Parameters**: Atom-type specific ε and σ values

### Force Field Implementation

All interactions are computed using smooth, differentiable functions to ensure:
- Continuous energy landscapes
- Proper force vector calculations (F = -∇E)
- Numerical stability in GPU implementations
- Physical realism in trajectory simulations
- pH-dependent protonation states via Henderson-Hasselbalch equation

### Protonation State Calculations

FluxMD uses the Henderson-Hasselbalch equation to determine protonation states:
- For acids: fraction_protonated = 1 / (1 + 10^(pH - pKa))
- For bases: fraction_protonated = 1 / (1 + 10^(pKa - pH))
- Atoms with >50% protonation probability are assigned their protonated state
- This affects H-bond donor/acceptor roles and formal charges

### Winding Trajectory Implementation

FluxMD uses winding trajectories that spiral around the protein like thread:

#### Winding Mode Features
- **Thread-like motion**: Spirals around protein using spherical coordinates
- **Free distance variation**: 5Å to 2.5×target distance (e.g., 5-50Å)
- **Angular momentum**: Smooth winding with momentum-based dynamics
- **Natural oscillation**: Breathing in/out motion during exploration

#### Physics Implementation
- **Stokes-Einstein equation**: D = k_B × T / (6π × η × r)
- **Molecular radius**: r ≈ 0.66 × MW^(1/3) Å
  - Gives ~4.4 Å for MW=300 Da (realistic for drug-like molecules)
- **Diffusion coefficient**: ~7.4×10^-5 Å²/fs for 300 Da molecule at 36.5°C
- **Step size**: Δx = √(2D × Δt) with distance constraint applied

#### Trajectory Algorithm
1. Update angular position with momentum (theta, phi)
2. Update radial distance with oscillation and momentum
3. Convert spherical to Cartesian coordinates
4. Transform using protein's principal axes
5. Add Brownian noise for additional randomness
6. Check collision and adjust if needed
7. Sample multiple rotations at accepted positions

#### Time Step Considerations
- **40 fs time step**: Appropriate for Brownian dynamics
- **Collision detection**: VDW-based using KD-trees
- **Rotation axis**: Determined by closest Cα atom direction

## Applications & Validation

FluxMD has been tested on diverse protein-ligand systems:
- **Enzyme active sites**: Successfully identifies catalytic pockets in serine proteases
- **Allosteric sites**: Detects cryptic binding sites not visible in crystal structures  
- **Protein-DNA interfaces**: Maps DNA-binding domains using generated B-DNA structures
- **Drug targets**: Validated on FDA-approved drug-target complexes

Key advantages over traditional methods:
- **No prior knowledge required**: Discovers binding sites without pocket detection
- **Handles flexible proteins**: Winding trajectories adapt to protein dynamics
- **pH-aware predictions**: Captures environment-dependent binding preferences
- **Ultra-fast analysis**: UMA optimization enables proteome-scale screening

## Citation

```bibtex
@software{fluxmd2024,
  title={FluxMD: Binding Site Identification via Energy Flux Analysis},
  author={Myunghyun Jeong},
  year={2024},
  url={https://github.com/panelarin/FluxMD}
}
```

## Acknowledgments

FluxMD builds upon the protein-ligand analysis my legacy framework from <https://github.com/jaehee831/protein-folding-analysis-bioinformatics>

## License

MIT License. See LICENSE file.

## Troubleshooting

### Common Issues

1. **GPU not detected on Apple Silicon**
   ```bash
   # Ensure PyTorch 2.0+ with MPS support
   pip install --upgrade torch>=2.0
   
   # Test MPS availability
   python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
   
   # If MPS still not available, try:
   pip uninstall torch
   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
   ```

2. **OpenBabel installation fails**
   ```bash
   # Use conda instead of pip
   conda install -c conda-forge openbabel
   ```

3. **Memory errors on large proteins**
   - The code automatically switches to memory-efficient algorithms for systems >50K atoms
   - Reduce batch size if needed by modifying `batch_size` in gpu_accelerated_flux.py

4. **No interactions detected**
   - Check ligand file format (HETATM records required for PDB ligands)
   - Verify protein and ligand are within interaction distance
   - Try increasing approach distance parameter

5. **SMILES conversion issues with aromatic rings**
   - For aromatic ligands (benzene, pyridine, etc.), use the generated PDB file instead of PDBQT
   - FluxMD now creates both formats and recommends PDB for aromatic systems
   - Verify aromatic atoms are properly connected (check atom count)
   - Install networkx for better aromatic detection: `pip install networkx`

6. **GPU performance issues**
   - The GPU optimization is now integrated - no patches needed!
   - If experiencing slow performance, verify GPU is being used:
     ```python
     python -c "import torch; print(f'Device: {torch.cuda.is_available() or torch.backends.mps.is_available()}')"
     ```
   - Check GPU memory usage: `torch.cuda.memory_summary()` (CUDA) or Activity Monitor (MPS)

## Support

Please reach out to <mhjonathan@gm.gist.ac.kr>

---

Tested on Apple M1. NVIDIA CUDA support requires validation.
