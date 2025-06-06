# FluxMD

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

FluxMD identifies protein-ligand binding sites through energy flux differential analysis. The method treats binding sites as energy sinkholes where molecular interactions converge.

## Method

FluxMD combines static intra-protein force fields with dynamic protein-ligand interactions to identify binding sites. The method:

1. Pre-calculates internal protein forces (one-time computation)
2. Generates **winding trajectories** that spiral around the protein like thread
3. Samples multiple ligand orientations (36 rotations) at each trajectory position
4. Calculates combined force vectors (합벡터) at each residue
5. Identifies binding sites where forces converge
6. Validates results using bootstrap statistical analysis

This approach reveals how proteins' internal stress fields guide ligand recognition.

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
git clone https://github.com/panelarin/FluxMD.git
cd FluxMD
conda create -n fluxmd python=3.8
conda activate fluxmd
pip install -r requirements.txt
conda install -c conda-forge openbabel
pip install torch>=2.0  # For GPU support (Apple Silicon MPS or NVIDIA CUDA)
pip install networkx  # For aromatic ring detection
```

### Test Installation
```bash
# Test Apple Silicon GPU detection
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Test CACTUS service
python test_cactus_benzene.py

# Full system check
python fluxmd.py
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

## Code Structure

### Core Modules

- **`fluxmd.py`** - Main entry point and workflow (user interface)
  - Interactive command-line interface
  - File format conversions (CIF→PDB, SMILES→PDBQT, DNA→PDB)
  - Parameter configuration and validation
  - Coordinates the complete analysis pipeline

- **`trajectory_generator.py`** - Winding trajectory simulation engine
  - **Winding mode**: Thread-like motion spiraling around protein geometry
  - Free distance variation (5Å to 2.5×target, e.g., 5-50Å range)
  - Samples 36 rotations per trajectory position
  - Spherical coordinates with angular momentum for smooth winding
  - Natural oscillatory in/out motion during exploration
  - Collision detection using VDW radii and KD-trees
  - Protonation-aware interaction detection (pH-dependent)
  - Integrated intra-protein force field calculations

- **`gpu_accelerated_flux.py`** - GPU acceleration module
  - Auto-detects Apple Silicon (MPS) or NVIDIA CUDA
  - Spatial hashing for systems >10K atoms
  - Octree optimization for very large systems
  - Batch processing with memory-aware algorithms
  - Supports combined inter/intra force calculations
  - **Integrated optimization**: Scatter operations for 100x+ speedup
  - **Direct flux calculation**: No intermediate file I/O needed
  - **Vector tracking**: Maintains force vectors through pipeline

- **`flux_analyzer.py`** - Statistical analysis and visualization
  - Energy flux differential calculation: Φᵢ = ⟨|Eᵢ|⟩ · Cᵢ · (1 + τᵢ)
  - Bootstrap validation (1000 iterations)
  - P-value and confidence interval computation
  - Heatmap generation and result reporting
  - Tracks separate inter/intra-protein contributions

- **`intra_protein_interactions.py`** - Static protein force field calculator
  - Pre-computes complete n×n residue-residue interaction matrix
  - Calculates all atom-atom forces between every residue pair
  - pH-dependent H-bond and salt bridge detection
  - Tests H-bonds, salt bridges, π-π, π-cation, VDW for each atom pair
  - Generates comprehensive residue-level force vectors
  - One-time O(n²) calculation, then O(1) lookup during trajectory

- **`protonation_aware_interactions.py`** - pH-dependent interaction detection
  - Henderson-Hasselbalch calculations for ionizable residues
  - Dynamic donor/acceptor role assignment based on pH
  - Handles ASP, GLU, HIS, LYS, ARG, CYS, TYR protonation states
  - Critical for accurate H-bond and salt bridge detection

- **`dna_to_pdb.py`** - DNA sequence to structure converter
  - Generates B-DNA double helix from sequence
  - Pure Python implementation with numpy
  - Automatic complementary strand generation
  - Watson-Crick base pairing (A-T, G-C)
  - Full atomic detail with proper sugar-phosphate backbone
  - Proper intertwined double helix geometry
  - Command-line tool for easy use
  - Enables protein-DNA binding site analysis

### Key Features by Module

| Module | Primary Function | Key Features |
|--------|-----------------|--------------|
| fluxmd.py | Workflow control | File conversions, GPU detection, pH input, user interface |
| trajectory_generator.py | Dynamics simulation | Brownian motion, collision detection, pH-aware interactions |
| gpu_accelerated_flux.py | GPU acceleration | Auto-detect GPU, integrated optimization, direct flux calc |
| flux_analyzer.py | Results analysis | Statistical validation, visualization, pH tracking |
| intra_protein_interactions.py | Internal forces | Complete n×n residue matrix, pH-aware interactions |
| protonation_aware_interactions.py | pH-dependent states | Henderson-Hasselbalch, donor/acceptor assignment |
| dna_to_pdb.py | DNA structure generation | B-DNA double helix, automatic complement, full atomic detail |

## Usage

### Quick Start - Test with Benzene
```bash
# Test CACTUS service with benzene
python test_cactus_benzene.py

# Or use the main program
python fluxmd.py
# Choose option 2, enter "c1ccccc1" for benzene
```

### Generate DNA Structure
```bash
# From command line - generates double helix with automatic complement
python dna_to_pdb.py ATCGATCG -o my_dna.pdb

# Example: input ATCG generates:
# Strand 1 (5'→3'): ATCG
# Strand 2 (3'→5'): CGAT (automatic complement)

# Or through main menu
python fluxmd.py
# Choose option 3, enter DNA sequence
```

### Run Complete Workflow
```bash
python fluxmd.py
# Choose option 1 for full analysis
```

### Run Individual Components
```bash
# Generate trajectories only
python trajectory_generator.py

# Analyze existing trajectory data
python flux_analyzer.py
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

FluxMD now includes integrated GPU optimization that dramatically improves performance:

**Automatic optimization:**
The GPU acceleration module now includes built-in scatter operations that eliminate Python loops and CPU-GPU synchronization. No patches or additional imports needed - just enable GPU mode!

**Performance improvements:**
- **Original**: ~50K interactions/second (bottlenecked by Python loops)
- **Optimized**: ~12M interactions/second (240x speedup)
- **Key technique**: PyTorch scatter operations replace all Python loops

Example performance (Apple M1 Pro):
| Workload | Original GPU | Integrated GPU | Speedup |
|----------|--------------|----------------|---------|
| 100K interactions | 2.3 sec | 0.08 sec | 29x |
| 5M interactions | 115 sec | 0.9 sec | 128x |
| 50M interactions | >10 min | 4.5 sec | >130x |

## Theory

FluxMD calculates energy flux differential Φᵢ for each residue using combined force analysis:

**Φᵢ = ⟨|E̅ᵢ|⟩ · Cᵢ · (1 + τᵢ)**

where:
- **E̅ᵢ = E_inter + E_intra** (합벡터 - combined force vector)
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
