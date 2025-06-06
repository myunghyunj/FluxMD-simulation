# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Running the Analysis
```bash
# Run complete FluxMD workflow (interactive) - main entry point
python fluxmd.py

# Generate trajectories directly (advanced usage)
python trajectory_generator.py
```

### Installation and Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install OpenBabel for molecular file conversions (required)  
conda install -c conda-forge openbabel

# Test GPU detection (Apple Silicon MPS or NVIDIA CUDA)
python -c "import torch; print(f'GPU: {torch.cuda.is_available() or torch.backends.mps.is_available()}')"

# Detailed GPU check for Apple Silicon
python -c "import torch, platform; print(f'Platform: {platform.platform()}'); print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### File Conversions
```bash
# Convert CIF to PDB
obabel input.cif -O output.pdb

# Convert SMILES to PDB (for ligand input)
# Method 1: Using NCI CACTUS (recommended - preserves aromaticity)
python fluxmd.py  # Choose option 2, then option 1
# Creates both .pdb and .sdf files with aromatic bonds preserved

# Method 2: Using OpenBabel (local fallback)
python fluxmd.py  # Choose option 2, then option 2
# Simple conversion, may have issues with aromatics
```

### Visualization
```bash
# Create publication-quality figures
python visualize_multiflux.py
```

### Testing and Validation
```bash
# Test GPU cutoff algorithms
python tests/test_gpu_cutoffs.py

# Validate trajectory generation
python trajectory_generator.py  # Has interactive test mode

# Check SMILES to PDBQT conversion
python fluxmd.py  # Option 2 in menu
```

## Architecture Overview

### Core Concept
FluxMD identifies protein-ligand binding sites through energy flux differential analysis. The method uniquely combines:
- **Inter-protein forces** (E_inter): Dynamic protein-ligand interactions during trajectory simulation
- **Intra-protein forces** (E_intra): Static internal protein stress field pre-calculated once
- **Combined force analysis** (합벡터): E̅ᵢ = E_inter + E_intra reveals how proteins are "pre-stressed" to recognize specific ligands

### Energy Flux Formula
**Φᵢ = ⟨|E̅ᵢ|⟩ · Cᵢ · (1 + τᵢ)**
- ⟨|E̅ᵢ|⟩ = mean magnitude of combined energy vectors
- Cᵢ = directional consistency (0-1)
- τᵢ = temporal fluctuation rate

### Core Analysis Pipeline
FluxMD implements a physics-based approach to binding site identification through energy flux differential analysis:

1. **Main Orchestrator** (`fluxmd.py`)
   - Interactive workflow with parameter selection
   - Automatic file format detection and conversion
   - GPU detection and configuration
   - Results validation and reporting

2. **Trajectory Generation** (`trajectory_generator.py`)
   - Brownian motion simulation with molecular weight-dependent diffusion
   - Collision detection using VDW radii
   - Surface point generation via KD-trees
   - Multiple approach angles (default: 100) and rotations

3. **GPU Acceleration** (`gpu_accelerated_flux.py`)
   - Auto-detects Apple Silicon (MPS) or NVIDIA CUDA
   - Spatial hashing for systems >10K atoms
   - Octree optimizations for very large systems
   - Unified memory management on Apple Silicon
   - Fallback to CPU with joblib parallelization

4. **Flux Analysis** (`flux_analyzer.py`)
   - Energy flux differential calculation: Φᵢ = ⟨|Eᵢ|⟩ · Cᵢ · (1 + τᵢ)
   - Bootstrap statistical validation (1000 iterations)
   - P-value and effect size computation
   - Heatmap visualization generation

### Key Technical Components

**Interaction Detection**
- Hydrogen bonds: Distance <3.5Å, angle >120°
- Salt bridges: Charged residue pairs <4.0Å
- π-π stacking: Aromatic ring centroids <4.5Å
- π-cation interactions: Cation-aromatic <6.0Å
- Van der Waals: 1.0-5.0Å with proper scaling
- Graph-based aromatic ring detection using networkx

**Robust Parsing** (`parse_ligand_robust.py`)
- Primary: BioPython PDBParser
- Fallback: Manual atom-by-atom parsing
- Validation: Compares parsed vs expected atom counts
- Special handling for problematic ligands (e.g., ML162)

**Statistical Framework**
- Bootstrap: 1000 iterations with replacement
- Confidence intervals: 95% (2.5-97.5 percentile)
- P-values: Fraction of bootstrap samples ≤ 0
- Effect size: Cohen's d relative to background

**Performance Optimizations**
- Hierarchical distance filtering
- Batch processing with configurable size
- Memory-aware algorithm selection
- Automatic CPU fallback for unsupported GPUs

### Output Structure
```
results/
├── processed_flux_data.csv          # Ranked binding sites with statistics
├── {protein}_flux_report.txt        # Detailed statistical analysis
├── {protein}_trajectory_flux_analysis.png  # Flux visualization heatmap
└── iteration_*/                     # Raw trajectory data per approach
    ├── trajectory_data_*.csv        # Frame-by-frame positions
    └── interaction_data_*.csv       # Detected interactions
```

### Dependencies
Core dependencies in requirements.txt:
- numpy>=1.20.0 (array operations)
- pandas>=1.3.0 (data management)
- scipy>=1.7.0 (spatial algorithms)
- matplotlib>=3.4.0 (visualizations)
- biopython>=1.79 (structure parsing)
- torch>=2.0.0 (GPU acceleration)
- joblib>=1.0.0 (CPU parallelization)

Additional dependencies imported but not in requirements.txt:
- networkx (aromatic ring detection - install with `pip install networkx`)
- openbabel (file conversions - install with `conda install -c conda-forge openbabel`)

## Key Workflow Steps

1. **Pre-computation**: Calculate static intra-protein force field using `intra_protein_interactions.py` (one-time)
2. **Trajectory Generation**: Create winding paths around protein with collision detection
3. **Rotation Sampling**: Test 36 ligand orientations at each trajectory position
4. **Force Calculation**: Compute combined inter+intra-protein forces
5. **Flux Analysis**: Calculate energy flux differentials with bootstrap validation
6. **Visualization**: Generate heatmaps highlighting significant binding sites (red = high flux)

## Important Technical Notes

- **Ligand Parsing**: BioPython fails when ligand PDB files contain duplicate atom names within a residue. The `parse_ligand_robust()` function in multiple modules provides automatic fallback to manual parsing.

- **PDBQT Element Detection**: PDBQT files use special atom types (A/AC=aromatic carbon, NA=aromatic nitrogen). The parser recognizes these and converts them to standard elements for proper interaction calculations.

- **Apple Silicon GPU Detection**: Fixed detection for Apple Silicon MPS. The code now checks `torch.backends.mps.is_available()` directly instead of checking processor type. If MPS is not detected, ensure PyTorch 2.0+ is installed with: `pip install torch>=2.0`. For troubleshooting, run: `python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"`

- **Memory Management**: Large systems (>50K atoms) automatically switch to memory-efficient algorithms including spatial hashing and octree-based distance calculations.

- **SMILES Conversion**: 
  - Primary: NCI CACTUS web service preserves aromaticity and generates proper 3D coordinates
  - Creates both PDB (for FluxMD) and SDF (with aromatic bond info) files
  - Correctly handles benzene as C6H6 with planar hexagonal geometry
  - Fallback: Simplified OpenBabel method (may have aromatic issues)

- **Visualization Colors**: In output heatmaps, red indicates high-flux binding sites with statistical significance (p<0.05), while blue indicates low flux regions.

## Winding Trajectory Implementation (v2.1)

FluxMD now uses "winding" trajectories that explore protein surfaces like thread winding around an object:

- **Spherical coordinate system**: Controls angular motion (theta, phi) with momentum
- **Distance freedom**: Allows 5Å to 2.5×target distance variation (e.g., 5-50Å range)
- **Natural oscillation**: Sinusoidal in/out motion for organic exploration
- **Principal axes alignment**: Uses PCA to align winding with protein shape
- **Momentum dynamics**: Angular and radial movements have momentum for smooth flow
- **Soft boundaries**: Gentle forces instead of hard constraints

This creates more natural, stochastic trajectories that thoroughly explore the protein surface while maintaining physical realism.

## Intra-Protein Force Field Integration

FluxMD uniquely combines two force components:

1. **Inter-protein forces** (E_inter): Dynamic protein-ligand interactions calculated during trajectory simulation
2. **Intra-protein forces** (E_intra): Static internal protein stress field pre-calculated using `intra_protein_interactions.py`

The combined force vector (합벡터) E̅ᵢ = E_inter + E_intra reveals how proteins are "pre-stressed" to recognize specific ligands. The intra-protein module performs a one-time O(n²) calculation of all residue-residue interactions, then provides O(1) lookup during trajectory analysis.

Key technical details:
- Complete n×n residue interaction matrix calculation
- All atom-atom forces between every residue pair
- Covers H-bonds, salt bridges, π-π stacking, π-cation, and VDW interactions
- Results in `intra_protein_flux` column in processed_flux_data.csv

## pH-Dependent Interaction Detection

FluxMD uses Henderson-Hasselbalch equation for protonation states:
- **ASP (pKa 3.9)**: Negative at pH > 4.9
- **GLU (pKa 4.2)**: Negative at pH > 5.2  
- **HIS (pKa 6.0)**: Positive at pH < 5.0, neutral/positive near pH 7.4
- **LYS (pKa 10.5)**: Positive at pH < 11.5
- **ARG (pKa 12.5)**: Always positive at physiological pH
- **CYS (pKa 8.3)**: Can be negative at pH > 9.3
- **TYR (pKa 10.1)**: Usually neutral, can lose proton at high pH

This affects H-bond donor/acceptor roles and charge-based interactions like salt bridges.