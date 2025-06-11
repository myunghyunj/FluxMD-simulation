# FluxMD

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**FluxMD** maps binding interfaces between two biomolecules by tracing the flow of interaction energy. Unlike traditional docking, which samples static conformers, FluxMD follows dynamic energy flux as molecules orbit and engage, exposing regions where forces perturbate. Intrinsically optimized from physics-level to signal-processing code for modern chip architectures—i.e. GPU, UMA—FluxMD enables accelerated high-throughput screening of molecular dynamics. The method applies to protein–protein and protein–ligand systems, with support for protein–nucleic acid interactions underway. Each run produces a **stress barcode**, a reproducible energy signature unique to the molecular pair.

![FluxMD Concept](images/fluxmd_concept.png)

## Program Flow

```mermaid
graph TD
    %% Entry Points
    CLI[Command Line] --> fluxmd[fluxmd Interactive]
    CLI --> fluxmd_uma[fluxmd-uma Direct]
    CLI --> fluxmd_dna[fluxmd-dna]
    
    %% Interactive Menu
    fluxmd --> |Option 1| Standard[Standard Pipeline]
    fluxmd --> |Option 2| UMA[UMA Pipeline]
    fluxmd --> |Option 3| SMILES[SMILES Converter]
    fluxmd --> |Option 4| DNA[DNA Generator]
    
    %% Input Processing
    Standard --> Parse1[Parse Protein<br/>PDB/CIF/mmCIF]
    Standard --> Parse2[Parse Ligand<br/>PDB/SMILES]
    UMA --> Parse1
    UMA --> Parse2
    fluxmd_uma --> Parse1
    fluxmd_uma --> Parse2
    
    Parse2 -->|SMILES| CACTUS[CACTUS API]
    CACTUS -->|Fallback| OpenBabel[OpenBabel]
    
    %% Core Processing
    Parse1 --> IntraForces[Calculate Intra-protein<br/>Forces with REF15]
    Parse2 --> Trajectory[Generate Helical<br/>Trajectories]
    IntraForces --> Forces[Calculate Inter-molecular<br/>Forces with REF15]
    Trajectory --> Forces
    
    %% Analysis Pipeline
    Forces --> |Standard| CSV[Write CSV Files]
    CSV --> FluxCalc[Calculate Flux<br/>Phi = E × C × (1+tau)]
    
    Forces --> |UMA| Tensors[GPU Tensors]
    Tensors --> FluxCalc
    
    FluxCalc --> Bootstrap[Bootstrap Analysis<br/>1000 iterations]
    Bootstrap --> Results[Generate Results]
    
    %% Output
    Results --> Report[Flux Report<br/>.txt]
    Results --> Data[Processed Data<br/>.csv]
    Results --> Viz[Visualization<br/>.png]
    
    %% DNA Pipeline
    DNA --> Sequence[DNA Sequence]
    fluxmd_dna --> Sequence
    Sequence --> BForm[Generate B-DNA<br/>Double Helix]
    BForm --> PDB[Output PDB with<br/>CONECT Records]
```

## Quick Start

```bash
# Interactive mode - recommended for first-time users
fluxmd

# Command-line mode for automation
fluxmd-uma protein.pdb ligand.pdb -o results/

# Generate DNA structure
fluxmd-dna ATCGATCG -o dna_structure.pdb

# Run benchmark
python benchmarks/benchmark_uma.py
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/FluxMD.git
cd FluxMD

# Create conda environment (required for OpenBabel)
conda create -n fluxmd python=3.8
conda activate fluxmd

# Install FluxMD with all features
pip install -e ".[dev,gpu,viz]"

# Install OpenBabel (must use conda)
conda install -c conda-forge openbabel

# For Apple Silicon MPS support (if needed)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Verify Installation

```bash
# Check GPU detection
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test FluxMD
fluxmd --help
```

## Entry Points

FluxMD provides multiple entry points optimized for different use cases:

### 1. `fluxmd` - Interactive Interface
The main entry point with a menu-driven interface:
- **Option 1**: Standard workflow - Complete analysis with file I/O
- **Option 2**: UMA workflow - GPU-accelerated unified memory pipeline  
- **Option 3**: SMILES converter - Chemical structure to PDB
- **Option 4**: DNA generator - Sequence to double helix structure (currently unstable!)
- FluxMD accepts .mmCIf inputs. (AlphaFold server friendly)
  
### 2. `fluxmd-uma` - Command-Line Interface
High-performance command-line tool for automation:
```bash
fluxmd-uma protein.pdb ligand.pdb [options]

Options:
  -o, --output          Output directory
  -p, --params          Load parameters from previous run
  -s, --steps           Steps per trajectory (default: 200)
  -i, --iterations      Number of iterations (default: 10)
  -a, --approaches      Number of approach angles (default: 10)
  -d, --distance        Starting distance in Å (default: 20.0)
  -r, --rotations       Rotations per position (default: 36)
  --ph                  Physiological pH (default: 7.4)
  --cpu                 Force CPU usage
  --save-trajectories   Save trajectory visualizations
```

### 3. SMILES to PDB Converter (via `fluxmd` Option 3)
- **Primary**: NCI CACTUS web service (preserves aromaticity)
- **Fallback**: OpenBabel (local, basic 3D generation)

### 4. `fluxmd-dna` - DNA Structure Generator
```bash
fluxmd-dna ATCGATCG -o dna_structure.pdb
```
- B-DNA double helix with accurate geometry ([Olson et al., 1998](https://www.pnas.org/doi/10.1073/pnas.95.19.11163))
- Complete atomic detail including sugar-phosphate backbone
- CONECT records for all covalent bonds

## Technical Implementation

### Physics Foundation
- **Trajectory Synthesis**: Helical orbits via Brownian dynamics with 40 fs temporal discretization
- **Energy Function**: Rosetta REF15 with all major terms (fa_atr, fa_rep, fa_sol, fa_elec, hbond)
- **Energy Bounds**: ±10 kcal/mol cap prevents singularities while preserving physiological range
- **pH Awareness**: Henderson-Hasselbalch for dynamic protonation states (default pH 7.4)

### Computational Architecture
- **GPU Acceleration**: Automatic detection of Apple MPS or NVIDIA CUDA
- **Memory Optimization**: 
  - Force field parameters in L1 cache
  - Protein chunks sized to GPU shared memory
  - Zero-copy operations on UMA systems
- **Adaptive Algorithms**:
  - Direct computation (<1M atom pairs)
  - Spatial hashing (1M-100M pairs)
  - Hierarchical filtering (>100M pairs)

### REF15 Energy Details
- **Atom Typing**: 167 Rosetta atom types with automatic PDB mapping
- **Energy Terms**:
  - Lennard-Jones (fa_atr/fa_rep): 6Å/4.5Å cutoffs with switching functions
  - Solvation (fa_sol): Lazaridis-Karplus implicit solvent model
  - Electrostatics (fa_elec): Distance-dependent dielectric (ε = 10r)
  - H-bonds (hbond): Orientation-dependent scoring
- **Intelligent Sampling**: Surface property analysis guides trajectory generation
- **Energy Modes**: `simplified` (fast), `ref15` (default), `ref15_fast` (GPU-optimized)

### Statistical Framework
- **Bootstrap Analysis**: 1000 iterations for confidence intervals
- **Significance Testing**: P-values for binding site identification
- **Flux Metric**: Φᵢ = ⟨|E̅ᵢ|⟩ · Cᵢ · (1 + τᵢ)
  - E̅ᵢ: Combined force vector
  - Cᵢ: Directional consistency
  - τᵢ: Temporal variation



## Input/Output

### Supported Input Formats
- **Protein**: PDB, CIF, mmCIF
- **Ligand**: PDB, PDBQT, SMILES (via NCI CACTUS), DNA sequence

### Output Files
```
results/
├── simulation_parameters.txt              # Complete simulation settings
├── processed_flux_data.csv               # Ranked binding sites with statistics
├── {protein}_flux_report.txt             # Detailed analysis report
├── {protein}_trajectory_flux_analysis.png # Heatmap visualization
├── all_iterations_flux.csv               # Raw flux data
└── iteration_*/                          # Per-iteration trajectories
```

### Key Output Columns
- `residue_index`, `residue_name`: Residue identification
- `average_flux`: Mean energy flux (binding site strength)
- `p_value`: Statistical significance (< 0.05 is significant)
- `ci_lower_95`, `ci_upper_95`: 95% confidence intervals
- `inter_intra_ratio`: Ligand vs internal force contribution

## Performance

| Pipeline | File I/O | Memory | Best For |
|----------|----------|---------|----------|
| Standard (`fluxmd`) | CSV files | Lower | Debugging, compatibility |
| UMA (`fluxmd-uma`) | In-memory | Higher | Production, large systems |

- **Standard**: Cross-platform compatibility, easier debugging
- **UMA**: Optimized for unified memory architectures (Apple Silicon)

## How It Works

1. **Trajectory Generation**: Ligand spirals around protein (5-50 Å range, 36 rotations/position)
2. **Force Calculation**: REF15 energy function computes static (intra-protein) and dynamic (protein-ligand) forces
3. **Flux Analysis**: Combines forces to calculate flux metric Φᵢ = ⟨|E̅ᵢ|⟩ · Cᵢ · (1 + τᵢ)
4. **Statistical Validation**: Bootstrap analysis identifies significant binding sites (p < 0.05)

High flux indicates energy convergence where forces align to create favorable binding environments.

## Usage Examples

### Basic Analysis
```bash
# Interactive mode (recommended for beginners)
fluxmd
# Select option 1, follow prompts

# Command-line mode
fluxmd-uma protein.pdb ligand.pdb -o results/
```

### Reuse Parameters
```bash
# Load parameters from previous run for consistent comparison
fluxmd-uma protein.pdb new_ligand.pdb -p previous_results/simulation_parameters.txt
```

### Generate DNA Ligand
```bash
# Create DNA structure
fluxmd-dna GCGATCGC -o dna_ligand.pdb

# Use DNA as ligand
fluxmd-uma protein.pdb dna_ligand.pdb -o dna_binding_results/
```

### Convert SMILES
```bash
# Through interactive interface
fluxmd
# Select option 3, enter SMILES string (e.g., "c1ccccc1" for benzene)
```

### Recover Interrupted Run
```bash
# If analysis was interrupted
python scripts/process_completed_iterations.py results/ protein.pdb

# Continue from existing data
python scripts/continue_analysis.py results/ protein.pdb
```

## Troubleshooting

### GPU Not Detected
```bash
# Update PyTorch for Apple Silicon
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu

# Verify MPS/CUDA
python -c "import torch; print(torch.backends.mps.is_available())"
```

### OpenBabel Installation
```bash
# Must use conda, not pip
conda install -c conda-forge openbabel
```

### Memory Issues
- Use standard pipeline for proteins >100K atoms
- Reduce batch size in GPU settings
- Use recovery scripts if analysis crashes

### No Interactions Detected
- Check ligand format (needs HETATM records)
- Increase approach distance
- Verify protein-ligand proximity

## Citation

```bibtex
@software{fluxmd2025,
  title={FluxMD: Biophysical Cartography for Biomolecular Energy Dynamics},
  author={Myunghyun Jeong},
  year={-},
  url={https://github.com/myunghyunj/FluxMD-simulation}
}
```

## License

MIT License. See [LICENSE](LICENSE) file for details.

## Contact

For questions or issues: mhjonathan@gm.gist.ac.kr

## Acknowledgement

Engineered on top of my previous 2023 term project: https://github.com/jaehee831/protein-folding-analysis-bioinformatics
