# FluxMD

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**FluxMD** is a modern chip architecture optimized computational biology tool that identifies binding sites between two biomolecules—i.e. protein-protein, protein-ligand, and protein-nucleic acid (under work!)—through energy flux analysis. Unlike traditional docking methods that sample static conformations, FluxMD reveals the overall energy dynamics per amino acid residues by hovering the molecules and sample the dynamic flow of non-covalent interaction across protein surfaces—finding where molecular forces naturally flow. Intriguingly, FluxMD results a 'stress barcode', which is unique to input biomolecules, but consistent across various simulation parameters. 

## Key Innovation

FluxMD treats proteins as pre-stressed mechanical systems where binding sites emerge as energy sinkholes. By combining static intra-protein forces (internal strain) with dynamic biomolecular interactions sampled through winding trajectories (a.k.a cocoon trajectory), FluxMD captures phenomena invisible to conventional methods:

- **Cryptic binding sites** that appear through conformational changes
- **Allosteric sites** revealed by force propagation patterns  
- **pH-dependent binding** through protonation state calculations
- **True binding affinity** from thermodynamic convergence

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

## Program Flow

```mermaid
graph TD
    Start[fluxmd Menu] -->|1| Standard[Standard Workflow]
    Standard --> Load1[Load protein PDB/CIF/mmCIF]
    Standard --> Load2[Load ligand or SMILES]
    Load2 -->|SMILES| Convert[Convert to PDB via CACTUS/OpenBabel]
    Standard --> Trajectory[Generate winding trajectories]
    Trajectory --> Forces[Calculate force interactions]
    Forces --> Analysis[Compute energy flux]
    Analysis --> Output[Generate visualizations & reports]

    Start -->|2| UMA[UMA-Optimized Workflow]
    UMA --> GPU[Unified memory GPU pipeline]
    GPU --> Direct[Direct tensor operations]
    Direct --> Fast[Optimized analysis]

    Start -->|3| SMILES[SMILES to PDB Converter]
    SMILES --> CACTUS[NCI CACTUS web service]
    SMILES --> OpenBabel[OpenBabel fallback]
    
    Start -->|4| DNA[DNA Structure Generator]
    DNA --> Builder[B-DNA double helix builder]
```

## Entry Points

FluxMD provides multiple entry points optimized for different use cases:

### 1. `fluxmd` - Interactive Interface
The main entry point with a menu-driven interface:
- **Option 1**: Standard workflow - Complete analysis with file I/O
- **Option 2**: UMA workflow - GPU-accelerated unified memory pipeline  
- **Option 3**: SMILES converter - Chemical structure to PDB
- **Option 4**: DNA generator - Sequence to double helix structure

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

### 3. `fluxmd-dna` - DNA Structure Generator
Generate accurate B-DNA double helix structures:
```bash
fluxmd-dna SEQUENCE [-o output.pdb]

# Example: Generate 8-bp DNA structure
fluxmd-dna ATCGATCG -o dna_structure.pdb
```
- Creates Watson-Crick paired double helix
- Uses crystallographic parameters (Olson et al., 1998)
- Includes full atomic detail with backbone


## Features

### Physics-Based Analysis
- **Winding trajectories**: Ligand spirals around protein exploring all surfaces
- **Force field integration**: Combines static protein forces with dynamic interactions
- **Protonation awareness**: pH-dependent charges and hydrogen bonding
- **Complete interaction types**: H-bonds, salt bridges, π-π stacking, π-cation, VDW

### GPU Acceleration
- **Automatic optimization**: Detects Apple Silicon MPS or NVIDIA CUDA
- **Pipeline-specific approaches**:
  - Standard GPU pipeline: Uses spatial hashing for medium systems (1M-100M atom pairs)
  - UMA pipeline: Direct distance matrix calculation via torch.cdist
- **Smart selection**: Benchmarks actual performance to choose GPU vs CPU
- **Spatial hashing** (standard GPU pipeline only):
  - Automatic algorithm selection based on system size
  - Hash table with linked-list collision handling
  - Batch query support for parallel neighbor lookups

### Statistical Validation
- **Bootstrap analysis**: 1000 iterations for confidence intervals
- **P-value calculation**: Identifies statistically significant sites
- **Flux ranking**: Quantitative binding site prioritization

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

### Standard vs UMA Pipeline

| Pipeline | File I/O | Speed | Memory | Best For |
|----------|----------|-------|---------|----------|
| Standard (`fluxmd`) | Yes (CSV) | 1x | Low | Compatibility, debugging |
| UMA (`fluxmd-uma`) | None | Optimized | High | Production, large datasets |

### Benchmark Results (Apple M1 Pro)
```
Processing 5M interactions:
- Standard pipeline: 115 seconds
- GPU optimized: 0.9 seconds  
- UMA optimized: 0.4 seconds
```

### When to Use Each Version
- **Standard**: Small molecules, debugging, cross-platform compatibility
- **UMA**: Large proteins, high-throughput screening, Apple Silicon systems

## How It Works

### Analysis Pipeline

1. **Trajectory Generation**
   - Ligand spirals around protein like thread on a spool
   - Samples 5-50 Å distance range with oscillatory motion
   - 36 rotations tested at each position

2. **Force Calculation**
   - Static intra-protein forces (pre-computed once)
   - Dynamic protein-ligand interactions at each trajectory point
   - pH-dependent protonation states affect charges and H-bonds
   - Neighbor search approaches:
     - **Standard GPU pipeline**: Adaptive optimization based on system size
       - Small systems (<1M pairs): Direct GPU computation
       - Medium systems (1M-100M pairs): Spatial hashing with O(1) lookups
       - Large systems (>100M pairs): Octree + hierarchical distance filtering
     - **UMA pipeline**: Direct distance matrix via torch.cdist for all system sizes

3. **Flux Analysis**  
   - Combines force vectors at each residue
   - Calculates directional consistency and temporal variation
   - Bootstrap statistics over multiple iterations

4. **Binding Site Identification**
   - High flux regions indicate energy convergence
   - Statistical significance via p-values
   - Ranked list of potential binding sites

### Theory

FluxMD calculates energy flux Φᵢ for each residue:

**Φᵢ = ⟨|E̅ᵢ|⟩ · Cᵢ · (1 + τᵢ)**

Where:
- **E̅ᵢ**: Combined force vector (static + dynamic)
- **⟨|E̅ᵢ|⟩**: Mean force magnitude
- **Cᵢ**: Directional consistency  
- **τᵢ**: Temporal fluctuation

High flux indicates energy convergence—where protein forces align with ligand interactions to create thermodynamic sinkholes.

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
@software{fluxmd2024,
  title={FluxMD: Biophysical Cartography for Biomolecular Energy Dynamics},
  author={Myunghyun Jeong},
  year={2025},
  url={https://github.com/panelarin/FluxMD}
}
```

## License

MIT License. See [LICENSE](LICENSE) file for details.

## Contact

For questions or issues: mhjonathan@gm.gist.ac.kr
