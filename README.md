# FluxMD

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**FluxMD** is a GPU-accelerated computational biology tool that identifies protein-ligand binding sites through energy flux analysis. Unlike traditional docking methods that sample static conformations, FluxMD reveals binding sites by analyzing the dynamic flow of interaction energy across protein surfaces—finding where molecular forces naturally converge.

## Key Innovation

FluxMD treats proteins as pre-stressed mechanical systems where binding sites emerge as energy sinkholes. By combining static intra-protein forces with dynamic protein-ligand interactions sampled through winding trajectories, FluxMD captures phenomena invisible to conventional methods:

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

## Entry Points

FluxMD provides multiple entry points optimized for different use cases:

### 1. `fluxmd` - Interactive Interface
The main entry point with a menu-driven interface:
- **Option 1**: Standard workflow (file-based, compatible with all systems)
- **Option 2**: UMA-optimized workflow (100x faster on Apple Silicon)
- **Option 3**: Convert SMILES to PDB structure
- **Option 4**: Generate DNA double helix from sequence
- **Option 5**: Visualize and compare multiple proteins

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

### 4. Visualization Tools
Compare flux results across multiple proteins:
```bash
# Interactive mode
python -m fluxmd.visualization.visualize_multiflux

# Command-line mode
python -m fluxmd.visualization.visualize_multiflux \
  --proteins wt.pdb mutant.pdb \
  --fluxes wt_flux.csv mutant_flux.csv \
  --labels "Wild-Type" "Mutant" \
  --output comparison.png
```

## Features

### Physics-Based Analysis
- **Winding trajectories**: Ligand spirals around protein exploring all surfaces
- **Force field integration**: Combines static protein forces with dynamic interactions
- **Protonation awareness**: pH-dependent charges and hydrogen bonding
- **Complete interaction types**: H-bonds, salt bridges, π-π stacking, π-cation, VDW

### GPU Acceleration
- **Automatic optimization**: Detects Apple Silicon MPS or NVIDIA CUDA
- **UMA optimization**: Zero-copy processing on unified memory architectures
- **Smart selection**: Benchmarks actual performance to choose GPU vs CPU
- **Massive speedup**: 100-240x faster than CPU for large systems

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
| UMA (`fluxmd-uma`) | None | 100x+ | High | Production, large datasets |

### Benchmark Results (Apple M1 Pro)
```
Processing 5M interactions:
- Standard pipeline: 115 seconds
- GPU optimized: 0.9 seconds  
- UMA optimized: 0.4 seconds (287x speedup)
```

### When to Use Each Version
- **Standard**: Small molecules, debugging, cross-platform compatibility
- **UMA**: Large proteins, high-throughput screening, Apple Silicon systems

## Theory

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
  title={FluxMD: GPU-Accelerated Binding Site Discovery via Energy Flux Analysis},
  author={Myunghyun Jeong},
  year={2024},
  url={https://github.com/panelarin/FluxMD}
}
```

## License

MIT License. See [LICENSE](LICENSE) file for details.

## Contact

For questions or issues: mhjonathan@gm.gist.ac.kr