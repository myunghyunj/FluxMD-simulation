# FluxMD

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

FluxMD identifies protein-ligand binding sites through energy flux differential analysis. The method treats binding sites as energy sinkholes where molecular interactions converge.

## Method

FluxMD combines static intra-protein force fields with dynamic protein-ligand interactions to identify binding sites. The method:

1. Pre-calculates internal protein forces (one-time computation)
2. Simulates ligand approach trajectories using Brownian motion
3. Calculates combined force vectors (합벡터) at each residue
4. Identifies binding sites where forces converge
5. Validates results using bootstrap statistical analysis

This approach reveals how proteins' internal stress fields guide ligand recognition.

## Features

- **Physics-based trajectories**: Brownian motion with molecular weight-dependent diffusion
- **Complete interaction detection**: H-bonds (3.5 Å), salt bridges (4.0 Å), π-π stacking (4.5 Å), π-cation (6.0 Å), VDW (1-5 Å)
- **Intra-protein force field**: Static internal protein forces combined with ligand interactions
- **GPU acceleration**: estimated 10-100x faster on Apple Silicon; actual speedups may vary (CUDA support implemented but untested)
> **Note:** CUDA acceleration has not been validated and performance results are hypothetical.
- **Statistical validation**: Bootstrap confidence intervals and p-values

## Installation

```bash
git clone https://github.com/panelarin/FluxMD.git
cd FluxMD
conda create -n fluxmd python=3.8
conda activate fluxmd
pip install -r requirements.txt
conda install -c conda-forge openbabel
pip install torch  # For GPU support
pip install networkx  # For aromatic ring detection
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

## Code Structure

### Core Modules

- **`fluxmd.py`** - Main entry point and workflow (user interface)
  - Interactive command-line interface
  - File format conversions (CIF→PDB, SMILES→PDBQT)
  - Parameter configuration and validation
  - Coordinates the complete analysis pipeline

- **`trajectory_generator.py`** - Ligand trajectory simulation engine
  - Brownian motion with molecular weight-dependent diffusion
  - Collision detection using VDW radii and KD-trees
  - Surface point generation for approach angles
  - Interaction detection (H-bonds, salt bridges, π-π stacking, etc.)
  - Integrated intra-protein force field calculations

- **`gpu_accelerated_flux.py`** - GPU acceleration module
  - Auto-detects Apple Silicon (MPS) or NVIDIA CUDA
  - Spatial hashing for systems >10K atoms
  - Octree optimization for very large systems
  - Batch processing with memory-aware algorithms
  - Supports combined inter/intra force calculations

- **`flux_analyzer.py`** - Statistical analysis and visualization
  - Energy flux differential calculation: Φᵢ = ⟨|Eᵢ|⟩ · Cᵢ · (1 + τᵢ)
  - Bootstrap validation (1000 iterations)
  - P-value and confidence interval computation
  - Heatmap generation and result reporting
  - Tracks separate inter/intra-protein contributions

- **`intra_protein_interactions.py`** - Static protein force field calculator
  - Pre-computes complete n×n residue-residue interaction matrix
  - Calculates all atom-atom forces between every residue pair
  - Tests H-bonds, salt bridges, π-π, π-cation, VDW for each atom pair
  - Generates comprehensive residue-level force vectors
  - One-time O(n²) calculation, then O(1) lookup during trajectory

### Key Features by Module

| Module | Primary Function | Key Features |
|--------|-----------------|--------------|
| fluxmd.py | Workflow control | File conversions, GPU detection, user interface |
| trajectory_generator.py | Dynamics simulation | Brownian motion, collision detection, interaction mapping |
| gpu_accelerated_flux.py | Performance optimization | 10-100x speedup, memory-efficient algorithms |
| flux_analyzer.py | Results analysis | Statistical validation, visualization, reporting |
| intra_protein_interactions.py | Internal forces | Complete n×n residue matrix, 합벡터 (combined vector) analysis |

## Usage

Run the complete workflow:
```bash
python fluxmd.py
```

Or run individual components:
```bash
# Generate trajectories only
python trajectory_generator.py

# Analyze existing trajectory data
python flux_analyzer.py
```

Input files:
- Protein: PDB, CIF, or mmCIF
- Ligand: PDB, PDBQT, or SMILES

The program guides you through parameter selection.

## Output

```
flux_analysis/
├── processed_flux_data.csv               # Binding site rankings with statistics
├── {protein}_flux_report.txt             # Detailed statistical analysis
├── {protein}_trajectory_flux_analysis.png # Flux heatmap visualization
├── all_iterations_flux.csv               # Raw flux data from all iterations
└── iteration_*/                          # Per-iteration trajectory data
    ├── trajectory_data_*.csv             # Ligand positions per frame
    └── flux_iteration_*_output_vectors.csv # Interaction vectors
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

| System | CPU | GPU (M1) | Speedup |
|--------|-----|----------|---------|
| 10K atoms | 60 min | 3 min | 20x |
| 50K atoms | 8 hours | 15 min | 32x |

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

#### 1. Hydrogen Bonds
- **Distance cutoff**: 3.5 Å (heavy atom to heavy atom)
- **Angle cutoff**: >120° (D-H-A angle)
- **Energy**: E = -2.0 · cos²(θ) · exp(-r/2.0) kcal/mol
- **Criteria**: Donor must have H, acceptor must be N, O, or S

#### 2. Salt Bridges (Ionic Interactions)
- **Distance cutoff**: 4.0 Å (between charged groups)
- **Energy**: E = 332.0 · q₁ · q₂ / (ε · r) kcal/mol
- **Charged residues**: 
  - Positive: ARG (NH1, NH2), LYS (NZ), HIS (ND1, NE2)
  - Negative: ASP (OD1, OD2), GLU (OE1, OE2)

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
   pip install --upgrade torch torchvision torchaudio
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

## Support

Please reach out to <mhjonathan@gm.gist.ac.kr>

---

Tested on Apple M1. NVIDIA CUDA support requires validation.
