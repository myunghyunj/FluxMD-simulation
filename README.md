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
- **GPU acceleration**: 10-100x faster on Apple Silicon (NVIDIA CUDA implemented, untested)
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

## PyMOL Visualization

FluxMD provides two visualization approaches for PyMOL users:

### Method 1: PyMOL Native Visualization (Recommended for PyMOL)

This method uses PyMOL's built-in rendering and grid system with the professional Berlin color palette (blue-white-red diverging colormap):

#### Method A: Color Already Loaded Proteins (Recommended)

First load your proteins in PyMOL, then color them:

```python
# Step 1: Load your proteins in PyMOL (File > Open or use load command)
load GPX4-wt.pdb
load GPX4-single.pdb
load GPX4-double.pdb

# Step 2: Run the coloring script
run pymol_colorflux.py

# Step 3: Color proteins with flux data

# Option A: Manual specification (most reliable)
colorflux GPX4-wt /full/path/to/wt_flux.csv
colorflux GPX4-single /full/path/to/single_flux.csv
colorflux GPX4-double /full/path/to/double_flux.csv

# Option B: Batch mode for multiple proteins
colorflux_batch GPX4-wt, /path/to/wt.csv, GPX4-single, /path/to/single.csv

# Option C: Auto-detection (requires specific file placement)
colorflux
# This searches for CSV files in PyMOL's current directory:
# - proteinname_flux_analysis/processed_flux_data.csv
# - proteinname_processed_flux_data.csv  
# - flux_analysis/processed_flux_data.csv
```

**Important**: The `colorflux` command without arguments only searches in PyMOL's current working directory. To change directory in PyMOL:
```python
cd /path/to/your/flux/results
pwd  # Check current directory
```

#### Method B: Load and Color in One Step

If you prefer to load PDB and CSV together:

```python
# Use the simple fload command
run pymol_fluxload.py
fload /path/to/protein.pdb /path/to/flux.csv WT

# Or use multiflux for multiple files
run pymol_multiflux.py
fluxload "pdb_file", "csv_file", "label"
```

#### Multiple Protein Comparison (Grid View)

**Option 1: Color already loaded proteins (Most Reliable)**
```python
# Load your proteins first
load wt.pdb
load single.pdb
load double.pdb

# Run coloring script
run pymol_colorflux.py

# Manually specify CSV paths
colorflux wt /path/to/wt_flux.csv
colorflux single /path/to/single_flux.csv
colorflux double /path/to/double_flux.csv

# Or use batch mode
colorflux_batch wt, /path/to/wt.csv, single, /path/to/single.csv, double, /path/to/double.csv

# Grid view is automatically enabled for multiple proteins
set grid_mode, 1
```

**Option 2: Load and color multiple proteins**
```python
# Use fload for each protein
run pymol_fluxload.py
fload wt.pdb wt_flux.csv WT
fload single.pdb single_flux.csv Single
fload double.pdb double_flux.csv Double

# Then enable grid view
set grid_mode, 1
```

**Option 3: Interactive mode with multiflux**

1. **Navigate to FluxMD directory in PyMOL:**
```python
cd /Users/myunghyun/Documents/GitHub/FluxMD
```

2. **Run the script:**
```python
run pymol_multiflux.py
```

3. **Load your protein structures:**
```python
load /path/to/GPX4-wt.pdb
load /path/to/GPX4-single.pdb
load /path/to/GPX4-double.pdb
```

4. **Apply flux coloring (interactive mode):**
```python
multiflux
# Dialog boxes will appear for each protein
# Enter the path to each CSV file or click Cancel to skip
# The script will also auto-search common locations if dialog is cancelled
```

**Alternative: Specify all CSV files at once:**
```python
# Format: Use quotes around the entire mapping string
multiflux "GPX4-wt=/path/to/wt.csv,GPX4-single=/path/to/single.csv,GPX4-double=/path/to/double.csv"

# Or without spaces after commas
multiflux GPX4-wt=/path/to/wt.csv,GPX4-single=/path/to/single.csv,GPX4-double=/path/to/double.csv
```

**Where to find CSV files:**
- After running FluxMD analysis, look in the output directory
- The file is named `processed_flux_data.csv`
- Typical path: `your_analysis_output/flux_analysis/processed_flux_data.csv`

**Common mistakes to avoid:**
- ❌ `python pymol_multiflux.py` (wrong - use `run` instead)
- ❌ `import pymol_multiflux` (wrong - use `run` instead)
- ✅ `run pymol_multiflux.py` (correct!)

**Available PyMOL flux visualization commands:**

| Script | Command | Description |
|--------|---------|-------------|
| `pymol_colorflux.py` | `colorflux` | Auto-detect CSV files via dialog/search |
| | `colorflux obj csv` | Color specific object with CSV |
| | `colorflux_batch obj1, csv1, obj2, csv2` | Batch coloring |
| `pymol_fluxload.py` | `fload pdb csv [label]` | Load PDB and color |
| `pymol_multiflux.py` | `multiflux` | Interactive mode (may have input issues) |
| | `fluxload pdb, csv, label` | Load with commas |

**Features of PyMOL native method:**
- Uses PyMOL's cartoon representation
- Native PyMOL grid view (set grid_mode)
- Berlin color palette for professional figures
- White background for optimal contrast
- Automatically hides ions, waters, and non-protein elements
- Interactive dialog boxes for file selection
- Auto-searches common flux data locations
- Interactive 3D manipulation
- Best for exploring structures interactively

### Method 2: Matplotlib Visualization (Publication Figures)

This creates static publication-quality figures using matplotlib:

#### From Terminal (Recommended)
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

#### From PyMOL Console (If matplotlib is available)
```python
# Check if matplotlib is available in PyMOL
import matplotlib
print("Matplotlib available!")

# Change to your FluxMD directory first
cd /path/to/FluxMD

# Run the visualization
run visualize_multiflux.py
# Follow the interactive prompts
# Note: This creates a separate matplotlib window
```

**Important for PyMOL users:**
- Most PyMOL installations don't include matplotlib by default
- To check: `import matplotlib` in PyMOL console
- If missing, use Method 1 (pymol_multiflux.py) instead
- Or run visualize_multiflux.py from terminal outside PyMOL

**Features of matplotlib method:**
- Publication-ready PNG output
- Smooth spline-interpolated ribbons using BioPython
- Grid layout for multiple proteins
- Consistent viewing angles
- Berlin color palette (same as PyMOL version)
- Best for figures and presentations
- Creates static images (not interactive)

### File Requirements

Both methods require:
- **PDB files**: Your protein structures
- **CSV files**: The `processed_flux_data.csv` file from FluxMD analysis (typically in `flux_analysis/` directory)

### Which Method to Use?

| Use Case | Recommended Method | Script |
|----------|-------------------|---------|
| Interactive exploration in PyMOL | Method 1 | `pymol_multiflux.py` |
| Publication figures | Method 2 | `visualize_multiflux.py` |
| Quick visualization in PyMOL | Method 1 | `pymol_multiflux.py` |
| Batch processing many proteins | Method 2 | `visualize_multiflux.py` |
| No PyMOL available | Method 2 | `visualize_multiflux.py` |

### Key Differences Between Scripts

| Feature | pymol_multiflux.py | visualize_multiflux.py |
|---------|-------------------|----------------------|
| **Environment** | Runs INSIDE PyMOL | Standalone Python script |
| **Output** | Interactive 3D in PyMOL | Static PNG images |
| **Visualization** | PyMOL's cartoon | Custom ribbon with BioPython |
| **Interactivity** | Full 3D rotation/zoom | Fixed viewing angle |
| **Color palette** | Berlin (blue-white-red) | Berlin (blue-white-red) |
| **Best for** | Exploration & analysis | Publications & reports |
| **Commands** | `fluxload`, `multiflux` | Command-line or interactive |

### Important Notes for PyMOL Visualization

1. **File paths**: Always use full absolute paths when specifying CSV files
2. **Auto-detection**: The `colorflux` command only searches in PyMOL's current directory
3. **Manual specification**: Most reliable method - directly specify CSV paths
4. **Batch mode**: Use `colorflux_batch` for multiple proteins at once

**Troubleshooting CSV file loading:**
```python
# Check PyMOL's current directory
pwd

# Change to where your CSV files are located
cd /Users/myunghyun/Desktop/FluxMD_results

# Or use full paths
colorflux GPX4-wt /Users/myunghyun/Desktop/FluxMD_results/wt_flux.csv
```

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
