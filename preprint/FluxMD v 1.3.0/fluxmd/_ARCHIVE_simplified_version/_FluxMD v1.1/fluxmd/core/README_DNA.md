# FluxMD Protein-DNA Interaction Analysis

## Overview

FluxMD now supports **Protein-DNA interaction analysis** through a specialized UMA-optimized workflow. This feature inverts the traditional protein-ligand analysis paradigm:

- **DNA** is the static target molecule (fixed at origin)
- **Protein** is the mobile molecule with simulated approach trajectories
- **Output** is flux per DNA nucleotide in 5'→3' order

This workflow is designed to identify DNA regions that experience high mechanical stress during protein binding, providing insights into protein-DNA recognition mechanisms and binding site specificity.

## Key Features

- **DNA-centric analysis**: DNA remains fixed while protein trajectories are simulated
- **Nucleotide-level resolution**: Flux values computed for each DNA base
- **Sequence-ordered output**: Results presented in 5'→3' template strand order
- **GPU-accelerated**: Leverages Apple Silicon UMA for optimal performance
- **Comprehensive interactions**: Tracks hydrogen bonds, electrostatic, and base stacking

## Implementation Architecture

### Shared Infrastructure (with Options 1 & 2)

The Protein-DNA workflow inherits core functionalities from the ligand simulation pipeline:

- **UMA Architecture**: Zero-copy tensor operations via unified memory (Option 2)
- **Cocoon Trajectories**: Helical winding paths with Brownian perturbations
- **Bootstrap Statistics**: 1000-iteration confidence intervals
- **GPU Acceleration**: Adaptive algorithm selection based on system size
- **Collision Detection**: KD-tree spatial indexing with VDW radius checks

### DNA-Specific Enhancements

Novel features exclusive to the DNA workflow:

- **Barcode Visualization**: Nucleotide bases rendered as colored bars (A: red, T: teal, G: blue, C: green) on secondary axis
- **Sequence-Aware Flux**: Results ordered by 5'→3' directionality
- **Phosphate Recognition**: Automatic detection of backbone atoms (OP1, OP2, O3', O5') as acceptors
- **Base-Specific Properties**: Distinct donor/acceptor patterns for purines vs pyrimidines
- **Parameter Persistence**: Load previous simulation configurations via `simulation_parameters.txt`
- **Stochastic Iterations**: Random angle offsets prevent trajectory degeneracy

## Installation

Follow the standard FluxMD installation instructions in the main README.md. The Protein-DNA workflow is included in the UMA-optimized version.

## Usage

### Interactive Mode

From the main FluxMD menu, select:
```
Option 5: Protein-DNA Interaction (UMA)
```

You will be prompted to:
1. Load existing parameters (optional)
2. Provide DNA PDB file path
3. Provide Protein PDB file path
4. Choose analysis mode or custom parameters
5. Enable/disable trajectory visualizations

### Command-Line Interface

```bash
fluxmd-protein-dna-uma dna.pdb protein.pdb -o results/
```

#### Parameters

- `dna_file`: Input DNA structure file (PDB format)
- `protein_file`: Input protein structure file (PDB format)
- `-o, --output`: Output directory (default: flux_results)
- `-s, --steps`: Steps per trajectory (default: 200)
- `-i, --iterations`: Number of iterations (default: 10)
- `-a, --approaches`: Number of approach angles (default: 10)
- `-d, --distance`: Starting distance in Å (default: 20.0)
- `-r, --rotations`: Rotations per position (default: 36)
- `--ph`: Physiological pH for protonation (default: 7.4)
- `--cpu`: Force CPU usage (disables GPU acceleration)

### Example

```bash
# Analyze transcription factor binding to DNA promoter region
fluxmd-protein-dna-uma promoter_dna.pdb tf_protein.pdb -o tf_binding_analysis/ -i 20 -a 15

# High-resolution analysis with more trajectories
fluxmd-protein-dna-uma dna_target.pdb dna_binding_protein.pdb \
    -o detailed_results/ \
    -s 300 \
    -i 30 \
    -a 20 \
    -r 72
```

## Technical Implementation

### Trajectory Generation

The DNA workflow employs the `generate_cocoon_trajectory` method, identical to ligand simulations:

```python
# Protein molecular weight approximation
protein_mw = len(protein_atoms) * 110.0  # Average amino acid MW

# Diffusion coefficient calculation
D = k_B * T / (6 * π * η * r_h)  # Stokes-Einstein

# Cocoon parameters
- Winding motion: 4 complete helical turns
- Approach frequency: 10 close encounters per trajectory
- Minimum distance: 1.5 Å (H-bond sampling)
- Collision tolerance: 0.7 × (r_VDW1 + r_VDW2)
```

### Role Inversion

Unlike ligand workflows, the DNA implementation inverts molecular roles:

```python
trajectory_generator = ProteinLigandFluxAnalyzer(
    protein_file=dna_file,      # DNA as static target
    ligand_file=protein_file,   # Protein as mobile entity
    target_is_dna=True
)
```

### Visualization Pipeline

Enhanced flux visualization incorporates:
- Twin x-axes: nucleotide positions (bottom), base identities (top)
- Color-coded base bars via `axvspan` with `ymin=0.98`
- Automatic label decimation for sequences >50 bp

## Output Files

The analysis generates several output files in the specified directory:

### Primary Results

- **`processed_flux_data.csv`**: Main results file containing:
  - Nucleotide position (5'→3' order)
  - Base type (A, T, G, C)
  - Chain ID
  - Mean flux value
  - Standard deviation
  - P-value from bootstrap analysis
  - Significance indicator

Example output:
```csv
Position,Base,Chain,Mean_Flux,Std_Dev,P_value,Significant
1,A,A,125.4,15.2,0.001,True
2,T,A,89.3,12.1,0.023,True
3,G,A,45.2,8.7,0.156,False
...
```

### Detailed Analysis

- **`single_flux_report.txt`**: Comprehensive analysis report
- **`single_flux_summary.png`**: Visualization of flux distribution
- **`iteration_*/`**: Per-iteration trajectory data and interaction logs

### Trajectory Visualizations

Generated via `visualize_trajectory_cocoon` method:
- **3D perspective**: Protein path with gradient coloring (plasma colormap)
- **Distance profile**: Minimum protein-DNA distance over time
- **Orthogonal projections**: XY (top view) and XZ (side view)
- **Smoothed backbone**: Spline-interpolated DNA phosphate trace
- **Output format**: `iteration_*/trajectory_iteration_*_approach_*.png`

## DNA File Requirements

The input DNA PDB file should:

1. Contain standard DNA residues (DA, DT, DG, DC)
2. Have proper chain identifiers
3. Include both strands for double-stranded analysis
4. Follow standard PDB formatting

Example DNA residue naming:
```
ATOM      1  P    DA A   1      ...
ATOM      2  OP1  DA A   1      ...
ATOM      3  OP2  DA A   1      ...
ATOM      4  O5'  DA A   1      ...
```

## Interpreting Results

### Flux Values

- **High flux (>100 kcal/mol·Å)**: Strong mechanical stress, likely binding site
- **Medium flux (50-100 kcal/mol·Å)**: Moderate interaction, possible secondary contacts
- **Low flux (<50 kcal/mol·Å)**: Minimal interaction

### Statistical Significance

- Results include bootstrap p-values (1000 iterations)
- Nucleotides marked "Significant=True" show consistent flux across simulations
- Focus on significant high-flux regions for binding site identification

### Sequence Context

Results are ordered by template strand (5'→3'), facilitating:
- Motif identification
- Comparison with known binding sequences
- Analysis of flanking region effects

## Advanced Usage

### Custom DNA Properties

The workflow recognizes DNA-specific interactions:

- **Watson-Crick base pairing**
- **Major/minor groove accessibility**
- **Base stacking energetics**
- **Phosphate backbone electrostatics**

### Integration with Sequence Analysis

Export results for downstream analysis:

```python
import pandas as pd

# Load FluxMD results
flux_data = pd.read_csv('results/processed_flux_data.csv')

# Extract high-flux sequence
high_flux_bases = flux_data[flux_data['Mean_Flux'] > 100]
sequence = ''.join(high_flux_bases['Base'].values)
print(f"High-flux motif: {sequence}")
```

## Computational Considerations

### Performance

- DNA-protein systems typically larger than protein-ligand
- GPU acceleration crucial for reasonable computation times
- Memory usage scales with (DNA_atoms × protein_atoms × trajectories)

### Recommended Settings

For different DNA sizes:

- **Small DNA (<50 bp)**: Default parameters
- **Medium DNA (50-200 bp)**: Reduce approaches (-a 5) or iterations (-i 5)
- **Large DNA (>200 bp)**: Use focused region or reduce trajectory density

## Limitations

1. **Static DNA assumption**: DNA flexibility not modeled (no intra-DNA forces)
2. **Single binding event**: Multiple protein copies not supported
3. **B-form DNA**: Non-standard DNA conformations may need preprocessing
4. **No base pair dynamics**: Watson-Crick pairs remain fixed

## Citation

If you use the Protein-DNA workflow in your research, please cite:

```
FluxMD: GPU-Accelerated Molecular Dynamics for Protein-DNA Interactions
[Citation details to be added upon publication]
```

## Troubleshooting

### Common Issues

1. **"DNA file not recognized"**
   - Ensure residue names follow standard DNA conventions
   - Check for non-standard bases or modifications

2. **High memory usage**
   - Reduce trajectory parameters
   - Use region of interest instead of full DNA

3. **No significant flux detected**
   - Increase trajectory density (-r flag)
   - Check protein-DNA complex orientation
   - Verify starting distance is appropriate

### Getting Help

- Report issues: https://github.com/scottlittle/FluxMD/issues
- Include: Input file headers, command used, error messages

## Recent Enhancements

Successfully implemented features:

- [x] Parameter persistence across simulations
- [x] Barcode-style base visualization
- [x] Cocoon trajectory adoption from ligand workflow
- [x] Stochastic iteration variability
- [x] Enhanced collision detection for DNA atoms

## Future Enhancements

Planned features for the Protein-DNA workflow:

- [ ] RNA support (A-form helix recognition)
- [ ] Multiple protein binding analysis
- [ ] DNA flexibility incorporation via elastic network models
- [ ] Sequence-specific parameter optimization
- [ ] Integration with ChIP-seq data
- [ ] Intra-DNA force calculations (base stacking, H-bonds)