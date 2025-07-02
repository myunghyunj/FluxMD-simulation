# FluxMD Protein-DNA Analysis

## Overview
Analyzes protein-DNA interactions by inverting the standard workflow: DNA remains fixed while protein trajectories are simulated around it. Outputs flux values per nucleotide in 5'→3' order.

## Usage

### Interactive Mode
```bash
fluxmd
# Select Option 5: Protein-DNA Interaction (UMA)
```

### Command Line
```bash
fluxmd-protein-dna-uma dna.pdb protein.pdb -o results/
```

### Parameters
- `-s, --steps`: Steps per trajectory (default: 200)
- `-i, --iterations`: Number of iterations (default: 10)
- `-a, --approaches`: Approach angles (default: 10)
- `-d, --distance`: Starting distance in Å (default: 20.0)
- `-r, --rotations`: Rotations per position (default: 36)
- `--ph`: pH for protonation (default: 7.4)

## Key Features

### DNA-Specific
- Fixed DNA at origin, mobile protein
- Per-nucleotide flux calculation
- Sequence-ordered output (5'→3')
- Base-specific visualization (A:red, T:teal, G:blue, C:green)
- Phosphate backbone recognition

### Shared with Standard Pipeline
- UMA GPU acceleration
- Cocoon trajectory generation
- Bootstrap statistics (1000 iterations)
- Collision detection with VDW radii

## Output Files
- `processed_flux_data.csv`: Per-nucleotide flux values with statistics
- `*_flux_report.txt`: Detailed analysis report
- `*_flux_summary.png`: Flux distribution visualization
- `iteration_*/`: Trajectory data and visualizations

## Input Requirements
- Standard DNA PDB format (DA, DT, DG, DC residues)
- Proper chain identifiers
- Both strands for double-helix analysis

## Interpreting Results
- **High flux (>100)**: Strong interaction, likely binding site
- **Medium flux (50-100)**: Moderate interaction
- **Low flux (<50)**: Minimal interaction
- P-values indicate statistical significance

## Limitations
- Static DNA (no flexibility modeling)
- Single protein copy only
- B-form DNA expected
- No base pair dynamics