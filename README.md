# FluxMD

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

FluxMD identifies protein-ligand binding sites through energy flux differential analysis. The method treats binding sites as energy sinkholes where molecular interactions converge.

## Method

FluxMD tracks ligands approaching proteins from multiple angles. It calculates energy flux from non-covalent interactions and identifies regions of high flux as binding sites. Bootstrap analysis validates statistical significance.

## Features

- **Physics-based trajectories**: Brownian motion with molecular weight-dependent diffusion
- **Complete interaction detection**: Hydrogen bonds, salt bridges, π-π stacking, van der Waals
- **GPU acceleration**: 10-100x faster on Apple Silicon (NVIDIA CUDA implemented, untested)
- **Statistical validation**: Bootstrap confidence intervals and p-values

## Installation

```bash
git clone https://github.com/panelarin/FluxMD.git
cd FluxMD
conda create -n fluxmd python=3.8
conda activate fluxmd
pip install numpy pandas scipy matplotlib biopython joblib
conda install -c conda-forge openbabel
python setup_gpu.py
```

## Usage

Run the complete workflow:
```bash
python complete_workflow_fixed.py
```

Input files:
- Protein: PDB, CIF, or mmCIF
- Ligand: PDB, PDBQT, or SMILES

The program guides you through parameter selection.

## Output

```
results/
├── processed_flux_data_FIXED.csv          # Binding site rankings with statistics
├── {protein}_flux_report_FIXED.txt        # Detailed analysis
├── {protein}_trajectory_flux_analysis_FIXED.png  # Flux heatmap
└── iteration_*/                           # Trajectory data
```

Red regions in visualizations indicate high-flux binding sites.

## PyMOL Visualization

```python
run visualize_coloraf.py
colorflux myprotein, processed_flux_data_FIXED.csv
```

## Performance

| System | CPU | GPU (M1) | Speedup |
|--------|-----|----------|---------|
| 10K atoms | 60 min | 3 min | 20x |
| 50K atoms | 8 hours | 15 min | 32x |

## Theory

FluxMD implements three principles:
1. Ligands follow Brownian trajectories based on molecular physics
2. Non-covalent interactions create measurable energy flux
3. Binding sites appear as flux convergence points

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

FluxMD builds upon the protein-ligand analysis framework from https://github.com/jaehee831/protein-folding-analysis-bioinformatics

## License

MIT License. See LICENSE file.

## Support

Please reach out to <mailto:mhjonathan@gm.gist.ac.kr>

---

Tested on Apple M1. NVIDIA CUDA support requires validation.
