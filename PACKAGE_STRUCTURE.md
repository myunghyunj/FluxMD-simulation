# FluxMD Package Structure

## Overview
FluxMD is organized as a Python package with the following module hierarchy:

```
fluxmd/
├── __init__.py                 # Main package exports
├── __version__.py              # Version information
├── cli.py                      # Command-line interface entry points
│
├── core/                       # Core functionality
│   ├── __init__.py
│   ├── trajectory_generator.py # Original FluxMD trajectory generation
│   ├── matryoshka_generator.py # Matryoshka physics-based engine
│   ├── ref15_energy.py         # REF15 energy calculations
│   ├── intra_protein_interactions.py
│   ├── protonation_aware_interactions.py
│   │
│   ├── surface/                # Surface generation modules
│   │   ├── __init__.py
│   │   ├── ses_builder.py      # Solvent-excluded surface
│   │   ├── layer_stream.py     # Multi-layer generation
│   │   └── dna_groove_detector.py # DNA groove detection
│   │
│   ├── geometry/               # Geometric analysis
│   │   ├── __init__.py
│   │   └── pca_anchors.py      # PCA-based anchor detection
│   │
│   └── dynamics/               # Molecular dynamics
│       ├── __init__.py
│       └── brownian_roller.py  # Brownian-Langevin dynamics
│
├── analysis/                   # Analysis modules
│   ├── __init__.py
│   ├── flux_analyzer.py        # CPU flux analysis
│   └── flux_analyzer_uma.py    # GPU/UMA flux analysis
│
├── gpu/                        # GPU acceleration
│   ├── __init__.py
│   ├── gpu_accelerated_flux.py
│   └── gpu_accelerated_flux_uma.py
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── pdb_parser.py           # PDB file I/O
│   ├── dna_to_pdb.py           # DNA structure generation
│   ├── cpu.py                  # Worker management
│   └── config_parser.py        # Configuration parsing
│
└── visualization/              # Visualization tools
    ├── __init__.py
    └── pymol_visualization.py
```

## Key Imports

### From `fluxmd`:
```python
from fluxmd import (
    __version__,
    ProteinLigandFluxAnalyzer,      # Original trajectory generator
    MatryoshkaTrajectoryGenerator,   # Physics-based trajectory engine
    TrajectoryFluxAnalyzer,          # Flux analysis
    PDBParser,                       # PDB file parsing
    dna_to_pdb_structure,           # DNA structure generation
)
```

### From `fluxmd.core`:
```python
from fluxmd.core import (
    MatryoshkaTrajectoryGenerator,
    REF15EnergyCalculator,
    surface,  # Submodule with SESBuilder, DNAGrooveDetector
    geometry, # Submodule with PCAAnchors
    dynamics, # Submodule with BrownianSurfaceRoller
)
```

### From `fluxmd.utils`:
```python
from fluxmd.utils import (
    parse_workers,          # Parse worker count from user input
    load_config,           # Load YAML/JSON config files
    create_example_config, # Generate example configuration
)
```

## Entry Points

The package provides command-line entry points defined in `pyproject.toml`:

- `fluxmd`: Interactive menu-driven interface
- `fluxmd-uma`: UMA-optimized command-line interface
- `fluxmd-dna`: DNA structure generation tool

## Notes

1. **Import Safety**: GPU modules gracefully handle missing PyTorch dependencies
2. **Submodule Organization**: Core functionality is organized into logical submodules
3. **Public API**: Main functionality is re-exported at package level for convenience
4. **Version Management**: Version is centralized in `__version__.py`