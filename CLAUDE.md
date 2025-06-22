# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FluxMD is a GPU-accelerated molecular dynamics simulation tool for mapping binding interfaces between biomolecules using dynamic energy flux analysis. It supports protein-protein, protein-ligand, and protein-DNA interactions.

## Key Commands

### Development Setup
```bash
# Create conda environment (required for OpenBabel)
conda create -n fluxmd python=3.8
conda activate fluxmd

# Install with development dependencies
pip install -e ".[dev,gpu,viz]"

# Install OpenBabel (must use conda)
conda install -c conda-forge openbabel

# For Apple Silicon MPS support
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Running FluxMD
```bash
# Interactive mode
fluxmd

# UMA-optimized direct mode (100x faster)
fluxmd-uma protein.pdb ligand.pdb -o results/

# Generate DNA structure
fluxmd-dna ATCGATCG -o dna_structure.pdb

# Run benchmarks
python benchmarks/benchmark_uma.py
```

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_determinism.py -v
pytest tests/test_end_to_end.py -v
pytest tests/test_performance_regression.py -v

# Run with coverage
pytest --cov=fluxmd --cov-report=html

# Run parallel tests
pytest -n auto
```

### Code Quality
```bash
# Format code
black fluxmd/ tests/
isort fluxmd/ tests/

# Lint code
ruff check fluxmd/
flake8 fluxmd/
pylint fluxmd/

# Type checking
mypy fluxmd/

# Security scan
bandit -r fluxmd/
safety check
```

## Architecture Overview

### Core Components

1. **Energy Calculation Framework** (`fluxmd/core/`)
   - Implements Rosetta REF15 energy function with 167 atom types
   - Three modes: simplified (legacy), ref15 (default), ref15_fast (GPU)
   - pH-aware protonation states using Henderson-Hasselbalch equation
   - Energy terms: Lennard-Jones, solvation, electrostatics, hydrogen bonds

2. **Trajectory Generation** (`fluxmd/core/trajectory_generator.py`)
   - Helical orbital trajectories around target molecules
   - Sampling strategies: uniform linear, cylindrical, intelligent cocoon
   - Recent addition: √v cylindrical sampling for DNA groove geometry (160° coverage)

3. **GPU Acceleration** (`fluxmd/gpu/`)
   - Two pipelines: Standard (file I/O) and UMA (unified memory architecture)
   - Automatic device detection (MPS for Apple Silicon, CUDA for NVIDIA)
   - Zero-copy GPU processing in UMA mode for 100x speedup

4. **Statistical Analysis** (`fluxmd/analysis/`)
   - Bootstrap analysis (1000 iterations) for confidence intervals
   - Signed flux metric: Φᵢ = ⟨E̅ᵢ⟩ · Cᵢ · (1 + τᵢ)
   - P-value calculation for binding site significance

### Entry Points

- `fluxmd.py` - Interactive menu-driven interface
- `fluxmd_uma.py` - Command-line interface for UMA pipeline
- `fluxmd/cli.py` - Package entry points that load the above scripts
- `fluxmd/utils/dna_to_pdb.py` - DNA structure generation

### Key Design Patterns

1. **Dual Pipeline Architecture**: Standard (compatibility) vs UMA (performance)
2. **Adaptive Algorithms**: Automatically adjusts parameters based on system size
3. **Memory Efficiency**: Batch processing and streaming for large systems
4. **Deterministic Results**: Fixed random seeds and controlled numerical operations

## Important Implementation Details

- The package uses dynamic imports in `cli.py` to load standalone scripts (`fluxmd.py`, `fluxmd_uma.py`) to avoid naming conflicts
- Energy calculations use vectorized NumPy/PyTorch operations for performance
- DNA groove modeling uses 160° arc geometry with √u cylindrical sampling
- Performance regression tests enforce ≤5% slowdown threshold
- Bootstrap confidence intervals use 1000 iterations by default

## Recent Updates (v1.4.0)

- Added DNA interaction support with groove-aware sampling
- Implemented vectorized √u cylindrical sampler for better coverage
- Added deterministic test suite for reproducibility
- Performance baseline guards to prevent regression

## Common Development Tasks

### Building Documentation
```bash
# Build documentation site
mkdocs build

# Serve documentation locally (http://127.0.0.1:8000/)
mkdocs serve
```

### Recovery from Interrupted Runs
```bash
# Process completed iterations from interrupted run
python scripts/process_completed_iterations.py /path/to/interrupted/results/

# Continue analysis from existing flux data
python scripts/continue_analysis.py /path/to/interrupted/results/
```

### Performance Testing
```bash
# Run performance regression tests
pytest tests/test_performance_regression.py -v

# Run CAP benchmark (optional in CI)
python benchmarks/benchmark_cap.py
```

## Critical Path Considerations

- **OpenBabel Dependency**: Must be installed via conda, not pip
- **GPU Memory**: Large systems may require batch processing (automatic in UMA mode)
- **File Permissions**: FluxMD creates temporary files during processing
- **Python Version**: Requires Python 3.8+ for compatibility with all dependencies