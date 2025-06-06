# Contributing to FluxMD

Thank you for your interest in contributing to FluxMD! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Code Style](#code-style)
9. [Performance Considerations](#performance-considerations)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/FluxMD.git
   cd FluxMD
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/FluxMD/FluxMD.git
   ```

## Development Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for GPU acceleration)
- OpenBabel (for molecular file conversions)

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install in development mode**:
   ```bash
   pip install -e .[dev]
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Install system dependencies**:
   ```bash
   # macOS
   brew install open-babel
   
   # Ubuntu/Debian
   sudo apt-get install openbabel
   
   # Conda
   conda install -c conda-forge openbabel
   ```

## Making Changes

### Branch Naming
- Feature branches: `feature/description`
- Bug fixes: `fix/description`
- Documentation: `docs/description`
- Performance: `perf/description`

### Workflow

1. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests locally**:
   ```bash
   pytest tests/
   ```

4. **Check code style**:
   ```bash
   black fluxmd/
   flake8 fluxmd/
   ```

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fluxmd --cov-report=html

# Run specific test file
pytest tests/unit/test_flux_analyzer.py

# Run only fast tests
pytest -m "not slow"

# Run GPU tests (requires GPU)
pytest -m gpu
```

### Writing Tests
- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use pytest fixtures for common test data
- Mock external dependencies
- Test edge cases and error conditions

Example test:
```python
import pytest
from fluxmd.core.flux_analyzer import calculate_flux

def test_calculate_flux():
    """Test flux calculation with known values"""
    energy = np.array([1.0, 2.0, 3.0])
    consistency = 0.8
    fluctuation = 0.1
    
    flux = calculate_flux(energy, consistency, fluctuation)
    expected = np.mean(np.abs(energy)) * consistency * (1 + fluctuation)
    
    assert np.isclose(flux, expected)
```

## Documentation

### Docstrings
Use NumPy-style docstrings:
```python
def calculate_interaction_energy(coords1, coords2, interaction_type='all'):
    """
    Calculate interaction energy between two sets of coordinates.
    
    Parameters
    ----------
    coords1 : np.ndarray
        First set of atomic coordinates, shape (n_atoms1, 3)
    coords2 : np.ndarray
        Second set of atomic coordinates, shape (n_atoms2, 3)
    interaction_type : str, optional
        Type of interaction to calculate. Options: 'all', 'hbond', 
        'salt_bridge', 'pi_stacking'. Default is 'all'.
    
    Returns
    -------
    float
        Total interaction energy in kcal/mol
    
    Examples
    --------
    >>> coords1 = np.array([[0, 0, 0], [1, 0, 0]])
    >>> coords2 = np.array([[0, 0, 3], [1, 0, 3]])
    >>> energy = calculate_interaction_energy(coords1, coords2)
    """
```

### Documentation Updates
- Update relevant .md files in `docs/`
- Add examples for new features
- Update API documentation
- Include performance considerations

## Submitting Changes

### Before Submitting
1. **Update from upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run full test suite**:
   ```bash
   pytest
   black --check fluxmd/
   flake8 fluxmd/
   ```

3. **Update documentation**

4. **Write meaningful commit messages**:
   ```
   feat: Add GPU memory estimation utility
   
   - Implement estimate_memory_usage() function
   - Add tests for different system sizes
   - Update documentation with usage examples
   
   Closes #123
   ```

### Pull Request Process
1. Push your branch to your fork
2. Create a Pull Request against `main` branch
3. Fill out the PR template completely
4. Ensure CI tests pass
5. Wait for code review
6. Address review comments
7. Squash commits if requested

## Code Style

### Python Style
- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use f-strings for formatting

### Naming Conventions
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Imports
Order imports as:
1. Standard library
2. Third-party packages
3. Local imports

```python
import os
import sys
from typing import List, Optional

import numpy as np
import torch
from scipy.spatial import cKDTree

from fluxmd.core import FluxAnalyzer
from fluxmd.gpu import get_device
```

## Performance Considerations

### GPU Code
- Always provide CPU fallback
- Use batch processing for efficiency
- Profile memory usage
- Document GPU memory requirements

### Memory Management
- Use generators for large datasets
- Implement chunking for large files
- Clear GPU cache when appropriate
- Use appropriate data types (float32 vs float64)

### Optimization Guidelines
- Profile before optimizing
- Document performance characteristics
- Add benchmarks for critical paths
- Consider memory vs speed tradeoffs

## Questions?

If you have questions, please:
1. Check existing issues and discussions
2. Read the documentation
3. Ask in the discussions forum
4. Create an issue if you've found a bug

Thank you for contributing to FluxMD!