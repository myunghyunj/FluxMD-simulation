#!/usr/bin/env python3
"""
Migration script to reorganize FluxMD into a proper Python package structure.
This script creates the new directory structure and provides guidance for file migration.
"""

import os
import shutil
from pathlib import Path
import argparse

# File mappings from old to new structure
FILE_MAPPINGS = {
    # Core modules
    'fluxmd.py': 'fluxmd/cli.py',
    'flux_analyzer.py': 'fluxmd/core/flux_analyzer.py',
    'intra_protein_interactions.py': 'fluxmd/core/intra_protein.py',
    'protonation_aware_interactions.py': 'fluxmd/core/protonation.py',
    
    # GPU modules (to be refactored)
    'gpu_accelerated_flux.py': 'fluxmd/gpu/accelerator.py',
    'gpu_accelerated_flux_uma.py': 'fluxmd/gpu/uma/accelerator.py',
    
    # Trajectory modules (to be refactored)
    'trajectory_generator.py': 'fluxmd/trajectory/generator.py',
    'trajectory_generator_uma.py': 'fluxmd/trajectory/uma/generator.py',
    
    # Visualization
    'visualize_multiflux.py': 'fluxmd/visualization/multiflux.py',
    
    # Utils
    'dna_to_pdb.py': 'fluxmd/utils/dna_converter.py',
    
    # Tests
    'test_gpu_optimization.py': 'tests/unit/test_gpu_acceleration.py',
    'test_uma_optimization.py': 'tests/unit/test_uma_optimization.py',
    'benchmark_uma.py': 'tests/benchmarks/benchmark_uma.py',
}

# Files to ignore during migration
IGNORE_FILES = {
    'flux_analyzer_backup.py',
    'gpu_accelerated_flux_backup.py',
    'trajectory_generator_backup.py',
    'my_dna.pdb',  # Test file
    '__pycache__',
}

# New directories to create
NEW_DIRECTORIES = [
    'fluxmd',
    'fluxmd/core',
    'fluxmd/gpu',
    'fluxmd/gpu/interactions',
    'fluxmd/gpu/memory',
    'fluxmd/gpu/uma',
    'fluxmd/trajectory',
    'fluxmd/trajectory/motion',
    'fluxmd/trajectory/surface',
    'fluxmd/trajectory/uma',
    'fluxmd/visualization',
    'fluxmd/utils',
    'tests',
    'tests/unit',
    'tests/integration',
    'tests/benchmarks',
    'examples',
    'examples/sample_data',
    'docs',
    'docs/api',
    'docs/tutorials',
    'scripts',
    '.github',
    '.github/workflows',
    '.github/ISSUE_TEMPLATE',
]

# __init__.py content template
INIT_TEMPLATE = '''"""
{module_description}
"""

__all__ = [{exports}]
'''

# Package metadata
PACKAGE_METADATA = {
    'name': 'fluxmd',
    'version': '0.1.0',
    'description': 'GPU-accelerated binding site prediction using flux differential analysis',
    'author': 'FluxMD Contributors',
    'license': 'MIT',
}


def create_directory_structure(base_path):
    """Create the new package directory structure"""
    print("Creating directory structure...")
    for directory in NEW_DIRECTORIES:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {directory}/")
    
    # Create __init__.py files
    init_files = [
        ('fluxmd', 'FluxMD: GPU-accelerated binding site prediction'),
        ('fluxmd/core', 'Core analysis modules'),
        ('fluxmd/gpu', 'GPU acceleration modules'),
        ('fluxmd/gpu/interactions', 'Molecular interaction calculations'),
        ('fluxmd/gpu/memory', 'Memory optimization algorithms'),
        ('fluxmd/gpu/uma', 'Unified Memory Architecture optimizations'),
        ('fluxmd/trajectory', 'Trajectory generation modules'),
        ('fluxmd/trajectory/motion', 'Motion algorithms'),
        ('fluxmd/trajectory/surface', 'Surface analysis'),
        ('fluxmd/visualization', 'Visualization utilities'),
        ('fluxmd/utils', 'Utility functions'),
        ('tests', 'FluxMD test suite'),
    ]
    
    for module, description in init_files:
        init_path = base_path / module / '__init__.py'
        init_path.write_text(INIT_TEMPLATE.format(
            module_description=description,
            exports=''  # To be filled during refactoring
        ))
        print(f"  Created: {module}/__init__.py")


def create_setup_files(base_path):
    """Create setup.py and pyproject.toml"""
    
    # Create pyproject.toml
    pyproject_content = '''[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fluxmd"
version = "0.1.0"
description = "GPU-accelerated binding site prediction using flux differential analysis"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "FluxMD Contributors", email = "fluxmd@example.com"}
]
keywords = ["molecular dynamics", "drug discovery", "gpu acceleration", "binding sites"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Chemistry",
]

dependencies = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "matplotlib>=3.4.0",
    "biopython>=1.79",
    "torch>=2.0.0",
    "joblib>=1.0.0",
    "networkx>=2.6",
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "pytest-cov",
    "black",
    "flake8",
    "mypy",
    "sphinx",
]

[project.urls]
Homepage = "https://github.com/yourusername/FluxMD"
Documentation = "https://fluxmd.readthedocs.io"
Repository = "https://github.com/yourusername/FluxMD"
Issues = "https://github.com/yourusername/FluxMD/issues"

[project.scripts]
fluxmd = "fluxmd.cli:main"

[tool.setuptools]
packages = ["fluxmd", "fluxmd.core", "fluxmd.gpu", "fluxmd.trajectory", "fluxmd.visualization", "fluxmd.utils"]

[tool.black]
line-length = 100
target-version = ['py38']

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
'''
    
    pyproject_path = base_path / 'pyproject.toml'
    pyproject_path.write_text(pyproject_content)
    print("Created: pyproject.toml")
    
    # Create setup.py for compatibility
    setup_content = '''#!/usr/bin/env python
"""Setup script for FluxMD - for compatibility with older tools"""

from setuptools import setup

if __name__ == "__main__":
    setup()
'''
    
    setup_path = base_path / 'setup.py'
    setup_path.write_text(setup_content)
    setup_path.chmod(0o755)
    print("Created: setup.py")


def create_github_actions(base_path):
    """Create GitHub Actions workflow files"""
    
    # Test workflow
    test_workflow = '''name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install system dependencies
      run: |
        if [ "$RUNNER_OS" == "Linux" ]; then
          sudo apt-get update
          sudo apt-get install -y openbabel
        elif [ "$RUNNER_OS" == "macOS" ]; then
          brew install open-babel
        fi
      shell: bash
    
    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Run tests
      run: |
        pytest --cov=fluxmd --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
'''
    
    test_path = base_path / '.github' / 'workflows' / 'tests.yml'
    test_path.write_text(test_workflow)
    print("Created: .github/workflows/tests.yml")


def create_gitignore(base_path):
    """Create comprehensive .gitignore file"""
    
    gitignore_content = '''# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# FluxMD specific
results/
*.pdb
*.cif
*.sdf
*.mol2
*.pdbqt
!examples/sample_data/*.pdb

# Backup files
*_backup.py
*.bak

# Large data files
*.h5
*.hdf5
*.npz
'''
    
    gitignore_path = base_path / '.gitignore'
    gitignore_path.write_text(gitignore_content)
    print("Created: .gitignore")


def print_migration_guide():
    """Print guidance for manual migration steps"""
    
    print("\n" + "="*60)
    print("MIGRATION GUIDE")
    print("="*60)
    
    print("\n1. AUTOMATIC STEPS COMPLETED:")
    print("   - Created new directory structure")
    print("   - Generated package configuration files")
    print("   - Created GitHub Actions workflows")
    print("   - Added comprehensive .gitignore")
    
    print("\n2. MANUAL STEPS REQUIRED:")
    
    print("\n   a) Refactor monolithic files:")
    print("      - gpu_accelerated_flux.py → Split into gpu/ subdirectories")
    print("      - trajectory_generator.py → Split into trajectory/ subdirectories")
    
    print("\n   b) Update imports in all files:")
    print("      OLD: from flux_analyzer import TrajectoryFluxAnalyzer")
    print("      NEW: from fluxmd.core.flux_analyzer import TrajectoryFluxAnalyzer")
    
    print("\n   c) Create test files:")
    print("      - tests/unit/test_flux_analyzer.py")
    print("      - tests/unit/test_trajectory.py")
    print("      - tests/integration/test_full_pipeline.py")
    
    print("\n   d) Move and update documentation:")
    print("      - README.md → Update with new structure")
    print("      - Create docs/installation.md")
    print("      - Create docs/quickstart.md")
    
    print("\n   e) Clean up:")
    print("      - Remove backup files (*_backup.py)")
    print("      - Remove test data (my_dna.pdb)")
    print("      - Remove __pycache__ directories")
    
    print("\n3. TESTING:")
    print("   pip install -e .")
    print("   pytest tests/")
    
    print("\n4. VERSION CONTROL:")
    print("   git add .")
    print("   git commit -m 'Reorganize into proper Python package structure'")
    
    print("\n" + "="*60)


def main():
    """Main migration function"""
    parser = argparse.ArgumentParser(description='Migrate FluxMD to package structure')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--target', type=Path, default=Path('fluxmd_package'),
                       help='Target directory for new structure (default: fluxmd_package)')
    args = parser.parse_args()
    
    if not args.dry_run:
        print(f"Creating new package structure in: {args.target}")
        args.target.mkdir(exist_ok=True)
        
        create_directory_structure(args.target)
        create_setup_files(args.target)
        create_github_actions(args.target)
        create_gitignore(args.target)
    
    print_migration_guide()
    
    if args.dry_run:
        print("\n[DRY RUN] No files were created. Run without --dry-run to create structure.")


if __name__ == '__main__':
    main()