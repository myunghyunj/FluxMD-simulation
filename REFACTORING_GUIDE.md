# FluxMD Refactoring Guide

## Overview
This guide provides detailed instructions for refactoring the monolithic files in FluxMD into a modular package structure.

## 1. Refactoring gpu_accelerated_flux.py (1793 lines)

### Current Structure Analysis
The file contains:
- GPU device detection and setup
- Interaction calculations (H-bonds, salt bridges, π-π, π-cation, VDW)
- Memory optimization algorithms (spatial hashing, octree)
- Batch processing logic
- Property pre-computation

### Proposed Module Breakdown

#### fluxmd/gpu/device_utils.py
```python
"""GPU device detection and management utilities"""

import torch
import platform
from typing import Union, Tuple

def get_device() -> torch.device:
    """Auto-detect best available device (CUDA, MPS, or CPU)"""
    # Move device detection logic here
    pass

def get_device_properties() -> dict:
    """Get properties of current GPU device"""
    pass

def estimate_memory_usage(n_atoms: int, n_frames: int) -> int:
    """Estimate GPU memory requirements"""
    pass
```

#### fluxmd/gpu/interactions/base.py
```python
"""Base classes for interaction calculations"""

from abc import ABC, abstractmethod
import torch

class InteractionCalculator(ABC):
    """Abstract base class for molecular interactions"""
    
    @abstractmethod
    def calculate(self, coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
        """Calculate interaction energies"""
        pass
```

#### fluxmd/gpu/interactions/hydrogen_bonds.py
```python
"""Hydrogen bond detection and energy calculation"""

import torch
from .base import InteractionCalculator

class HydrogenBondCalculator(InteractionCalculator):
    """GPU-accelerated hydrogen bond calculations"""
    
    def __init__(self, distance_cutoff: float = 3.5, angle_cutoff: float = 120.0):
        self.distance_cutoff = distance_cutoff
        self.angle_cutoff = angle_cutoff
    
    def calculate(self, donors: torch.Tensor, acceptors: torch.Tensor) -> torch.Tensor:
        # Move H-bond calculation logic here
        pass
```

#### fluxmd/gpu/memory/spatial_hashing.py
```python
"""Spatial hashing for large molecular systems"""

import torch
from typing import Tuple

class SpatialHashGrid:
    """GPU-accelerated spatial hashing for neighbor searches"""
    
    def __init__(self, cell_size: float = 5.0):
        self.cell_size = cell_size
    
    def build_grid(self, coords: torch.Tensor) -> None:
        # Move spatial hashing logic here
        pass
    
    def query_neighbors(self, query_coords: torch.Tensor, radius: float) -> torch.Tensor:
        # Move neighbor query logic here
        pass
```

#### fluxmd/gpu/accelerator.py
```python
"""Main GPU acceleration interface"""

from typing import Dict, List, Optional
import torch
import pandas as pd

from .device_utils import get_device
from .interactions import (
    HydrogenBondCalculator,
    SaltBridgeCalculator,
    PiStackingCalculator,
    PiCationCalculator,
    VanDerWaalsCalculator
)
from .memory import SpatialHashGrid, OctreeGrid

class GPUAcceleratedInteractionCalculator:
    """Main class for GPU-accelerated interaction calculations"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self._init_calculators()
    
    def _init_calculators(self):
        """Initialize interaction calculators"""
        self.calculators = {
            'hydrogen_bond': HydrogenBondCalculator(),
            'salt_bridge': SaltBridgeCalculator(),
            'pi_stacking': PiStackingCalculator(),
            'pi_cation': PiCationCalculator(),
            'van_der_waals': VanDerWaalsCalculator(),
        }
    
    # Keep main interface methods here
```

## 2. Refactoring trajectory_generator.py (1974 lines)

### Current Structure Analysis
The file contains:
- Brownian motion simulation
- Winding trajectory generation
- Collision detection
- Surface point generation
- File I/O operations

### Proposed Module Breakdown

#### fluxmd/trajectory/motion/brownian.py
```python
"""Brownian motion simulation for molecular trajectories"""

import numpy as np
from typing import Tuple

class BrownianMotion:
    """Generate Brownian motion trajectories"""
    
    def __init__(self, diffusion_coefficient: float, temperature: float = 298.15):
        self.D = diffusion_coefficient
        self.T = temperature
    
    def generate_trajectory(self, start_pos: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
        # Move Brownian motion logic here
        pass
```

#### fluxmd/trajectory/motion/winding.py
```python
"""Winding trajectory generation around proteins"""

import numpy as np
from typing import Tuple, Optional

class WindingTrajectory:
    """Generate winding trajectories that explore protein surfaces"""
    
    def __init__(self, n_windings: int = 10, momentum: float = 0.8):
        self.n_windings = n_windings
        self.momentum = momentum
    
    def generate(self, center: np.ndarray, radius: float, n_points: int) -> np.ndarray:
        # Move winding trajectory logic here
        pass
```

#### fluxmd/trajectory/surface/kdtree.py
```python
"""KD-tree based surface analysis"""

from scipy.spatial import cKDTree
import numpy as np

class SurfaceAnalyzer:
    """Analyze molecular surfaces using KD-trees"""
    
    def __init__(self, coords: np.ndarray, probe_radius: float = 1.4):
        self.tree = cKDTree(coords)
        self.probe_radius = probe_radius
    
    def find_surface_points(self, n_points: int = 1000) -> np.ndarray:
        # Move surface point generation here
        pass
```

#### fluxmd/trajectory/collision.py
```python
"""Collision detection for trajectory generation"""

import numpy as np
from typing import Tuple, List

class CollisionDetector:
    """Detect and handle molecular collisions"""
    
    def __init__(self, vdw_radii: dict):
        self.vdw_radii = vdw_radii
    
    def check_collision(self, pos: np.ndarray, molecule_coords: np.ndarray) -> bool:
        # Move collision detection logic here
        pass
```

#### fluxmd/trajectory/generator.py
```python
"""Main trajectory generation interface"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict

from .motion import BrownianMotion, WindingTrajectory
from .surface import SurfaceAnalyzer
from .collision import CollisionDetector

class ProteinLigandTrajectoryGenerator:
    """Generate molecular trajectories for flux analysis"""
    
    def __init__(self, protein_file: str, ligand_file: str):
        self.protein_file = protein_file
        self.ligand_file = ligand_file
        self._load_structures()
    
    # Keep main interface methods here
```

## 3. Import Updates

After refactoring, all imports need to be updated:

### Old imports:
```python
from gpu_accelerated_flux import GPUAcceleratedInteractionCalculator
from trajectory_generator import ProteinLigandFluxAnalyzer
from flux_analyzer import TrajectoryFluxAnalyzer
```

### New imports:
```python
from fluxmd.gpu import GPUAcceleratedInteractionCalculator
from fluxmd.trajectory import ProteinLigandTrajectoryGenerator
from fluxmd.core.flux_analyzer import TrajectoryFluxAnalyzer
```

## 4. Testing the Refactored Code

### Unit Tests
Create focused unit tests for each new module:

```python
# tests/unit/gpu/test_hydrogen_bonds.py
import pytest
import torch
from fluxmd.gpu.interactions import HydrogenBondCalculator

def test_hydrogen_bond_detection():
    calc = HydrogenBondCalculator()
    # Test with known H-bond geometry
    donors = torch.tensor([[0, 0, 0]])
    acceptors = torch.tensor([[0, 0, 2.8]])
    
    bonds = calc.calculate(donors, acceptors)
    assert len(bonds) == 1
    assert bonds[0].energy < -2.0  # H-bonds are attractive
```

### Integration Tests
Test that refactored modules work together:

```python
# tests/integration/test_gpu_pipeline.py
def test_gpu_acceleration_pipeline():
    from fluxmd.gpu import GPUAcceleratedInteractionCalculator
    from fluxmd.gpu.device_utils import get_device
    
    device = get_device()
    calc = GPUAcceleratedInteractionCalculator(device)
    # Test full calculation pipeline
```

## 5. Migration Checklist

- [ ] Create new directory structure
- [ ] Extract device utilities → gpu/device_utils.py
- [ ] Extract each interaction type → gpu/interactions/*.py
- [ ] Extract memory algorithms → gpu/memory/*.py
- [ ] Extract motion algorithms → trajectory/motion/*.py
- [ ] Extract surface analysis → trajectory/surface/*.py
- [ ] Update all imports in existing code
- [ ] Create __init__.py files with proper exports
- [ ] Write unit tests for each new module
- [ ] Run integration tests
- [ ] Update documentation
- [ ] Remove old monolithic files

## 6. Benefits of Refactoring

1. **Maintainability**: Easier to find and fix bugs in focused modules
2. **Testability**: Can test each component in isolation
3. **Reusability**: Other projects can import specific components
4. **Performance**: Can optimize individual modules without affecting others
5. **Collaboration**: Multiple developers can work on different modules
6. **Documentation**: Easier to document focused modules
7. **Type Safety**: Better type hints and IDE support