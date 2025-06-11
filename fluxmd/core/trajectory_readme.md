# Trajectory Generation in FluxMD

## Overview

FluxMD uses an adaptive trajectory generation system that creates "cocoon" trajectories around target biomolecules. The system automatically analyzes molecular geometry and generates appropriate trajectories for different molecular shapes.

## Core Concepts

### 1. Automatic Geometry Detection

FluxMD now automatically analyzes molecular geometry using Principal Component Analysis (PCA) to determine the optimal trajectory strategy:

```python
# Automatic shape classification
Shape type: linear
Dimensions: 245.3 x 23.5 x 21.2 Å
Aspect ratios: 10.4:1.1:1.0
→ Using cylindrical trajectory for linear molecule
```

**Shape Classifications:**
- **Linear**: DNA, fibrous proteins (aspect ratios > 3:3:1)
  - Uses cylindrical cocoon trajectories
  - Helical paths along major axis
  - Proper surface distance calculations
  
- **Globular**: Most proteins (aspect ratios < 2:2:1)
  - Uses spherical cocoon trajectories
  - Winding paths around center
  - Traditional approach
  
- **Planar**: Sheet-like structures (aspect ratio > 3:1 but < 3:3)
  - Uses disc-like trajectories
  
- **Irregular**: Complex assemblies
  - Uses adaptive sampling

### 2. Multi-Layer Cocoon Approach

Trajectories are generated in concentric shells around the target:
```
Layer 1 (5-10 Å): Close contacts, H-bonds
Layer 2 (10-15 Å): Salt bridges, strong interactions  
Layer 3 (15-25 Å): Electrostatic interactions
Layer 4 (25-50 Å): Initial approach, recognition
```

### 3. Surface Distance Calculations

A major improvement in trajectory generation is the use of **surface distances** instead of center distances:

```python
# Old method (problematic for linear molecules):
distance = ||position - molecule_center|| - molecule_radius  # Can be negative!

# New method (works for all shapes):
distance = min(||position - atom_i|| for all atoms)  # Always positive
```

This fixes the negative distance issue with DNA and provides accurate distance statistics for all molecular shapes.

## User-Configurable Parameters

### Step Size Configuration

The trajectory step size is now user-configurable across all FluxMD modes:

```python
# In interactive mode (fluxmd)
Enter trajectory step size (5-50 Å, default: 20): 15

# In command-line mode (fluxmd-uma)
fluxmd-uma protein.pdb ligand.pdb -o results/ --step-size 15

# In protein-DNA mode (fluxmd-protein-dna-uma)
fluxmd-protein-dna-uma protein.pdb dna.pdb -o results/ --step-size 10
```

### Key Parameters

1. **step_size** (5-50 Å): Distance between trajectory points
   - Smaller values (5-10 Å): Dense sampling, better for small binding sites
   - Medium values (15-25 Å): Balanced sampling (recommended)
   - Larger values (30-50 Å): Sparse sampling, faster computation

2. **n_steps** (50-500): Number of trajectory points per approach
   - Default: 200 for standard runs
   - Use 100 for quick screening
   - Use 500 for detailed analysis

3. **n_approaches** (5-20): Number of different approach angles
   - Default: 10 for balanced coverage
   - Use 5 for quick runs
   - Use 20 for comprehensive sampling

4. **starting_distance** (10-50 Å): Initial distance from target surface
   - Default: 20 Å
   - Smaller for compact binding sites
   - Larger for initial screening

5. **approach_distance** (1-5 Å): Distance step between approaches
   - Default: 2.5 Å
   - Controls shell spacing

## Trajectory Generation Methods

### 1. Linear Molecule Trajectories (DNA, Fibrous Proteins)

**Automatically activated when aspect ratios > 3:3:1**

The cylindrical cocoon trajectory system:
- **Helical motion**: 2.5 complete turns along the molecule
- **Axial coverage**: Sinusoidal motion covers full length
- **Radial oscillations**: Base distance ± 10 Å variations
- **Close approaches**: Every 10% of trajectory for H-bond sampling
- **Surface-based distances**: All distances calculated to nearest atom

```python
# Example output for DNA:
Molecular geometry analysis:
  Shape type: linear
  Dimensions: 245.3 x 23.5 x 21.2 Å
  Linear molecule: length=245.3 Å, radius=11.8 Å
  → Using cylindrical trajectory for linear molecule
  
Trajectory distance statistics:
  Min distance from surface: 2.1 Å
  Max distance from surface: 48.5 Å
  Mean distance from surface: 15.3 Å  # No more negative distances!
```

### 2. Globular Protein Trajectories

For roughly spherical molecules:
- Uses spherical coordinate system
- Winding motion with momentum
- Periodic close approaches for H-bond sampling
- Adaptive collision avoidance

```python
# Trajectory parameters for proteins
- Angular velocity: 4 complete winds
- Radial oscillation: Natural in/out motion
- Close approaches: Every n_steps/10 frames
- Minimum distance: 2.0 Å (allows H-bond detection)
```

### 3. Collision Detection

All trajectories include intelligent collision detection:
- KD-tree based spatial indexing
- Van der Waals radius checking
- Adaptive radius adjustment
- Smooth trajectory corrections

## Distance Calculations (Fixed!)

### Surface Distance Implementation

All trajectories now use proper surface distance calculations:

```python
def calculate_surface_distance(point, molecule_coords):
    """Calculate distance from point to nearest surface atom."""
    distances = cdist([point], molecule_coords)[0]
    return np.min(distances)  # Always positive!
```

**Benefits:**
- ✅ No more negative distances for linear molecules
- ✅ Accurate distance statistics for all shapes
- ✅ Better trajectory quality control
- ✅ Consistent behavior across molecular types

### Distance Statistics Output

```
Trajectory distance statistics:
  Min distance from surface: 2.1 Å      # Always positive
  Max distance from surface: 48.5 Å     # Meaningful range
  Mean distance from surface: 15.3 Å    # Accurate average
  Close approaches (<3.5Å): 42 frames   # H-bond sampling
  H-bond range (<3.5Å): 21.0% of frames # Quality metric
```

## Implementation Details

### Geometry Analysis Pipeline (Now Active!)

1. **PCA Analysis**: 
   - Compute covariance matrix of atomic coordinates
   - Extract eigenvalues and eigenvectors
   - Sort by magnitude to find principal axes

2. **Shape Classification**:
   - Calculate dimensions along each principal axis
   - Compute aspect ratios: dim[0]/dim[1], dim[0]/dim[2]
   - Classify based on ratios:
     - Linear: ratios > 3:3:1
     - Globular: ratios < 2:2:1
     - Planar: first ratio > 3, second < 2
     - Irregular: all other cases

3. **Trajectory Method Selection**:
   - Linear → Cylindrical cocoon (NEW!)
   - Globular → Spherical cocoon (enhanced)
   - Others → Adaptive methods (future)

4. **Surface Distance Calculation**:
   - Build KD-tree of molecule atoms (already done)
   - For each trajectory point, find nearest atom
   - Report true surface distance

### Adaptive Sampling Strategy

The system automatically adjusts sampling density based on:
- Local curvature (more samples in grooves/pockets)
- Electrostatic potential (focus on charged regions)
- Surface accessibility (avoid buried regions)
- Previous collision history (learn from failures)

## Best Practices

### For DNA-Protein Interactions
- Use step_size = 10-15 Å for groove sampling
- Increase n_approaches to 15-20 for full coverage
- Set starting_distance = 15 Å for close initial contact

### For Protein-Protein Interactions
- Use step_size = 15-20 Å for balanced sampling
- Standard n_approaches = 10 is usually sufficient
- Set starting_distance = 20-25 Å for gradual approach

### For Small Molecule Screening
- Use step_size = 5-10 Å for detailed sampling
- Reduce n_steps to 100 for faster computation
- Multiple iterations (20+) recommended

## Troubleshooting

### Issue: Too few collision-free positions
**Solution**: Increase starting_distance or reduce ligand size

### Issue: Poor surface coverage
**Solution**: Increase n_approaches or adjust step_size

### Issue: Negative distances from surface (DNA)
**Solution**: Fixed! Now using surface distance calculation for all molecules

### Issue: Poor DNA coverage
**Solution**: Fixed! Linear molecules now use cylindrical trajectories with full-length coverage

### Issue: Slow trajectory generation
**Solution**: Increase step_size or reduce n_steps

## Future Enhancements

1. **Machine Learning Integration**: Learn optimal trajectories from successful binding events
2. **Adaptive Mesh Refinement**: Dynamic trajectory density based on local interactions
3. **Multi-Scale Trajectories**: Coarse-to-fine sampling strategies
4. **GPU-Accelerated Generation**: Parallel trajectory computation
5. **Interactive Visualization**: Real-time trajectory adjustment

## References

- Brownian Dynamics: Ermak & McCammon (1978)
- Surface Analysis: Connolly (1983)
- PCA for Molecular Shape: Bakan et al. (2011)
- Adaptive Sampling: Shirts & Chodera (2008)