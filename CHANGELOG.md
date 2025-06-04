# FluxMD Changelog

## Version Updates

### 2025-01-06
- **Fixed pi-stacking distance cutoff**: Changed from 7.0 Å to 4.5 Å in all modules
  - `gpu_accelerated_flux.py`: Updated cutoff in self.cutoffs dictionary
  - `trajectory_generator.py`: Updated distance check for pi-stacking
  - `intra_protein_interactions.py`: Updated cutoff for consistency
  - This brings the cutoff in line with literature values (typical pi-stacking: 3.4-4.5 Å)

### Known Issues
- Pi-stacking detection still uses simplified aromatic ring identification
- Ligand aromatic detection marks all C/N/O/S atoms as potentially aromatic
- No actual ring detection algorithm implemented (NetworkX not used despite documentation)
- Recommend implementing proper graph-based ring detection in future versions

### Visualization Updates
- Added `visualize_multiflux.py` with Berlin colormap for publication-quality figures