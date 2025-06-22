# Critical Physics Fixes for Matryoshka Trajectory Engine

## Issues Fixed

### 1. Geodesic Distance Calculation ✓
**Problem**: Used simple Euclidean distance between anchors instead of surface geodesic path.

**Solution**: Implemented Dijkstra's algorithm to compute shortest path on surface mesh:
```python
def _compute_geodesic_distance(self) -> float:
    # Build adjacency graph from mesh faces
    # Run Dijkstra to find shortest path
    # Fallback to Euclidean if no path exists
```

### 2. Time-Based Guidance Ramp-Up ✓
**Problem**: Guidance force ramped based on spatial progress (distance-based), not time.

**Solution**: Implemented proper time-based soft ramp-up over ~1 ps:
```python
# Track activation time
if path_length >= 0.75 * self.geodesic_distance:
    self.guidance_active = True
    self.guidance_activation_time = current_time

# Ramp k_effective over 1 ps
time_since_activation = current_time - self.guidance_activation_time
ramp_factor = min(1.0, time_since_activation / self.guidance_anneal_time)
k_effective = self.k_guid * ramp_factor
```

### 3. Efficient Surface Queries ✓
**Problem**: O(n) brute force closest point search on every timestep.

**Solution**: Pre-built KD-tree for O(log n) queries:
```python
self.surface_kdtree = cKDTree(self.surface.vertices)
distance, closest_idx = self.surface_kdtree.query(point)
```

## Physics Validation

The implementation now correctly models:

1. **Brownian Exploration First**: 75% of the geodesic path is pure diffusion
2. **Gentle Guidance**: Weak harmonic spring (k_guid ~ 0.5 kcal/mol/Å²)
3. **Smooth Activation**: No unphysical jolts from instant force application
4. **Deterministic Completion**: Guaranteed arrival at target anchor

## Performance Impact

- Geodesic computation: One-time O(V² log V) at initialization
- KD-tree queries: Improved from O(V) to O(log V) per timestep
- Memory: Added small overhead for KD-tree and guidance tracking

## Next Steps

With physics correctly implemented, ready for Phase 5:
- CLI integration
- Parameter tuning
- Performance benchmarking
- Production testing