# Matryoshka Engine Architectural Review and Refactoring Roadmap

This document evaluates the current FluxMD Matryoshka engine, highlighting key approximations and providing guidance for refactoring toward a more rigorous simulation framework.

## 1. Brownian Dynamics Model

### Isotropic Diffusion Approximation
The ligand is simplified to a pseudo-sphere when computing inertial properties. Only the average of the inertia tensor is used:
```python
# matryoshka_generator._calculate_ligand_sphere
# ... inertia tensor assembled ...
# For simplicity, use trace/3 as effective moment (isotropic approximation)
effective_inertia = np.trace(inertia_tensor) / 3.0
```
[matryoshka_generator.py L149-L150](https://github.com/myunghyunj/FluxMD-simulation/blob/main/fluxmd/core/matryoshka_generator.py#L149-L150)

This neglects anisotropic diffusion where translation and rotation are coupled. Rigid-body Brownian dynamics typically employs a full 6×6 diffusion tensor constructed from atomic coordinates [(Ermak & McCammon, 1978)](https://doi.org/10.1063/1.436761).

**Refactoring Plan**
- Introduce a `Hydrodynamics` service computing the full diffusion tensor from atomic structure.
- Modify `BrownianSurfaceRoller` to propagate motion using this tensor within the BAOAB integrator.
- Add a configuration option `dynamics.hydrodynamics_model` with `isotropic` (default) and `anisotropic` modes.

## 2. Trajectory Generation and Biased Guidance

The roller applies two harmonic potentials (`k_surf`, `k_guid`) to steer the ligand. The guidance force ramps up near the end of the path:
```python
# brownian_roller._guidance_force
k_effective = self.k_guid * ramp_factor
return k_effective * (self.end_anchor - position)
```
[brownian_roller.py L409-L417](https://github.com/myunghyunj/FluxMD-simulation/blob/main/fluxmd/core/dynamics/brownian_roller.py#L409-L417)

This hard‑coded bias may not correspond to a physical free energy surface.

**Refactoring Plan**
- Define a `ForceProvider` abstract class. The existing heuristic becomes `HeuristicForceProvider`.
- Implement adaptive approaches such as metadynamics [(Laio & Parrinello, 2002)](https://doi.org/10.1073/pnas.202427399) or adaptive biasing force (ABF) [(Darve & Pohorille, 2001)](https://doi.org/10.1063/1.1385153) in subclasses.
- Inject the selected provider into `BrownianSurfaceRoller` via configuration (`simulation.sampling_method`).

## 3. Energy Calculation Variance

Energy for layer hopping uses a random subset of protein atoms:
```python
n_sample = max(100, int(n_protein * sample_fraction))
sample_indices = np.random.choice(n_protein, n_sample, replace=False)
...
energies *= n_protein / n_sample
```
[matryoshka_generator.py L330-L353](https://github.com/myunghyunj/FluxMD-simulation/blob/main/fluxmd/core/matryoshka_generator.py#L330-L353)

This introduces large variance. Methods such as grid-based potentials (e.g., AutoDock) or MM/PBSA snapshots provide more stable estimates.

**Refactoring Plan**
- Remove `sample_fraction` and construct a spatial neighbor search (k‑d tree or cell list) with configurable `energy.cutoff_radius`.
- Pass neighbor indices to `_calculate_ref15_energy_batch` to compute energies without scaling.

## 4. Static Surface Model

The generator builds a single SES surface and keeps the receptor rigid:
```python
self.base_surface = self.ses_builder.build_ses0()
```
[matryoshka_generator.py L100-L104](https://github.com/myunghyunj/FluxMD-simulation/blob/main/fluxmd/core/matryoshka_generator.py#L100-L104)

Realistic docking often requires an ensemble of receptor conformations to capture induced fit [(Amaro et al., 2018)](https://doi.org/10.1093/bib/bbw004).

**Refactoring Plan**
- Allow `MatryoshkaTrajectoryGenerator` to accept multiple receptor structures and iterate over them.
- Update workflow scripts to aggregate results across the ensemble.

## 5. TMI Flux Metric

The analysis likely bins trajectory coordinates to compute entropies, which can introduce discretization artifacts. A k-nearest‑neighbor estimator [(Kraskov et al., 2004)](https://doi.org/10.1103/PhysRevE.69.066138) reduces sensitivity to bin sizes.

**Refactoring Plan**
- Encapsulate the probability estimation in `FluxAnalyzer` via a pluggable `ProbabilityEstimator`.
- Provide both `BinnedProbabilityEstimator` (current behavior) and `KnnMutualInformationEstimator` implementations.
- Add tests comparing both estimators on a small analytical system.

## 6. GPU Kernel Optimization

The GPU backend computes interaction energies (`ref15_energy_gpu.py` and `gpu_accelerated_flux_uma.py`). Profiling is required to ensure memory coalescence and minimal divergence.

**Refactoring Plan**
- Establish performance baselines using existing benchmarks.
- Profile kernels with NVIDIA Nsight to inspect global-memory load efficiency and shared-memory usage.
- Refactor data layouts to structure‑of‑arrays and introduce shared‑memory tiling where appropriate.

---

### Summary
Implementing the above changes will transition the Matryoshka engine from heuristic approximations to a physically rigorous and extensible framework. The roadmap aligns the codebase with practices widely adopted in computational biophysics, enabling advanced sampling techniques, ensemble flexibility, and optimized GPU performance.

## References
1. D.L. Ermak and J.A. McCammon. *J. Chem. Phys.* **69**, 1352 (1978).
2. A. Laio and M. Parrinello. *Proc. Natl. Acad. Sci. USA* **99**, 12562 (2002).
3. E. Darve and A. Pohorille. *J. Chem. Phys.* **115**, 9169 (2001).
4. R.E. Amaro, et al. *Briefings in Bioinformatics* **19**, 785 (2018).
5. A. Kraskov, H. Stögbauer, and P. Grassberger. *Phys. Rev. E* **69**, 066138 (2004).
6. U. Essmann, et al. *J. Chem. Phys.* **103**, 8577 (1995).
