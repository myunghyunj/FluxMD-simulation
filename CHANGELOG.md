# Changelog

All notable changes to FluxMD will be documented in this file.

## [Unreleased]

### Fixed
- Fixed crash when workers prompt left blank (auto CPU detection) - now properly detects optimal worker count
- Guarded REF15 solvation term against zero atomic volume division errors with fallback values
- Added robust worker count parsing with auto-detection of CPU cores
- Added parameter validation for REF15 energy calculator to detect problematic atom types early

### Added
- New CPU utilities module (`fluxmd.utils.cpu_utils`) for robust worker management
- Comprehensive unit tests for worker parsing and zero volume handling
- Better user feedback for worker configuration in CLI

## [Previous entries would be here...]

## [1.4.0] – 2025-06-21
### Added / Changed — Improved Cylinderical Coordinates for DNA run
* 160° groove geometry & sequence-aware radii
* Vectorised √u cylindrical sampler (CuPy-ready)
* Deterministic test suite & baseline perf guard (≤5% Δ)

## [2.0.1] – 2025-06-22
### Fixed — Introduction of Matryoshka Trajectories
- Exposed `quaternion_multiply` utility and removed worker crash when running with `n_workers > 1`.
- Centralised worker count parsing to a shared utility to prevent `None > 1` TypeError.

### Changed
- Major version bump to 2.0.0 to reflect new Matryoshka trajectory engine.
- Reordered main menu to prioritize Matryoshka; legacy workflows moved to a submenu.
- Archived old development phase summaries into `docs/archive/`.

### Added
- Trajectory figure generation for Matryoshka runs (`--save-figures`).
- Archive utility to zip run results (`--archive-run`).
- Unit tests for quaternion helper, worker parsing, and parallel execution.
