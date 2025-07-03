## Unreleased

### Added
- `.pre-commit-config.yaml` with Black, isort, flake8, codespell, markdownlint.
- docs/ARCHITECTURAL_REVIEW.md integrated in MkDocs.
- Scaffold `build_hybrid_shell()` and placeholder test.
- GPU-enabled Codex CI integration using CUDA-enabled PyTorch wheels.

### Known Issues
- 5 tests currently marked as expected failures (xfail); see issue tracker for details.
