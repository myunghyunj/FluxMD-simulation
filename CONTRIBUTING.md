# Contributing

## Coding standards

Frozen directories: Do not modify files under preprint/.
They are archival and excluded from lint/format hooks.

### Slow tests
Tests marked `@pytest.mark.slow` are skipped in CI.
Run the full suite locally with:

```bash
pytest -q          # fast subset
pytest -q -m slow  # slow tests only
```

Developers: ensure 'ruff==0.8.6' is in your dev environment; the
pre-commit hooks now use the interpreter's global Ruff.
