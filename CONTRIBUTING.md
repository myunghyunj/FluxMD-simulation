# Contributing

To set up a development environment:

```bash
git clone <repo_url>
cd FluxMD-simulation
pip install -e ".[dev]"
pre-commit install
ruff format . && pre-commit run --all-files
pytest -q
```

Please run the formatter and linters before opening a pull request.
