import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def run_cli(config_path):
    # Use the script entry point directly since the package lacks __main__
    script = pathlib.Path(__file__).resolve().parents[2] / "fluxmd.py"
    cmd = [
        sys.executable,
        str(script),
        "--config",
        str(config_path),
        "--dry-run",
        "--seed",
        "5",
        "--print-backend",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return completed.stdout


def test_cli_backend_banner(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
mode: matryoshka
protein_file: protein.pdb
ligand_file: ligand.sdf
n_layers: 1
n_trajectories_per_layer: 1
layer_step: 1.5
probe_radius: 0.75
k_surf: 2.0
k_guid: 0.5
"""
    )
    out = run_cli(cfg)
    assert "Backend:" in out
    assert "Dtype:" in out
    assert "Seed: 5" in out
