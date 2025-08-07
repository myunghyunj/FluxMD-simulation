import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from fluxmd.utils.config_parser import load_config


def test_matroshika_alias(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
mode: matroshika
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
    config = load_config(str(cfg))
    assert config["mode"] == "matryoshka"
