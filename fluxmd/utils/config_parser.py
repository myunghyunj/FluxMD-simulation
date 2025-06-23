# fluxmd/utils/config_parser.py
"""Configuration file parser for FluxMD batch mode."""

import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

__all__ = ["load_config", "save_config", "validate_config", "print_derived_constants"]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        if config_path.suffix in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    # Validate configuration
    validate_config(config)

    # Expand paths
    config = _expand_paths(config)

    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration structure and values.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    # Required fields
    required_fields = {
        "mode": ["cocoon", "matryoshka", "uma"],
        "protein_file": str,
        "ligand_file": str,
    }

    for field, expected in required_fields.items():
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

        if isinstance(expected, list):
            if config[field] not in expected:
                raise ValueError(f"Invalid {field}: {config[field]}. Must be one of {expected}")
        elif isinstance(expected, type):
            if not isinstance(config[field], expected):
                raise ValueError(f"Invalid type for {field}: expected {expected.__name__}")

    # Mode-specific validation
    mode = config["mode"]

    if mode == "matryoshka":
        matryoshka_fields = {
            "n_layers": (int, type(None)),
            "n_trajectories_per_layer": int,
            "layer_step": float,
            "probe_radius": float,
            "k_surf": float,
            "k_guid": float,
        }

        for field, expected_type in matryoshka_fields.items():
            if field in config:
                if isinstance(expected_type, tuple):
                    if not any(
                        isinstance(config[field], t) for t in expected_type if t is not type(None)
                    ):
                        if config[field] is not None:
                            raise ValueError(f"Invalid type for {field}")
                else:
                    if not isinstance(config[field], expected_type):
                        raise ValueError(
                            f"Invalid type for {field}: expected {expected_type.__name__}"
                        )

    # Validate numeric ranges
    numeric_ranges = {
        "physiological_pH": (0, 14),
        "n_iterations": (1, 10000),
        "n_steps": (1, 1000000),
        "temperature": (0, 1000),
        "layer_step": (0.1, 10.0),
        "probe_radius": (0.1, 5.0),
        "k_surf": (0.0, 100.0),
        "k_guid": (0.0, 100.0),
    }

    for field, (min_val, max_val) in numeric_ranges.items():
        if field in config:
            value = config[field]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{field} out of range: {value} not in [{min_val}, {max_val}]")


def _expand_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand relative paths to absolute paths.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with expanded paths
    """
    path_fields = ["protein_file", "ligand_file", "output_dir", "checkpoint_dir"]

    for field in path_fields:
        if field in config and config[field]:
            config[field] = os.path.abspath(os.path.expanduser(config[field]))

    return config


def print_derived_constants(config: Dict[str, Any]) -> None:
    """Print derived physical constants for dry-run mode.

    Args:
        config: Configuration dictionary
    """
    print("\n" + "=" * 80)
    print("DERIVED PHYSICAL CONSTANTS")
    print("=" * 80)

    # Temperature-dependent constants
    T = config.get("temperature", 298.15)
    kT = 0.0019872041 * T  # Boltzmann constant in kcal/mol
    print(f"\nTemperature: {T} K")
    print(f"Thermal energy (kT): {kT:.4f} kcal/mol")

    # Diffusion constants (if Matryoshka mode)
    if config.get("mode") == "matryoshka":
        viscosity = config.get("viscosity", 0.00089)  # Pa·s

        # Example calculation for 5 Å radius ligand
        R_ligand = 5.0  # Angstroms
        eta_converted = viscosity * 1.439e-4  # Convert to kcal·ps/mol/Å²

        D_t = kT / (6 * 3.14159 * eta_converted * R_ligand)
        D_r = kT / (8 * 3.14159 * eta_converted * R_ligand**3)

        print(f"\nViscosity: {viscosity} Pa·s")
        print("Example for 5 Å ligand:")
        print(f"  Translational diffusion (D_t): {D_t:.2e} Å²/ps")
        print(f"  Rotational diffusion (D_r): {D_r:.2e} rad²/ps")

        # Adaptive timestep
        target_rms = 0.3  # Å
        dt_ps = (target_rms**2) / (6 * D_t)
        dt_fs = dt_ps * 1000

        print(f"  Adaptive timestep: {dt_fs:.1f} fs")
        print(f"  Target RMS displacement: {target_rms} Å")

        # Layer parameters
        layer_step = config.get("layer_step", 1.5)
        n_layers = config.get("n_layers", "auto")
        print(f"\nLayer step size: {layer_step} Å")
        print(f"Number of layers: {n_layers}")

        if isinstance(n_layers, int):
            max_offset = n_layers * layer_step
            print(f"Maximum surface offset: {max_offset:.1f} Å")

    # Force constants
    if "k_surf" in config:
        print(f"\nSurface adherence constant: {config['k_surf']} kcal/mol/Å²")
        # Typical force at 1 Å deviation
        f_typical = config["k_surf"] * 1.0
        print(f"  Typical force at 1 Å: {f_typical:.2f} kcal/mol/Å")

    if "k_guid" in config:
        print(f"\nGuidance force constant: {config['k_guid']} kcal/mol/Å²")
        # Force at 10 Å from target
        f_guid = config["k_guid"] * 10.0
        print(f"  Force at 10 Å from target: {f_guid:.2f} kcal/mol/Å")

    print("\n" + "=" * 80)


# Example configuration template
EXAMPLE_CONFIG = """# FluxMD Configuration File
# Example configuration for Matryoshka trajectory generation

# Mode selection: cocoon, matryoshka, or uma
mode: matryoshka

# Input files (required)
protein_file: examples/protein.pdb
ligand_file: examples/ligand.pdb

# Output settings
output_dir: matryoshka_output
save_trajectories: true

# General parameters
physiological_pH: 7.4
temperature: 298.15  # Kelvin

# Matryoshka-specific parameters
n_layers: 10  # or 'auto' for adaptive
n_trajectories_per_layer: 10
layer_step: 1.5  # Angstroms
probe_radius: 0.75  # Angstroms

# Physics parameters
viscosity: 0.00089  # Pa·s (water at 25°C)
k_surf: 2.0  # Surface force constant (kcal/mol/Å²)
k_guid: 0.5  # Guidance force constant (kcal/mol/Å²)

# DNA-specific (optional)
groove_preference: major  # major or minor

# Computational settings
n_workers: 8  # Parallel workers (or 'auto')
checkpoint_dir: ./checkpoints
max_steps: 1000000  # Maximum steps per trajectory
use_gpu: true

# Legacy cocoon mode parameters (if mode: cocoon)
n_iterations: 100
n_steps: 100
n_approaches: 5
n_rotations: 36
starting_distance: 15.0
approach_distance: 2.5
"""


def create_example_config(output_path: str = "example_config.yaml") -> None:
    """Create an example configuration file.

    Args:
        output_path: Path to save example config
    """
    with open(output_path, "w") as f:
        f.write(EXAMPLE_CONFIG)
    print(f"Created example configuration: {output_path}")
