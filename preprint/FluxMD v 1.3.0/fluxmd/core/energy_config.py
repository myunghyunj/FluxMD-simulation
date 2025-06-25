# Energy bounds for capping to prevent singularities
# REF15 uses these to maintain numerical stability
ENERGY_BOUNDS = {
    "vdw": {"min": -20.0, "max": 10.0},
    "salt_bridge": {"min": -20.0, "max": 0.0},
    "default": {"min": -20.0, "max": 10.0},
    # REF15-specific bounds
    "fa_atr": {"min": -10.0, "max": 0.0},
    "fa_rep": {"min": 0.0, "max": 10.0},
    "fa_sol": {"min": -10.0, "max": 10.0},
    "fa_elec": {"min": -20.0, "max": 20.0},
    "hbond": {"min": -5.0, "max": 0.0},
}

# Energy function selection
ENERGY_FUNCTION_TYPES = {
    "simplified": "legacy",  # Original FluxMD energy function
    "ref15": "rosetta",  # Full REF15 implementation
    "ref15_fast": "rosetta_gpu",  # GPU-accelerated REF15
}

# Default energy function
DEFAULT_ENERGY_FUNCTION = "ref15"

# REF15 switching function parameters
REF15_SWITCHES = {
    "fa_atr": {"start": 5.5, "end": 6.0},
    "fa_sol": {"start": 5.2, "end": 5.5},
    "fa_elec": {"start": 5.5, "end": 6.0},
    "hbond": {"start": 1.9, "end": 3.3},
}
