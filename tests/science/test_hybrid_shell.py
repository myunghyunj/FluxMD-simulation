import numpy as np
import importlib.util
import pathlib

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "fluxmd" / "core" / "solvent" / "hybrid_shell.py"
)

spec = importlib.util.spec_from_file_location("hybrid_shell", MODULE_PATH)
hybrid_shell = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid_shell)
build_hybrid_shell = hybrid_shell.build_hybrid_shell


def test_scaffold_runs():
    lig = np.zeros((10, 3))
    rec = np.zeros((100, 3))
    waters = build_hybrid_shell(lig, rec)
    assert waters.shape[1] == 3
