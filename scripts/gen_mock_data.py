"""Generate minimal fixture data for tests."""

from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "tests" / "data"
BASE.mkdir(parents=True, exist_ok=True)

PROTEIN = """ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  GLY A   1       1.5     0.000   0.000  1.00  0.00           C
ATOM      3  C   GLY A   1       2.0     1.5     0.000  1.00  0.00           C
ATOM      4  O   GLY A   1       1.2     2.3     0.000  1.00  0.00           O
TER
END
"""

LIGAND = """Benzene
  Codex    0

 12 12  0  0  0  0            0 V2000
    0.0000    1.4027    0.0000 C   0  0  0  0  0  0
   -1.2148    0.7014    0.0000 C   0  0  0  0  0  0
   -1.2148   -0.7014    0.0000 C   0  0  0  0  0  0
    0.0000   -1.4027    0.0000 C   0  0  0  0  0  0
    1.2148   -0.7014    0.0000 C   0  0  0  0  0  0
    1.2148    0.7014    0.0000 C   0  0  0  0  0  0
    0.0000    2.4900    0.0000 H   0  0  0  0  0  0
   -2.1567    1.2450    0.0000 H   0  0  0  0  0  0
   -2.1567   -1.2450    0.0000 H   0  0  0  0  0  0
    0.0000   -2.4900    0.0000 H   0  0  0  0  0  0
    2.1567   -1.2450    0.0000 H   0  0  0  0  0  0
    2.1567    1.2450    0.0000 H   0  0  0  0  0  0
 1  2  2  0
 2  3  1  0
 3  4  2  0
 4  5  1  0
 5  6  2  0
 6  1  1  0
 1  7  1  0
 2  8  1  0
 3  9  1  0
 4 10  1  0
 5 11  1  0
 6 12  1  0
M  END
$$$$
"""

PARAMS = """mode: matryoshka
protein_file: protein.pdb
ligand_file: ligand.sdf
n_layers: 1
n_trajectories_per_layer: 1
"""

(BASE / "protein.pdb").write_text(PROTEIN)
(BASE / "ligand.sdf").write_text(LIGAND)
(BASE / "params.yaml").write_text(PARAMS)
print(f"Mock data written to {BASE}")
