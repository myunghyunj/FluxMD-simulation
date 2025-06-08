#!/usr/bin/env python3
"""
dna_builder.py  ·  sequence-dependent B-DNA generator
----------------------------------------------------
• Uses atom templates from `atomseq.py` if present;
  otherwise reverts to the built-in minimal templates.
• Dinucleotide-specific twist + propeller-twist
  (Olson et al., 1998, Nucleic Acids Res. 26:3820-29).
• Constant rise 3.38 Å, C1′ radius 5.40 Å (canonical B-form).

2025-06-08
"""

import math
import argparse
import importlib

import numpy as np

# ----------------------------------------------------------------------
#  0 │  DINUCLEOTIDE GEOMETRY  (twist°, propeller°)
# ----------------------------------------------------------------------
DINUC_PARAMS = {
    "AA": (35.6, -18.66),  "AC": (34.4, -13.10),
    "AG": (27.9, -14.00),  "AT": (32.1, -15.01),
    "CA": (34.5,  -9.45),  "CC": (33.7,  -8.11),
    "CG": (29.8, -10.03),  "CT": (27.9, -14.00),
    "GA": (36.9, -13.48),  "GC": (40.0, -11.08),
    "GG": (33.7,  -8.11),  "GT": (34.4, -13.10),
    "TA": (36.0, -11.85),  "TC": (36.9, -13.48),
    "TG": (34.5,  -9.45),  "TT": (35.6, -18.66),
}

RISE_PER_BASE = 3.38      # Å
HELIX_RADIUS  = 5.40      # Å (C1′ to axis)
C1_N_BOND     = 1.48      # Å glycosidic C1′–N9/1

COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

# ----------------------------------------------------------------------
#  1 │  ATOM TEMPLATES  — import from `atomseq` if available
# ----------------------------------------------------------------------
def _load_templates():
    """Return SUGAR_ATOMS, BASE_ATOMS from atomseq or built-in minimal set."""
    try:
        atomseq = importlib.import_module("atomseq")
        SUGAR = atomseq.SUGAR_ATOMS
        BASE = atomseq.BASE_ATOMS
        print("[info] Using templates from atomseq.py")
    except (ModuleNotFoundError, AttributeError):
        print(
            "[warn] atomseq.py not found or incomplete – using built-in minimal templates (C1′ and N9/1 only)."
        )
        SUGAR = [("C1'", 'C', np.array([0.0, 0.0, 0.0]))]
        BASE = {
            'A': [("N9", 'N', np.array([0.0, 0.0, C1_N_BOND]))],
            'G': [("N9", 'N', np.array([0.0, 0.0, C1_N_BOND]))],
            'C': [("N1", 'N', np.array([0.0, 0.0, C1_N_BOND]))],
            'T': [("N1", 'N', np.array([0.0, 0.0, C1_N_BOND]))],
        }
    return SUGAR, BASE


SUGAR_ATOMS, BASE_ATOMS = _load_templates()

# ----------------------------------------------------------------------
#  2 │  GEOMETRIC HELPERS
# ----------------------------------------------------------------------
def rot_x(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    x, y, z = v
    return np.array([x, y * c - z * s, y * s + z * c])

def rot_y(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    x, y, z = v
    return np.array([x * c + z * s, y, -x * s + z * c])

def rot_z(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    x, y, z = v
    return np.array([x * c - y * s, x * s + y * c, z])

# ----------------------------------------------------------------------
#  3 │  MAIN BUILDER CLASS
# ----------------------------------------------------------------------
class DNABuilder:
    def __init__(self):
        self.atoms = []
        self.aid = 1  # atom serial counter

    # ---- core pair placement ----
    def _place_pair(self, bA, bB, i, zpos, cum_twist, propeller):
        """Place a base pair with twist and propeller into the helix."""
        sugarA = []
        C1_A = None
        for name, el, crd in SUGAR_ATOMS:
            xyz = crd.copy()
            sugarA.append(("A", name, el, i + 1, xyz))
            if name == "C1'":
                C1_A = xyz

        baseA = []
        for name, el, crd in BASE_ATOMS[bA]:
            xyz = rot_y(crd.copy(), math.pi)
            xyz = rot_x(xyz, +propeller / 2)
            xyz += C1_A
            baseA.append(("A", name, el, i + 1, xyz))

        sugarB = []
        C1_B = None
        for name, el, crd in SUGAR_ATOMS:
            xyz = crd.copy()
            sugarB.append(("B", name, el, i + 1, xyz))
            if name == "C1'":
                C1_B = xyz

        baseB = []
        for name, el, crd in BASE_ATOMS[bB]:
            xyz = rot_x(crd.copy(), -propeller / 2)
            xyz += C1_B
            baseB.append(("B", name, el, i + 1, xyz))

        for chain, name, el, resid, xyz in sugarA + baseA + sugarB + baseB:
            if chain == 'A':
                xyz[0] += HELIX_RADIUS
            else:
                xyz[0] -= HELIX_RADIUS
            xyz = rot_z(xyz, cum_twist)
            xyz[2] += zpos
            self._store_atom(chain, resid, name, el, xyz)

    def _store_atom(self, chain, resid, name, element, xyz):
        self.atoms.append(
            dict(
                serial=self.aid,
                name=name,
                res="DNA",
                chain=chain,
                resid=resid,
                x=xyz[0],
                y=xyz[1],
                z=xyz[2],
                element=element,
            )
        )
        self.aid += 1

    def build(self, seq_5to3):
        seq = seq_5to3.upper()
        if not set(seq) <= set("ATGC"):
            raise ValueError("Sequence may contain only A,T,G,C.")

        comp = ''.join(COMPLEMENT[b] for b in seq)
        z, cum_tw = 0.0, 0.0
        for i, (bA, bB) in enumerate(zip(seq, comp)):
            if i < len(seq) - 1:
                prop_deg = DINUC_PARAMS[seq[i:i + 2]][1]
            else:
                prop_deg = DINUC_PARAMS[seq[i - 1:i + 1]][1]
            prop_rad = math.radians(prop_deg)
            self._place_pair(bA, bB, i, z, cum_tw, prop_rad)
            if i < len(seq) - 1:
                tw_deg = DINUC_PARAMS[seq[i:i + 2]][0]
                cum_tw += math.radians(tw_deg)
                z += RISE_PER_BASE

    def write_pdb(self, fname: str, conect: bool = False):
        with open(fname, "w") as fh:
            for a in self.atoms:
                fh.write(
                    (
                        "ATOM  {serial:5d} {name:<4s} {res:>3s} {chain}"
                        "{resid:4d}    "
                        "{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           "
                        "{element:>2s}\n"
                    ).format(**a)
                )
            if conect:
                fh.write("END\n")
        print(f"[done] {fname} written  – {len(self.atoms)} atoms")


# ----------------------------------------------------------------------
#  4 │  CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build sequence-dependent B-DNA PDB (twist + propeller-twist)."
    )
    ap.add_argument("sequence", help="DNA sequence 5′→3′ (A,T,G,C)")
    ap.add_argument("-o", "--output", default="dna.pdb", help="output PDB file (default dna.pdb)")
    ap.add_argument("--conect", action="store_true", help="append dummy END/CONECT records")
    args = ap.parse_args()

    builder = DNABuilder()
    builder.build(args.sequence)
    builder.write_pdb(args.output, conect=args.conect)
