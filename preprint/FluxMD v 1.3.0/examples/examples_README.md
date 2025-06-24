# FluxMD Exemplars

A précis of FluxMD’s capabilities for biomolecular cartography—protein–ligand and protein–DNA alike.

---

## 1 · Interactive Tutorial

```bash
python examples/basic_usage.py
```

Master FluxMD via an executable walkthrough that elucidates:

- **Proteome–ligand analytics** | Ubiquitin × Benzene  
- **Protein–DNA interrogation** | p53 tumor suppressor  
- **SMILES → 3‑D ligands**  
- **Sequence → B‑DNA duplexes**  
- **Parameter heuristics**  
- **Result hermeneutics**

---

## 2 · Scientific Vignette — Allosteric Inhibition of **GPX4**

Liu *et al.* 2022 revealed that RSL3 eschews the canonical U46 active site, anchoring instead at an allosteric pocket near **Cys‑66**—a tipping point that triggers ferroptosis‑shielding activity.

### FluxMD Deployment

```bash
fluxmd-uma GPX4.pdb RSL3.pdb -o gpx4_rsl3_results/ --save-trajectories
```

**Anticipated insights**

- **Flux apogee at C66** (Y63 · E65 · G67) → allosteric epicenter  
- **Hydrophobic cradle** (Y63 · L166 · P167 · F170) spotlighted  
- **Conformational kinetics** of the L166–F170 loop captured via τ‑resolved flux

---

## 3 · Command‑Line Recipes

| Goal | Prototype Command | Rationale |
|------|------------------|-----------|
| *Rapid scan* ≈ 5 min | `fluxmd-uma protein.pdb ligand.pdb --steps 50 --iterations 5` | Triage insight |
| *Standard study* ≈ 30 min | `fluxmd-uma … --steps 100 --iterations 10 --rotations 36` | Balanced fidelity |
| *Publication‑grade* 2 h + | `fluxmd-uma … --steps 500 --iterations 20 --rotations 72 --save-trajectories` | Exhaustive sampling |
| DNA duplex synthesis | `fluxmd-dna GCGATCGCG -o dna_duplex.pdb` | B‑form generation |
| Protein–DNA assay | `fluxmd-protein-dna-uma protein.pdb dna.pdb -o dna_binding_results/` | Protein mobile, DNA fixed |
| SMILES → PDB | `echo "c1ccccc1" | fluxmd-smiles -o benzene.pdb` | 3‑D ligand creation |
| pH‑shifted binding | `fluxmd-uma protein.pdb ligand.pdb --ph 5.5` | Endosomal milieu |

---

## 4 · Interpreting Output

FluxMD results a `results/` directory containing:

- **`processed_flux_data.csv`** — residue‑ranked flux with 95 % CIs & *p*-values  
- **`{protein}_flux_report.txt`** — human‑readable précis  
- **`{protein}_trajectory_flux_analysis.png`** — sequence‑mapped heatmap

**Flux magnitude rubric**

- **> 1.0** → principal binding sink  
- **0.5 – 1.0** → ancillary surface  
- **< 0.5** → stochastic background

---

## 5 · Troubleshooting Codex

| Symptom | Etiology | Remediation |
|---------|----------|-------------|
| Sparse flux | Under‑sampling | Elevate `--steps ≥ 500`, `--iterations ≥ 20` |
| CUDA/MPS OOM | Memory saturation | Use disk‑buffered pipeline or temper `--rotations` |
| Distorted protein–DNA trajectory | Pre‑v2 geometry bug | Upgrade ≥ v2.0 & invoke `fluxmd-protein-dna-uma` |

---
