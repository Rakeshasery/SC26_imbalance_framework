# SC26_imbalance_framework

Artifact repository for the SC26 2026 paper:
**"Hardware-Consistent GPU Imbalance Sensitivity:Cross-Node Validation for Energy-Efficient HPC"**

---

## Important Note for Reviewers (Double-Anonymous Review)

> **Artifact A1 (`DATA_Collection.py` + `RUN_Batch.py`) does NOT need to be
> executed during evaluation.** It requires specialised GPU hardware and
> approximately 45–55 hours of wall-clock time per full batch for each node. Hardware
> configuration details are withheld in accordance with SC26 double-anonymous
> review guidelines and will be disclosed upon paper acceptance.
>
> The complete pre-collected dataset `SC26_data.zip` produced by A1 is provided separately
> (see **Pre-collected Dataset** section below). Reviewers should download
> this dataset and execute **only A2 and A3**, which are CPU-only and
> complete in approximately **10–20 minutes total**.
>
> **Setup in one step:** Unzip `SC26_data.zip` and place all `.py` files
> (`Standalone_Mode9.py`, `Unified_Pipeline.py`, `DATA_Collection.py`,
> `RUN_Batch.py`) and `requirements.txt` directly inside the unzipped
> `SC26_data/` folder. Then install requirements and run A2 and A3 from
> that folder — no path configuration needed.

---

## Overview

This repository contains three artifacts supporting the paper's reproducibility:

| Artifact | Script(s) | Purpose | Run by reviewer? |
|----------|-----------|---------|-----------------|
| A1 | `DATA_Collection.py` + `RUN_Batch.py` | GPU telemetry & workload generation | No — pre-collected data provided |
| A2 | `Standalone_Mode9.py` | Multi-node cross-validation | **Yes (~2–5 min, CPU-only)** |
| A3 | `Unified_Pipeline.py` | Unified analysis & all figure/table generation | **Yes (~5–10 min, CPU-only)** |

Logical pipeline: A1 → A2 → A3.
**Reviewers run: A2 → A3 only**, using the pre-collected dataset in place of A1.

---

## Quick Start for Reviewers

**Total time: approximately 10–20 minutes. No GPU required.**

### Step 1 — Download and set up in one folder (~3 min)

**Dataset DOI:** [10.5281/zenodo.19658531](https://doi.org/10.5281/zenodo.19658531)

**Direct reviewer download link (restricted access — click to download):**

[https://zenodo.org/records/19658531?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc3NjY2OTgwMywiZXhwIjoxNzk1MTMyNzk5fQ.eyJpZCI6ImE2M2ZjYjJlLWU0ZGMtNGJkNC05NzNiLTkwMmIzMjJjZjNiNCIsImRhdGEiOnt9LCJyYW5kb20iOiI3MGY3ZGMwN2RlN2ZkM2U3MDM5MjFlMDcwZGJlNGVjYiJ9.znISe7p9g-OQSB20Mlm8ElMrQbUsE_nAXiukTRIQeKPSYPKwy_paarqkkfVnJtySpjbvDz_IInGi2ACbaIXvNg](https://zenodo.org/records/19658531?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc3NjY2OTgwMywiZXhwIjoxNzk1MTMyNzk5fQ.eyJpZCI6ImE2M2ZjYjJlLWU0ZGMtNGJkNC05NzNiLTkwMmIzMjJjZjNiNCIsImRhdGEiOnt9LCJyYW5kb20iOiI3MGY3ZGMwN2RlN2ZkM2U3MDM5MjFlMDcwZGJlNGVjYiJ9.znISe7p9g-OQSB20Mlm8ElMrQbUsE_nAXiukTRIQeKPSYPKwy_paarqkkfVnJtySpjbvDz_IInGi2ACbaIXvNg)

> Click the link above → Zenodo page opens → click **Download** next to `SC26_data.zip`.
> No Zenodo account needed.

Unzip it:

```bash
unzip SC26_data.zip
```

**Place all `.py` files and `requirements.txt` inside the unzipped
`SC26_data/` folder.** Your folder should look like this:

```
SC26_data/
├── Standalone_Mode9.py       ← A2 — place here
├── Unified_Pipeline.py       ← A3 — place here
├── DATA_Collection.py        ← A1 — place here (reference only)
├── RUN_Batch.py              ← A1 — place here (reference only)
├── requirements.txt          ← place here
├── r04gn01/
│   ├── S00.csv ... S29.csv
│   └── *.json
├── r04gn02/ ... r05gn06/    (12 node folders total)
```

Then change into that folder:

```bash
cd SC26_data
```

### Step 2 — Install dependencies (~2 min)

```bash
pip install -r requirements.txt
```

### Step 3 — Run A2 with a single click or command (~2–5 min)

```bash
python Standalone_Mode9.py
```

> **Single click / single run:** Open `Standalone_Mode9.py` in any Python
> environment — Spyder (Run → Run File), VS Code (▶ Run Python File),
> PyCharm (▶ Run), Jupyter (via `%run Standalone_Mode9.py`), or any
> terminal (`python Standalone_Mode9.py`). No arguments or configuration
> needed.

What this does:
- Auto-discovers all 12 node folders inside the current directory
- Reads `S*.csv` scenario files from each node
- Performs per-node OLS regression + bootstrap (n=2000)
- Runs pairwise t-tests and one-way ANOVA across all nodes
- Writes results to `Multinode/` inside the current folder

Expected output files in `SC26_data/Multinode/`:
```
multinode_validation_*.json        ← required input for A3 Step 4
multinode_report_*.txt
multinode_regression_table_*.tex
figure_regression_lines_*.png
figure_slope_comparison_*.png
figure_r2_comparison_*.png
```

### Step 4 — Run A3 with a single click or command (~5–10 min)

```bash
python Unified_Pipeline.py
```

> **Single click / single run:** Open `Unified_Pipeline.py` in any Python
> environment — Spyder (Run → Run File), VS Code (▶ Run Python File),
> PyCharm (▶ Run), Jupyter (via `%run Unified_Pipeline.py`), or any
> terminal (`python Unified_Pipeline.py`). No arguments or configuration
> needed — it auto-discovers all node folders and the `Multinode/` JSON
> produced by A2 automatically. **Run A2 first.**

What this does:
- Auto-discovers all node folders and their `S*.csv` files
- Runs all 7 analysis steps (metrics, validate, economic, multinode,
  all_metrics, all_validate, all_economic)
- Generates ALL paper figures, tables, and statistical summaries

Expected output locations inside `SC26_data/`:
```
Results_Figure/<node>/              ← per-node results (Steps 1–3)
Multinode/Results_Figure_All_Node/  ← all-nodes pooled results (Steps 5–7)
Multinode/                          ← cross-node results (Step 4)
```

---

## Validating Paper Claims

All figures, tables, and statistics in the paper are generated by A3.
After running Steps 3 and 4 above, verify these key results:

| Paper element | Output file | Expected value |
|--------------|-------------|----------------|
| Fig. 1 (CV scatter) | `Multinode/Results_Figure_All_Node/fig1_cv_scatter.png` | 336 points, 12 colours, pooled regression line |
| Fig. 2 (scenario overview) | `Multinode/Results_Figure_All_Node/fig2_scenario_overview.png` | 30 scenarios, mean ± 1 std |
| Fig. 3 (residual analysis) | `Multinode/Results_Figure_All_Node/fig_allnodes_val_03_residuals__ALL_NODES.png` | Homoscedastic, zero-centred residuals (mean=0, SD=0.01386 TFLOPS/W), confirming linear model adequacy |
| Fig. 4 (cost per TFLOP-hr) | `Multinode/Results_Figure_All_Node/fig_allnodes_econ_01_cost_per_tflop__ALL_NODES.png` | Cost range 627.7–900.8 µ$/TFLOP-hr across workloads; up to 30.3% reduction with decreasing imbalance; Burst-Avg best |
| Fig. 5 (ROI timeline) | `Multinode/Results_Figure_All_Node/fig_allnodes_econ_04_roi_timeline__ALL_NODES.png` | Payback 14.2–32.6 months across workload categories; up to ~$42k cumulative savings at 36 months |
| Fig. 6 (regression lines) | `Multinode/figure_regression_lines_*.png` | 12 per-node lines overlaid |
| Fig. 7 (per-node R²) | `Multinode/Results_Figure_All_Node/fig7_pernode_r2.png` | Per-node R² and Pearson \|r\| |
| Table IV | `Multinode/multinode_regression_table_*.tex` | Slope CV ≈ 3.43% across 12 nodes |
| C3 model | `Multinode/Results_Figure_All_Node/all_nodes_validation_results.csv` | R²=0.578, r=−0.760, p=1.55×10⁻⁶⁴, N=336 |
| C4 ROI | `Multinode/Results_Figure_All_Node/all_nodes_economic_summary.csv` | Shortest payback 14.2 months; best cost 627.7 µ$/TFLOP-hr |
| All F-tests / t-tests | `Multinode/multinode_statistics_*.json` | p > 0.59 for all 66 pairs |

Minor numerical variations (<0.01%) from the paper values are expected
due to platform floating-point differences.

---

## Repository Structure

```
SC26_imbalance_framework/
│
├── DATA_Collection.py      ← A1: interactive GPU characterisation (reference only)
├── RUN_Batch.py            ← A1: non-interactive batch runner    (reference only)
├── Standalone_Mode9.py     ← A2: multi-node cross-validation    ← RUN FIRST
├── Unified_Pipeline.py     ← A3: unified analysis pipeline      ← RUN SECOND
│
├── requirements.txt        ← CPU-only dependencies (A2 + A3)
├── requirements_gpu.txt    ← Full dependencies including GPU (A1)
├── LICENSE
└── README.md
│
└── SC26_data/              ← downloaded from Zenodo (not in repo)
    ├── r04gn01/
    │   ├── S00.csv ... S29.csv
    │   └── *.json
    ├── r04gn02/ ... r05gn06/   (12 node folders total)
    └── Multinode/              ← created by A2
        └── multinode_validation_*.json
```

---

## Pre-collected Dataset

The `SC26_data/` folder contains GPU telemetry collected from 12 nodes
(4 GPUs each, 48 GPUs total). It includes:

- 30 scenario CSV files per node (`S00.csv`–`S29.csv`)
- Approximately 14,800 rows per scenario per node
- Approximately 177,620 total samples across all 12 nodes
- CV% range: 0% to 173.21%
- JSON side-files: rebalancing summaries, statistical validation,
  economic projections

Hardware configuration details are withheld during double-anonymous
review and will be disclosed upon paper acceptance.

**Download:** [10.5281/zenodo.19658531](https://doi.org/10.5281/zenodo.19658531) — use the reviewer download link in the Quick Start section above.

---

## Requirements

### For A2 and A3 (CPU-only — what reviewers need)

```
Python 3.10+
numpy >= 1.24
pandas >= 2.0
scipy >= 1.10
matplotlib >= 3.7
scikit-learn >= 1.3
seaborn >= 0.12     (optional)
```

Install:
```bash
pip install -r requirements.txt
```

### For A1 (GPU cluster — reference only, not needed for evaluation)

```
Python 3.10+
CUDA 12.x + NVIDIA driver 535+
pynvml >= 11.5
numpy >= 1.24
pandas >= 2.0
PyTorch >= 2.1
scipy >= 1.10
matplotlib >= 3.7
scikit-learn >= 1.3
seaborn >= 0.12     (optional)
Linux with cpufreq governor access
```

Install:
```bash
# Install PyTorch with CUDA 12.x support first:
pip install torch>=2.1 --index-url https://download.pytorch.org/whl/cu121

# Then install remaining dependencies:
pip install -r requirements_gpu.txt
```

---

## A2 — Execution Options

```bash
# Auto-discover all nodes (recommended):
python Standalone_Mode9.py

# Specific nodes (use folder names from SC26_data/):
python Standalone_Mode9.py --nodes <node_name1> <node_name2> ...

# Custom base directory:
python Standalone_Mode9.py --base_dir /path/to/SC26_data

# Dry run (list CSVs found, no analysis):
python Standalone_Mode9.py --dry-run
```

---

## A3 — Execution Options

> **Note:** A3 requires A2 to have been run first. Specifically, Step 4
> (`multinode`) reads `Multinode/multinode_validation_*.json` produced by A2.
> Running the full pipeline without this file will skip Step 4 and omit
> Table IV and cross-node figures (Fig. 6). Always run A2 before A3.

```bash
# Full pipeline, auto-discover all nodes (recommended — run A2 first):
python Unified_Pipeline.py

# Specific steps only:
python Unified_Pipeline.py --steps metrics,validate

# Pooled analysis only (Steps 5–7, does not need A2 outputs):
python Unified_Pipeline.py --steps all_metrics,all_validate,all_economic

# Single node only:
python Unified_Pipeline.py --node_names <node_name>

# Group of nodes:
python Unified_Pipeline.py --node_names <node1> <node2> <node3>

# Custom data directory:
python Unified_Pipeline.py --base_data_dir /path/to/SC26_data

# Multinode step only (requires A2 outputs — Multinode/multinode_validation_*.json must exist):
python Unified_Pipeline.py --steps multinode
```

A3 auto-discovers all available node result folders. The following
directories are automatically skipped during auto-discovery:
`Multinode`, `Results_Figure`, `__pycache__`, `.git`.

---

## A1 — Reference Only (Do Not Run During Evaluation)

`DATA_Collection.py` and `RUN_Batch.py` are provided for transparency
and to fully document how the pre-collected dataset was generated.
They are fully functional scripts that can be used to collect new
telemetry on any compatible NVIDIA GPU cluster.

**Do not run during artifact evaluation** because:
- Requires NVIDIA data-centre GPUs (A100/H100 or equivalent)
- Takes approximately 45–55 hours per full batch on a single node
- Requires 12 nodes for full replication of the paper results
- Hardware configuration details withheld during double-anonymous review

### A1 key modes (for reference)

| Mode | Purpose | Time |
|------|---------|------|
| Mode 1 | 24h production monitoring | ~24h/node |
| Mode 4 | All 30 scenarios | ~9–10h/node |
| Mode 5 | Controlled rebalancing | ~40 min |
| Mode 6 | Adaptive sampling evaluation | ~25 min |
| Mode 7 | Enhanced statistical validation | ~5 min |
| Mode 8 | CV-aware paired rebalancing | ~40 min |
| Mode 10 | Economic projections | ~1 min |
| Mode 11 | SLURM template generation | ~1 min |
| Mode 12 | List all 30 scenarios | instant |
| Mode 13 | System info / GPU detection | instant |

---

## Troubleshooting

**"No node directories found"**
→ Make sure `SC26_data/` is extracted in the same folder as the scripts.
  Run: `ls SC26_data/` to confirm 12 node folders are present.

**"Module not found"**
→ Run: `pip install -r requirements.txt`

**A3 Step 4 fails or skips**
→ A2 must be run first. Check that
  `SC26_data/Multinode/multinode_validation_*.json` exists.

**Minor numerical differences from paper**
→ Expected. Variations less than 0.01% may occur due to platform
  floating-point differences. This does not affect the validity of results.

**Figures not generated**
→ Check that `matplotlib` is installed: `pip show matplotlib`

---

## Reuse Beyond This Paper

To apply this framework to a different GPU cluster:

1. Run `python DATA_Collection.py` (Mode 13) to detect your GPUs
2. Run Mode 4 to collect imbalance telemetry across your scenarios
3. Copy output `SC26_data/<your_node>/` folders to the analysis machine
4. Run `python Standalone_Mode9.py` for cross-node validation
5. Run `python Unified_Pipeline.py` for full analysis and figures

The framework supports any number of nodes and any NVIDIA data-centre GPU.

---

## License

MIT License. See `LICENSE` for details.

---

## Citation

```bibtex
@inproceedings{anonymous2026sc26,
  title     = {Hardware-Consistent GPU Imbalance Sensitivity:Cross-Node Validation for Energy-Efficient HPC},
  author    = {Anonymous},
  booktitle = {Proceedings of the International Conference for
               High Performance Computing, Networking, Storage
               and Analysis (SC26)},
  year      = {2026},
  doi       = {10.5281/zenodo.19658531}
}
```
