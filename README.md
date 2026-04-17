# SC26_imbalance_framework

Artifact repository for the SC26 paper:
**"Hardware-Consistent GPU Imbalance Sensitivity: Cross-Node Validation for Energy-Efficient HPC"**

## Overview

This repository contains three artifacts supporting the paper's
reproducibility:

| Artifact | Script | Purpose |
|----------|--------|---------|
| A1 | `DATA_Collection.py` + `RUN_Batch.py` | GPU telemetry & workload generation |
| A2 | `Standalone_Mode9.py` | Multi-node cross-validation |
| A3 | `Unified_Pipeline.py` | Unified analysis & figure generation |

**Run in order: A1 → A2 → A3**

---

## Requirements

### Hardware
- **Minimum (A1):** 1 node with ≥2 NVIDIA data-centre GPUs (A100/H100),
  PCIe or NVLink, persistence mode support
- **Full replication (A1):** 12 nodes × 4 GPUs across two rack groups
- **A2 & A3:** CPU-only (no GPU needed)

### Software
