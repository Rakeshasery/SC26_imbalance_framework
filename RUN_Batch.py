"""
################################################################################
# SC26 BATCH RUNNER — Single-Node Full Study
################################################################################
#
# Purpose: Run multiple modes in a single terminal command, non-interactively.
#          No prompts. All parameters are set at the top of this file.
#          Designed for a single node (e.g. r05gn01).
#
# Mode Sequence (single-node, reviewer-complete):
#
#   Mode 1  → 24h production monitoring (baseline dataset)
#   Mode 4  → All 30 scenarios (~9-10h full characterization)
#   Mode 5  → [STEP 1] Controlled Rebalancing Experiment
#   Mode 6  → [STEP 6] Adaptive Sampling Evaluation
#   Mode 8  → [STEP 1+] CV-Aware Paired Rebalancing (S28 extreme)
#   Mode 7  → [STEP 3] Enhanced Statistical Validation  ← needs Mode 4 data
#   Mode 10 → [STEP 5] Cluster-Scale Economic Projections
#   Mode 11 → [STEP 4] Scheduler Integration Guide
#
# Modes NOT run here (require 2 nodes):
#   Mode 9  → [STEP 2] Multi-Node Cross-Validation  ← run on second node after
#
# Total estimated wall-clock time: ~45-55 hours
#   Mode 1:  24h
#   Mode 4:  ~9-10h
#   Mode 5:  ~40min  (2 phases × 600s + 2min cooling)
#   Mode 6:  ~25min  (2 runs × 600s + 60s cooling)
#   Mode 8:  ~40min  (2 phases × 1200s)
#   Mode 7:  ~5min   (statistical computation only)
#   Mode 10: ~1min   (computation only, no GPU needed)
#   Mode 11: ~1min   (computation only, no GPU needed)
#
# Usage:
#   $ python RUN_Batch.py            # Run full sequence
#   $ python RUN_Batch.py --dry-run  # Print plan, do not execute
#   $ python RUN_Batch.py --from 4   # Start from Mode 4 (skip Mode 1)
#   $ python RUN_Batch.py --only 7   # Run only Mode 7
#
# Output:
#   All CSVs and JSONs saved to:
#     SC26_data/<node_name>/
#
# Log:
#   Progress is written to: sc26_batch_<node>_<timestamp>.log
#   (same directory as this script, or specify LOG_FILE below)
#
################################################################################
"""
import os
import sys
import glob
import json
import time
import argparse
import logging
from datetime import datetime
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these values before running
# ═══════════════════════════════════════════════════════════════════════════════
# ── Mode 1 settings ───────────────────────────────────────────────────────────
MODE1_DURATION     = 86400   # 24 hours (set to 3600 for a 1h quick test)
MODE1_INTERVAL     = 10      # Base sampling interval in seconds
MODE1_ADAPTIVE     = True    # Enable adaptive exponential-decay sampling

# ── Mode 4 settings ───────────────────────────────────────────────────────────
MODE4_CONFIRM      = True    # Must be True (bypasses interactive confirm)

# ── Mode 5 settings (Controlled Rebalancing Experiment) ───────────────────────
MODE5_SCENARIO     = 'S19_all_gradient_ascending'  # Baseline scenario
MODE5_PHASE_DUR    = 600     # Duration of each phase in seconds (default 10 min)
MODE5_CV_THRESH    = 22.0    # CV% trigger threshold for rebalancing

# ── Mode 6 settings (Adaptive Sampling Evaluation) ────────────────────────────
MODE6_SCENARIO     = 'S19_all_gradient_ascending'  # Scenario to evaluate on
MODE6_DURATION     = 600     # Duration per run in seconds (10 min per run)

# ── Mode 8 settings (CV-Aware Paired Rebalancing, S28) ────────────────────────
MODE8_DURATION     = 1200    # Duration per phase in seconds (20 min each)

# ── Mode 7 settings (Enhanced Statistical Validation) ─────────────────────────
# No settings needed — auto-discovers all CSVs in the node output folder

# ── Mode 10 settings (Cluster-Scale Economics) ────────────────────────────────
MODE10_BASE_SAVINGS   = 42000  # Annual savings per GPU (USD)
MODE10_SCALING_FACTOR = 0.5   # Conservative scaling factor (50%)

# ── Mode 11 settings (Scheduler Integration Guide) ────────────────────────────
# No settings needed — fully automated

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = None   # None = auto-generate name. Or set e.g. "sc26_batch.log"

# ═══════════════════════════════════════════════════════════════════════════════
# DO NOT EDIT BELOW THIS LINE
# ═══════════════════════════════════════════════════════════════════════════════
def setup_logging(node_name: str) -> logging.Logger:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_FILE or f"sc26_batch_{node_name}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    log = logging.getLogger("sc26_batch")
    log.info(f"Batch runner started. Log: {log_path}")
    return log

def print_banner(title: str, width: int = 100):
    print("\n" + "=" * width)
    pad = (width - len(title) - 2) // 2
    print(" " * pad + title)
    print("=" * width)

def print_plan(modes: list, node_name: str):
    """Print the execution plan without running anything."""
    print_banner("SC26 BATCH RUNNER — EXECUTION PLAN")
    print(f"\n  Node:          {node_name}")
    print(f"  Output dir:    SC26_data/{node_name}/")
    print(f"  Modes to run:  {modes}")
    print()

    details = {
        1:  f"Mode 1  — Basic Monitoring         | {MODE1_DURATION}s ({MODE1_DURATION/3600:.1f}h) | adaptive={MODE1_ADAPTIVE}",
        4:  f"Mode 4  — Complete Study (30 scen.) | ~9-10h",
        5:  f"Mode 5  — Controlled Rebalancing    | scenario={MODE5_SCENARIO} | {MODE5_PHASE_DUR}s/phase | CV_thresh={MODE5_CV_THRESH}%",
        6:  f"Mode 6  — Adaptive Sampling Eval.   | scenario={MODE6_SCENARIO} | {MODE6_DURATION}s/run",
        8:  f"Mode 8  — Paired Rebalancing (S28)  | {MODE8_DURATION}s/phase",
        7:  f"Mode 7  — Enhanced Statistical Val. | auto-discovers CSVs from {node_name}/",
        10: f"Mode 10 — Cluster-Scale Economics   | base=${MODE10_BASE_SAVINGS:,}/GPU | factor={MODE10_SCALING_FACTOR}",
        11: f"Mode 11 — Scheduler Integration     | prints guide + saves SLURM template",
    }
    for m in modes:
        print(f"  {'[SKIP]' if m not in details else '      '}  {details.get(m, f'Mode {m}')}")

    print()
    total_h = (
        (MODE1_DURATION / 3600 if 1 in modes else 0) +
        (10 if 4 in modes else 0) +
        (MODE5_PHASE_DUR * 2 / 3600 + 0.033 if 5 in modes else 0) +
        (MODE6_DURATION * 2 / 3600 + 0.017 if 6 in modes else 0) +
        (MODE8_DURATION * 2 / 3600 if 8 in modes else 0) +
        (0.1 if 7 in modes else 0) +
        (0.02 if 10 in modes else 0) +
        (0.02 if 11 in modes else 0)
    )
    print(f"  Estimated total wall-clock time: ~{total_h:.1f}h")
    print()

def import_framework():
    """Import the SC26 framework from the same directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fw_path = os.path.join(script_dir, "SC26_Conference_1Mar2026.py")
    if not os.path.isfile(fw_path):
        print(f"✗ ERROR: Cannot find SC26_Conference_1Mar2026.py in {script_dir}")
        print("  Place RUN_Batch.py in the same directory as the framework.")
        sys.exit(1)
    sys.path.insert(0, script_dir)
    import importlib
    fw = importlib.import_module("SC26_Conference_1Mar2026")
    return fw

# ═══════════════════════════════════════════════════════════════════════════════
# PER-MODE RUNNER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def run_mode_1(fw, log):
    """Mode 1: 24h production monitoring. No prompts."""
    log.info("=" * 80)
    log.info("STARTING MODE 1 — Basic Monitoring")
    log.info(f"  Duration:  {MODE1_DURATION}s ({MODE1_DURATION/3600:.1f}h)")
    log.info(f"  Interval:  {MODE1_INTERVAL}s  |  Adaptive: {MODE1_ADAPTIVE}")
    log.info("=" * 80)

    t0 = time.time()
    output = fw.main_jupyter(
        duration=MODE1_DURATION,
        interval=MODE1_INTERVAL,
        output_prefix='mode1_production',
        enable_adaptive=MODE1_ADAPTIVE
    )
    elapsed = (time.time() - t0) / 3600
    log.info(f"MODE 1 DONE  ({elapsed:.2f}h)  →  {output}")
    return output

def run_mode_4(fw, log):
    """Mode 4: All 30 scenarios. Auto-confirms."""
    log.info("=" * 80)
    log.info("STARTING MODE 4 — Complete Study (all 30 scenarios)")
    log.info("  Auto-confirmed (no interactive prompt)")
    log.info("=" * 80)

    t0 = time.time()
    # Monkey-patch input() so it returns 'yes'/'START' for the confirmation prompts
    original_input = __builtins__['input'] if isinstance(__builtins__, dict) else __builtins__.input
    import builtins
    _call_count = [0]
    def _auto_confirm(prompt=''):
        _call_count[0] += 1
        # run_complete_publication_study asks for 'yes' then 'START'
        response = 'yes' if _call_count[0] == 1 else 'START'
        print(f"{prompt}{response}  [auto-confirmed by batch runner]")
        return response
    builtins.input = _auto_confirm

    try:
        results = fw.run_complete_publication_study()
    finally:
        builtins.input = original_input   # Always restore

    elapsed = (time.time() - t0) / 3600
    log.info(f"MODE 4 DONE  ({elapsed:.2f}h)  →  {len(results)} scenarios completed")
    return results

def run_mode_5(fw, log):
    """Mode 5: Controlled Rebalancing Experiment. No prompts."""
    log.info("=" * 80)
    log.info("STARTING MODE 5 — Controlled Rebalancing Experiment [STEP 1]")
    log.info(f"  Scenario:    {MODE5_SCENARIO}")
    log.info(f"  Phase dur:   {MODE5_PHASE_DUR}s")
    log.info(f"  CV trigger:  >{MODE5_CV_THRESH}%")
    log.info("=" * 80)

    t0 = time.time()
    results = fw.run_controlled_rebalancing_experiment(
        baseline_scenario=MODE5_SCENARIO,
        phase_duration=MODE5_PHASE_DUR,
        cv_trigger_threshold=MODE5_CV_THRESH
    )
    elapsed = (time.time() - t0) / 60
    impr = results.get('improvement', {})
    log.info(f"MODE 5 DONE  ({elapsed:.1f}min)")
    log.info(f"  Efficiency gain: {impr.get('efficiency_gain_pct', 0):+.1f}%")
    log.info(f"  Energy saving:   {impr.get('energy_saving_pct',   0):+.1f}%")
    log.info(f"  CV reduction:    {impr.get('cv_reduction_pct',    0):+.1f}%")
    return results

def run_mode_6(fw, log):
    """Mode 6: Adaptive Sampling Evaluation. No prompts."""
    log.info("=" * 80)
    log.info("STARTING MODE 6 — Adaptive Sampling Evaluation [STEP 6]")
    log.info(f"  Scenario:  {MODE6_SCENARIO}")
    log.info(f"  Duration:  {MODE6_DURATION}s per run")
    log.info("=" * 80)

    t0 = time.time()
    results = fw.run_adaptive_sampling_evaluation(
        scenario_key=MODE6_SCENARIO,
        duration=MODE6_DURATION
    )
    elapsed = (time.time() - t0) / 60
    cmp = results.get('comparison', {})
    log.info(f"MODE 6 DONE  ({elapsed:.1f}min)")
    log.info(f"  Volume reduction: {cmp.get('volume_reduction_pct', 0):.1f}%")
    log.info(f"  Slope error:      {cmp.get('slope_error_pct', 0):.2f}%")
    return results

def run_mode_7(fw, log, node_name: str):
    """
    Mode 7: Enhanced Statistical Validation [STEP 3].
    Auto-discovers all CSVs already generated in the node output folder.
    Requires Mode 4 (or at least Mode 3) to have run first.
    """
    log.info("=" * 80)
    log.info("STARTING MODE 7 — Enhanced Statistical Validation [STEP 3]")
    log.info("=" * 80)

    node_output_dir = os.path.join(fw.OUTPUT_DIR, node_name)
    csv_files = glob.glob(os.path.join(node_output_dir, "*.csv"))

    if not csv_files:
        log.warning(f"  No CSVs found in {node_output_dir}. Skipping Mode 7.")
        log.warning("  Run Mode 4 first to generate scenario data.")
        return None

    log.info(f"  Found {len(csv_files)} CSV files in {node_output_dir}")

    # Build CV vs Efficiency dataset from all CSVs
    import csv as _csv
    all_cv, all_eff = [], []
    for csv_path in csv_files:
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = _csv.DictReader(f)
                seen = {}
                for row in reader:
                    sid = row.get('Sample_ID', '')
                    if sid not in seen:
                        seen[sid] = row
            for row in seen.values():
                try:
                    cv_v  = float(row.get('Load_Imbalance_CV_%',   0))
                    eff_v = float(row.get('Proxy_TFLOPS_per_Watt', 0))
                    if cv_v > 0 and eff_v > 0:
                        all_cv.append(cv_v)
                        all_eff.append(eff_v)
                except (ValueError, TypeError):
                    continue
        except Exception as e:
            log.warning(f"  Could not parse {os.path.basename(csv_path)}: {e}")
            continue

    if len(all_cv) < 10:
        log.warning(f"  Only {len(all_cv)} valid data points — need ≥10. Skipping.")
        return None

    log.info(f"  Dataset: {len(all_cv)} samples  →  running validation...")

    t0 = time.time()
    X = np.array(all_cv)
    y = np.array(all_eff)
    results = fw.enhanced_statistical_validation(X, y)
    elapsed = (time.time() - t0)

    # Save results JSON
    os.makedirs(node_output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(
        node_output_dir,
        f"step3_statistical_validation_{node_name}_{ts}.json"
    )
    save_results = {k: v for k, v in results.items() if k != 'model_comparison'}
    with open(summary_file, 'w') as f:
        json.dump(save_results, f, indent=2, default=float)

    log.info(f"MODE 7 DONE  ({elapsed:.1f}s)  →  {summary_file}")
    return results

def run_mode_8(fw, log, node_name: str):
    """Mode 8: CV-Aware Paired Rebalancing Experiment (S28). No prompts."""
    log.info("=" * 80)
    log.info("STARTING MODE 8 — CV-Aware Paired Rebalancing [STEP 1+]")
    log.info(f"  S28 extreme imbalance (90/5/5/5) → rebalanced after 5 min")
    log.info(f"  Phase duration: {MODE8_DURATION}s each")
    log.info("=" * 80)

    def _monitor_func(duration, node):
        return fw.run_mode1_monitoring_enhanced(duration=duration, node_name=node)

    def _workload_func(config):
        pass  # Handled inside run_rebalancing_experiment

    t0 = time.time()
    results = fw.run_rebalancing_experiment(
        monitor_func=_monitor_func,
        workload_func=_workload_func,
        duration=MODE8_DURATION,
        node_name=node_name
    )
    elapsed = (time.time() - t0) / 60
    impr = results.get('improvement', {})
    log.info(f"MODE 8 DONE  ({elapsed:.1f}min)")
    log.info(f"  Energy reduction: {impr.get('energy_reduction_pct', 0):+.1f}%")
    log.info(f"  Efficiency gain:  {impr.get('efficiency_gain_pct',  0):+.1f}%")
    return results

def run_mode_10(fw, log, node_name: str):
    """Mode 10: Cluster-Scale Economic Projections. No prompts."""
    log.info("=" * 80)
    log.info("STARTING MODE 10 — Cluster-Scale Economic Projections [STEP 5]")
    log.info(f"  Base savings: ${MODE10_BASE_SAVINGS:,}/GPU/year")
    log.info(f"  Scaling:      {MODE10_SCALING_FACTOR*100:.0f}% (conservative)")
    log.info("=" * 80)

    t0 = time.time()
    results = fw.scale_economic_impact(
        base_annual_savings_per_gpu=MODE10_BASE_SAVINGS,
        cluster_configs=[
            {'nodes': 64,  'gpus_per_node': 4},
            {'nodes': 128, 'gpus_per_node': 4},
            {'nodes': 256, 'gpus_per_node': 4},
            {'nodes': 512, 'gpus_per_node': 4},
        ],
        scaling_factor=MODE10_SCALING_FACTOR
    )

    node_output_dir = os.path.join(fw.OUTPUT_DIR, node_name)
    os.makedirs(node_output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(
        node_output_dir,
        f"step5_economic_projections_{node_name}_{ts}.json"
    )
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)

    elapsed = time.time() - t0
    log.info(f"MODE 10 DONE  ({elapsed:.1f}s)  →  {summary_file}")
    return results

def run_mode_11(fw, log, node_name: str):
    """Mode 11: Scheduler Integration Guide. No prompts."""
    log.info("=" * 80)
    log.info("STARTING MODE 11 — Scheduler Integration Guide [STEP 4]")
    log.info("=" * 80)

    t0 = time.time()

    # Print guide
    fw.generate_scheduler_integration_guide()

    # Policy demo
    policy = fw.CVAwareSchedulingPolicy()
    test_cases = [
        ([85, 84, 83, 86], "Balanced high load"),
        ([82, 65, 48, 30], "Realistic mixed (S22)"),
        ([90, 10, 10, 10], "Extreme imbalance (S28)"),
        ([25, 50, 75, 95], "Gradient ascending (S19)"),
    ]
    print("\n  Example Policy Decisions:")
    for utils, label in test_cases:
        rec = policy.recommend_action(utils)
        cv  = fw.CVAwareSchedulingPolicy.calculate_cv([float(u) for u in utils])
        print(f"\n  {label}")
        print(f"    Utils: {utils}  →  CV={cv:.1f}%")
        print(f"    Action: {rec['action'].upper()} | Priority: {rec['priority']}")
        print(f"    Reason: {rec['reason']}")

    # Save SLURM template
    node_output_dir = os.path.join(fw.OUTPUT_DIR, node_name)
    os.makedirs(node_output_dir, exist_ok=True)
    slurm_file = os.path.join(node_output_dir, "cv_monitor_epilog.sh")
    with open(slurm_file, 'w') as f:
        f.write(policy.slurm_epilog_template())

    elapsed = time.time() - t0
    log.info(f"MODE 11 DONE  ({elapsed:.1f}s)  →  SLURM template: {slurm_file}")
    return slurm_file

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BATCH ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════
# Full recommended single-node sequence:
#   1  → 4  → 5  → 6  → 8  → 7  → 10  → 11
#   (Mode 7 intentionally comes after 4/5/6/8 so it has maximum data)
FULL_SEQUENCE = [1, 4, 5, 6, 8, 7, 10, 11]

def prompt_run_selection() -> list:
    """
    Interactively ask the user whether to run the full sequence or select
    specific modes manually.

    Returns the list of mode numbers to execute.
    """
    valid_modes = [1, 4, 5, 6, 7, 8, 10, 11]

    print()
    print("=" * 60)
    print("  SC26 BATCH RUNNER — SELECT RUN MODE")
    print("=" * 60)
    print()
    print("  1 — Run full sequence  (1 → 4 → 5 → 6 → 8 → 7 → 10 → 11)")
    print("  2 — Select specific modes manually")
    print()

    while True:
        choice = input("  Enter selection [1 or 2]: ").strip()
        if choice in ('1', '2'):
            break
        print("  Invalid input. Please enter 1 or 2.")

    if choice == '1':
        print()
        print(f"  ✓ Full sequence selected: {FULL_SEQUENCE}")
        print()
        return list(FULL_SEQUENCE)

    # ── Option 2: manual mode selection ───────────────────────────────────────
    print()
    print(f"  Available modes: {valid_modes}")
    print("  Note: Mode 9 (multi-node) is not available in single-node batch runner.")
    print()

    while True:
        raw = input("  Enter mode number(s) separated by spaces (e.g. 5 6 8): ").strip()
        if not raw:
            print("  No input received. Please enter at least one mode number.")
            continue

        parts = raw.split()
        try:
            modes = [int(p) for p in parts]
        except ValueError:
            print("  Invalid input — please enter integers only (e.g. 5 6 8).")
            continue

        invalid = [m for m in modes if m not in valid_modes]
        if invalid:
            print(f"  Unrecognised mode(s): {invalid}. Valid choices: {valid_modes}")
            continue

        break

    print()
    print(f"  ✓ Selected modes: {modes}")
    print()
    return modes

def main():
    parser = argparse.ArgumentParser(
        description="SC26 Batch Runner — run multiple modes non-interactively",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python RUN_Batch.py                   Interactive selection (Option 1 or 2)
  python RUN_Batch.py --dry-run         Print plan only, no execution
  python RUN_Batch.py --from 4          Start from Mode 4 (skip Mode 1)
  python RUN_Batch.py --from 5          Start from Mode 5 (skip 1 and 4)
  python RUN_Batch.py --only 7          Run only Mode 7 (stats validation)
  python RUN_Batch.py --only 10 11      Run only Modes 10 and 11
  python RUN_Batch.py --sequence 5 6 8  Custom sequence: 5 then 6 then 8
        """
    )
    parser.add_argument('--dry-run',   action='store_true',
                        help='Print execution plan without running anything')
    parser.add_argument('--from',      type=int, dest='start_from', default=None,
                        metavar='MODE',
                        help='Skip all modes before this mode number')
    parser.add_argument('--only',      type=int, nargs='+', default=None,
                        metavar='MODE',
                        help='Run only these specific mode(s)')
    parser.add_argument('--sequence',  type=int, nargs='+', default=None,
                        metavar='MODE',
                        help='Run a custom sequence of modes in given order')
    args = parser.parse_args()

    # ── Determine which modes to run ──────────────────────────────────────────
    # If any CLI flag is given, use the existing CLI-driven logic (no prompt).
    # If the script is called with no arguments, show the interactive selector.
    cli_driven = args.sequence or args.only or args.start_from or args.dry_run

    if cli_driven:
        # Legacy / CI / scripted path — preserve original behaviour exactly
        if args.sequence:
            modes = args.sequence
        elif args.only:
            modes = args.only
        elif args.start_from:
            idx = FULL_SEQUENCE.index(args.start_from) if args.start_from in FULL_SEQUENCE else 0
            modes = FULL_SEQUENCE[idx:]
        else:
            modes = FULL_SEQUENCE
    else:
        # Interactive path — ask the user to choose Option 1 or 2
        modes = prompt_run_selection()

    node_name = os.uname()[1]
    print_plan(modes, node_name)

    if args.dry_run:
        print("  DRY RUN — nothing executed.\n")
        return

    log = setup_logging(node_name)
    log.info(f"Node: {node_name}")
    log.info(f"Mode sequence: {modes}")

    # Import framework
    log.info("Importing SC26 framework...")
    fw = import_framework()
    log.info(f"  Framework loaded. OUTPUT_DIR = {fw.OUTPUT_DIR}")

    # Verify NVML is available before starting long runs
    try:
        fw.nvmlInit()
        gpu_count = fw.nvmlDeviceGetCount()
        fw.nvmlShutdown()
        log.info(f"  NVML OK — {gpu_count} GPU(s) detected")
    except Exception as e:
        log.error(f"  NVML initialisation failed: {e}")
        log.error("  Check: nvidia-smi works and pynvml is installed.")
        sys.exit(1)

    # ── Execute modes in order ─────────────────────────────────────────────────
    batch_start = time.time()
    completed   = []
    failed      = []

    for mode in modes:
        log.info(f"\n{'#'*100}")
        log.info(f"  BATCH STEP: MODE {mode}")
        log.info(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(f"{'#'*100}")

        try:
            if mode == 1:
                run_mode_1(fw, log)
            elif mode == 4:
                run_mode_4(fw, log)
            elif mode == 5:
                run_mode_5(fw, log)
            elif mode == 6:
                run_mode_6(fw, log)
            elif mode == 7:
                run_mode_7(fw, log, node_name)
            elif mode == 8:
                run_mode_8(fw, log, node_name)
            elif mode == 10:
                run_mode_10(fw, log, node_name)
            elif mode == 11:
                run_mode_11(fw, log, node_name)
            elif mode == 9:
                log.warning("Mode 9 (Multi-Node Cross-Validation) requires 2 nodes.")
                log.warning("Run this separately when both nodes are ready.")
                log.warning("Command: python RUN_Batch.py --only 9")
                continue
            else:
                log.warning(f"Mode {mode} not recognised in batch runner. Skipping.")
                continue

            completed.append(mode)
            log.info(f"  ✓ Mode {mode} completed successfully.")

        except KeyboardInterrupt:
            log.warning(f"\n  ⚠  Interrupted during Mode {mode}.")
            log.warning("  Completed so far: " + str(completed))
            print("\n  To resume from next mode, run:")
            next_modes = [m for m in FULL_SEQUENCE if m not in completed]
            print(f"    python RUN_Batch.py --sequence {' '.join(map(str, next_modes))}")
            sys.exit(0)

        except Exception as e:
            log.error(f"  ✗ Mode {mode} FAILED: {e}", exc_info=True)
            failed.append(mode)
            log.warning("  Continuing with next mode...")

    # ── Final summary ──────────────────────────────────────────────────────────
    total_elapsed = (time.time() - batch_start) / 3600
    log.info("\n" + "=" * 100)
    log.info("  BATCH RUN COMPLETE")
    log.info("=" * 100)
    log.info(f"  Total elapsed:  {total_elapsed:.2f}h")
    log.info(f"  Completed:      {completed}")
    log.info(f"  Failed:         {failed if failed else 'None'}")
    log.info(f"  Output dir:     {fw.OUTPUT_DIR}/{node_name}/")

    if 9 not in modes:
        log.info("")
        log.info("  NEXT STEP (run on second node):")
        log.info("    python RUN_Batch.py --only 9")
        log.info("    (Then compare slopes from both nodes for Mode 9 cross-validation)")

    if failed:
        log.info(f"\n  To retry failed modes: python RUN_Batch.py --sequence {' '.join(map(str, failed))}")

    log.info("=" * 100)

if __name__ == "__main__":
    main()
