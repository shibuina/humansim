import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from arena_humansim.local_planner import LocalPlanner
from arena_humansim.utils.evaluation.analyze import run_analysis
from arena_humansim.utils.evaluation.cli.sweep import run_sweep


def main() -> None:
    data_dir = os.environ.get("ARENA_DATA_DIR")
    if not data_dir:
        sys.exit("error: $ARENA_DATA_DIR required")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(data_dir) / "peds" / f"{ts}_smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Recordings: {run_dir}")

    planners = [name for name, info in LocalPlanner.info().items() if not info.robot_policy]
    trials = [("simple_crossing", p, "", 42) for p in planners]
    run_sweep(trials=trials, sim_duration=60, output_dir=run_dir)

    print()
    print("=== Smoke-test recordings ===")
    for mcap in sorted(run_dir.rglob("*.mcap")):
        size = mcap.stat().st_size
        print(f"  {size:>12} {mcap}")

    print()
    print("=== Pre-flight analysis ===")
    run_analysis(recordings_dirs=run_dir, out_dir=run_dir, n_bootstrap=100)

    print()
    print("=== Per-driver kinematics ===")
    kin_csv = run_dir / "kinematics_per_trial.csv"
    if kin_csv.exists():
        print(pd.read_csv(kin_csv).to_string(index=False))

    print()
    print("=== Verdict ===")
    print("Pass criteria:")
    print("  (1) Six bags exist, all within 2x size of each other.")
    print("  (2) All rows in the kinematics table have non-zero jerk/curvature.")
    print("  (3) socialgail row in particular is non-zero -- if all zero, drop socialgail from benchmark and from the abstract kicker.")
    print("  (4) headline.csv 'nav' row's ratio_K is finite (smoke run is single-scenario, so K_lo/K_hi uses row-bootstrap).")


if __name__ == "__main__":
    main()
