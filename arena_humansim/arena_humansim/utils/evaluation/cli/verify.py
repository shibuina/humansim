"""Retroactive integrity check over a sweep run dir.

    evaluate verify --run_dir <dir>            # report
    evaluate verify --run_dir <dir> --fix      # also drop .done so --resume runs them again

`ros2 launch` exits 0 when `arena_humansim_node` exits 1, which is what a policy that
fails to import does. The sweep then touches `.done` over a stub bag that simulated
nothing. This verb applies the in-worker gate's tripwires to trials recorded before
the gate existed.

Tripwires, most conclusive first:

  finished_at    the recording manifest gets it from BagRecorder.close(), which only
                 runs from a clean destroy_node()
  overhead_row   fallback for trials without a recording manifest, a row in
                 launch_overhead.csv is written only after the final RTF line
  bag_bytes      under STUB_BAG_BYTES means the recorder wrote a header and little else
                 (skipped for a pruned trial, no bag dir but an _extracted.json)
  n_frames       under MIN_BAG_FRAMES recorded frames, from <trial>/_extracted.json
  contaminated   the recording manifest reports more than one publisher on a contract topic
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

from arena_humansim.utils.evaluation.cli._resume import latest_sweep_dir
from arena_humansim.utils.evaluation.cli.sweep import MIN_BAG_FRAMES, STUB_BAG_BYTES, _bag_bytes

_Key = tuple[str, str, str, str]


def _overhead_keys(run_dir: Path) -> set[_Key]:
    path = run_dir / "launch_overhead.csv"
    if not path.exists():
        return set()
    keys: set[_Key] = set()
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            keys.add((row["scenario"], row["planner"], row["robot_policy"], str(row["seed"])))
    return keys


def _trial_key(name: str) -> _Key | None:
    parts = name.split("__")
    if len(parts) == 3:
        return parts[0], parts[1], "", parts[2]
    if len(parts) == 4:
        return parts[0], parts[1], parts[2], parts[3]
    return None


def verify_trial(trial_dir: Path, overhead: set[_Key]) -> list[str]:
    problems: list[str] = []
    if not (trial_dir / ".done").exists():
        problems.append("not done")
    pruned = not (trial_dir / "bag").is_dir() and (trial_dir / "_extracted.json").exists()
    if not pruned:
        nbytes = _bag_bytes(trial_dir)
        if nbytes < STUB_BAG_BYTES:
            problems.append(f"stub bag ({nbytes} bytes)")
    manifest = trial_dir / "recording.yaml"
    if manifest.exists():
        try:
            info = yaml.safe_load(manifest.read_text()) or {}
        except (OSError, yaml.YAMLError) as exc:
            info = {}
            problems.append(f"unreadable recording.yaml: {exc}")
        if info.get("contaminated"):
            problems.append(f"contaminated: publishers={info.get('publishers')}")
        if not info.get("finished_at"):
            problems.append("recording.yaml has no finished_at (the node never shut down cleanly)")
    else:
        key = _trial_key(trial_dir.name)
        if overhead and key is not None and key not in overhead:
            problems.append("no launch_overhead row (tick loop never finished)")
    extracted = trial_dir / "_extracted.json"
    if extracted.exists():
        try:
            first = json.loads(extracted.read_text().splitlines()[0])
        except (ValueError, IndexError):
            problems.append("_extracted.json is not JSON")
        else:
            if first.get("ok") is False:
                problems.append(f"extraction failed: {first.get('error')}")
            frames = first.get("n_frames")
            if isinstance(frames, int) and frames < MIN_BAG_FRAMES:
                problems.append(f"only {frames} recorded frames")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", type=str, default=None, help="sweep run dir (default: latest under $ARENA_DATA_DIR/peds)")
    parser.add_argument("--fix", action="store_true", help="remove .done from failed trials so --resume re-runs them")
    parser.add_argument("--out", type=str, default=None, help="write the per-trial verdicts to this CSV")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        found = latest_sweep_dir()
        if found is None:
            sys.exit("no run dir found; pass --run_dir")
        run_dir = found[0]
    if not run_dir.is_dir():
        sys.exit(f"{run_dir} is not a directory")

    overhead = _overhead_keys(run_dir)
    trials = sorted(d for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith((".", "_")) and d.name not in ("logs", "progress", "trials"))

    verdicts = []
    bad = 0
    for trial in trials:
        problems = verify_trial(trial, overhead)
        verdicts.append((trial.name, not problems, "; ".join(problems)))
        if problems:
            bad += 1
            print(f"FAIL {trial.name}: {'; '.join(problems)}")
            if args.fix:
                sentinel = trial / ".done"
                if sentinel.exists():
                    sentinel.unlink()

    print()
    print(f"{len(trials) - bad}/{len(trials)} trials clean under {run_dir}")
    if bad and args.fix:
        print(f"dropped .done from {bad} trials; re-invoke `evaluate benchmark --run_dir {run_dir} --resume`")
    elif bad:
        print("re-run with --fix to drop their .done sentinels")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["trial_id", "clean", "problems"])
            for name, clean, problems in verdicts:
                w.writerow([name, int(clean), problems])
        print(f"verdicts: {args.out}")

    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
