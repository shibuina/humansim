import importlib
import sys

VERBS = {
    "sweep": "arena_humansim.utils.evaluation.cli.sweep",
    "analyze": "arena_humansim.utils.evaluation.cli.analyze",
    "monitor": "arena_humansim.utils.evaluation.cli.monitor",
    "benchmark": "arena_humansim.utils.evaluation.cli.benchmark",
    "test": "arena_humansim.utils.evaluation.cli.test",
    "backfill_snapshots": "arena_humansim.utils.evaluation.cli.backfill_snapshots",
    "verify": "arena_humansim.utils.evaluation.cli.verify",
    "export": "arena_humansim.utils.evaluation.cli.export",
    "render": "arena_humansim.utils.evaluation.cli.render",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in VERBS:
        verbs = " | ".join(VERBS)
        sys.exit(f"usage: python3 -m arena_humansim.utils.evaluation.cli <{verbs}> [args...]")
    verb = sys.argv[1]
    sys.argv = [verb, *sys.argv[2:]]
    importlib.import_module(VERBS[verb]).main()


if __name__ == "__main__":
    main()
