import argparse
from pathlib import Path

from .data import collect, DataAccessError
from .io import read_json
from .workflow import prepare_run, run, verify_run


def main():
    parser = argparse.ArgumentParser(description="TSETMC retrospective research pipeline")
    parser.add_argument("stage", choices=["import", "import-hourly", "collect", "prepare", "train", "all", "verify"])
    parser.add_argument("--unadjusted", type=Path)
    parser.add_argument("--adjusted", type=Path)
    parser.add_argument("--export-script", type=Path)
    parser.add_argument("--hourly", type=Path)
    parser.add_argument("--daily", type=Path)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--trades", type=Path)
    parser.add_argument("--root", type=Path, default=Path("artifacts/local"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--run", type=Path, help="Saved run directory for verify")
    args = parser.parse_args()
    try:
        if args.stage == "verify":
            if args.run is None:
                parser.error("verify requires --run")
            print(verify_run(args.run))
            return
        config = read_json(args.config)
        if args.stage == "import-hourly":
            if not args.hourly or not args.daily or not args.coverage:
                parser.error("import-hourly requires --hourly, --daily and --coverage")
            from .hourly import import_hourly
            print(import_hourly(args.root, config, args.hourly, args.daily, args.coverage, args.export_script, args.trades)[0])
        elif args.stage == "import":
            if not args.unadjusted or not args.adjusted:
                parser.error("import requires --unadjusted and --adjusted")
            from .import_export import import_exports
            print(import_exports(args.root, config, args.unadjusted, args.adjusted, args.export_script)[0])
        elif args.stage == "collect":
            print(collect(args.root, config)[0])
        elif args.stage == "prepare":
            print(prepare_run(args.root, config)[0])
        else:
            print(run(args.root, config, acquire=args.stage == "all"))
    except (DataAccessError, ValueError) as error:
        parser.exit(1, f"Stopped: {error}\nNo successful deliverables are claimed for this failure.\n")


if __name__ == "__main__":
    main()
