"""Production commands; all original research CLI behavior is preserved."""
import argparse
import json
import logging
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Verified hourly LSTM historical replay service")
    commands = parser.add_subparsers(dest="command", required=True)
    bundle = commands.add_parser("bundle")
    bundle.add_argument("--source", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)
    for name in ["verify", "register", "replay-evaluate", "serve"]:
        sub = commands.add_parser(name)
        sub.add_argument("--bundle", type=Path, default=os.environ.get("MARKETFORECAST_BUNDLE", "/bundle"))
        if name in ["serve", "replay-evaluate"]:
            sub.add_argument("--state", type=Path, default=os.environ.get("MARKETFORECAST_STATE", "/state/audit.sqlite"))
        if name == "register":
            sub.add_argument("--uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
            sub.add_argument("--output", type=Path, default="/registry/registration.json")
            sub.add_argument("--promote", action="store_true")
        if name == "serve":
            sub.add_argument("--host", default="0.0.0.0")
            sub.add_argument("--port", type=int, default=8000)
    backup = commands.add_parser("backup")
    backup.add_argument("--output", type=Path, required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.command == "bundle":
        from .bundle import build_bundle
        result = build_bundle(args.source, args.output)
    elif args.command == "verify":
        from .bundle import verify_bundle
        result = verify_bundle(args.bundle)
    elif args.command == "register":
        from .registry import register
        result = register(args.bundle, args.uri, args.output, args.promote)
    elif args.command == "replay-evaluate":
        from .runtime import Predictor
        from .audit import AuditStore
        from .monitoring import Monitor
        result = Monitor(Predictor(args.bundle), AuditStore(args.state)).evaluate()
    elif args.command == "backup":
        from .operations import backup
        result = backup(args.output)
    elif args.command == "restore":
        from .operations import restore
        result = restore(args.input)
    else:
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(args.bundle, args.state), host=args.host, port=args.port,
                    workers=1, access_log=False, timeout_graceful_shutdown=30)
        return
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
