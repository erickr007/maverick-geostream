from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None

from api.dependencies import Settings
from services.generator.generator import Generator
from services.generator.plan import build_plan


def _load_settings(config_path: str | None) -> Settings:
    if config_path is None:
        return Settings()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML config files. Install it with: pip install PyYAML")
        data = yaml.safe_load(raw_text) or {}
    else:
        import json as _json

        data = _json.loads(raw_text)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a dictionary object: {path}")

    return Settings(**data)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Build the device and shard plan for a run")
    plan_parser.add_argument("--config", type=str, default=None, help="Optional config file path")
    plan_parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    plan_parser.add_argument("--out", type=str, default="./output", help="Output root; the plan is written to <out>/plan")

    generate_parser = subparsers.add_parser("generate", help="Generate data and write parquet output")
    generate_parser.add_argument("--config", type=str, default=None, help="Optional config file path")
    generate_parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    generate_parser.add_argument("--workers", type=int, default=1, help="Number of worker shards to target")
    generate_parser.add_argument("--out", type=str, default="./output", help="Output directory for generated parquet files")

    verify_parser = subparsers.add_parser("verify", help="Validate parquet output and distribution checks")
    verify_parser.add_argument("--out", type=str, default="./output", help="Output directory to verify")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "plan":
            settings = _load_settings(args.config)
            summary = build_plan(settings, seed=args.seed, out_dir=args.out)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        if args.command == "generate":
            settings = _load_settings(args.config)
            generator = Generator(settings=settings, seed=args.seed, workers=args.workers, out_dir=args.out)
            summary = generator.generate(out_dir=args.out, workers=args.workers)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        if args.command == "verify":
            generator = Generator(out_dir=args.out)
            summary = generator.verify(out_dir=args.out)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        parser.error(f"Unknown command: {args.command}")
        return 2
    except Exception as exc:  # pragma: no cover - CLI error path
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
