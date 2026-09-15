"""Source-only Sparkov external-validation authorization planning.

This command can only print a plan or dry-run report.  It never writes an
authorization, creates a runtime attempt, imports a model, or queries a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scripts.run_external_validation_v1 import build_sparkov_validation_plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "dry-run"), required=True)
    parser.add_argument("--approval-text", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_sparkov_validation_plan(
        repo_root=args.repo_root.resolve(),
        config_path=args.config.resolve(),
        approval_text=args.approval_text,
        mode=args.mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
