"""Run selected stages of the LangPro base dataset construction pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


CONSTRUCTION_DIR = Path(__file__).resolve().parent
STEPS_DIR = CONSTRUCTION_DIR / "steps"

GROUP_PREFIXES = {
    "core": ("01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_", "09_", "10_"),
    "structure": ("11_", "12_", "13_", "14_", "15_", "16"),
    "literature": ("17_", "18_", "19_", "20", "21", "22", "23_", "24", "25"),
    "finalize": ("26_",),
}


def available_steps() -> list[Path]:
    return sorted(STEPS_DIR.glob("*.py"), key=lambda path: path.name.lower())


def steps_for_group(group: str) -> list[Path]:
    prefixes = GROUP_PREFIXES[group]
    return [path for path in available_steps() if path.name.startswith(prefixes)]


def resolve_step(name: str) -> Path:
    requested = name if name.endswith(".py") else f"{name}.py"
    candidate = STEPS_DIR / requested
    if not candidate.is_file():
        choices = ", ".join(path.stem for path in available_steps())
        raise SystemExit(f"Unknown step '{name}'. Available steps: {choices}")
    return candidate


def run_steps(steps: Iterable[Path], dry_run: bool, keep_going: bool) -> None:
    failed: list[str] = []
    for step in steps:
        command = [sys.executable, str(step)]
        print(f"[construction] {' '.join(command)}")
        if dry_run:
            continue
        result = subprocess.run(command, cwd=CONSTRUCTION_DIR)
        if result.returncode:
            failed.append(step.name)
            if not keep_going:
                raise SystemExit(result.returncode)
    if failed:
        raise SystemExit(f"Failed stages: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LangPro dataset construction stages.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--step", help="Run one stage by filename or stem.")
    selection.add_argument("--group", choices=tuple(GROUP_PREFIXES), help="Run a stage group.")
    selection.add_argument("--all", action="store_true", help="Run every stage.")
    selection.add_argument("--list", action="store_true", help="List available stages.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed stage.")
    args, stage_args = parser.parse_known_args()

    if args.list:
        for group in GROUP_PREFIXES:
            print(f"{group}:")
            for step in steps_for_group(group):
                print(f"  {step.stem}")
        return

    if args.step:
        steps = [resolve_step(args.step)]
    elif args.group:
        steps = steps_for_group(args.group)
    elif args.all:
        steps = available_steps()
    else:
        parser.error("choose --step, --group, --all, or --list")

    if stage_args:
        if len(steps) != 1:
            parser.error("additional stage arguments can only be used with --step")
        if stage_args[0] == "--":
            stage_args = stage_args[1:]
        command = [sys.executable, str(steps[0]), *stage_args]
        print(f"[construction] {' '.join(command)}")
        if not args.dry_run:
            raise SystemExit(subprocess.run(command, cwd=CONSTRUCTION_DIR).returncode)
        return

    run_steps(steps, dry_run=args.dry_run, keep_going=args.keep_going)


if __name__ == "__main__":
    main()
