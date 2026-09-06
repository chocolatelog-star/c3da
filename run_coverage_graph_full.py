"""Run the historical-best tail from an existing Coverage-only graph extractor."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from run_plan_a_graph_best import build_adapter_manifest, build_full_command


def main() -> int:
    parser = argparse.ArgumentParser(description="Reuse a completed Coverage-only Phase A and evaluate final target F1")
    parser.add_argument("--phase_a_output", required=True)
    parser.add_argument("--full_output_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    phase_a = Path(args.phase_a_output).resolve()
    # Accept either the treatment-only run root or its treatment directory.
    # The manifest lives at the run root while model/pseudo artifacts live
    # under treatment/.
    if (phase_a / "treatment_only_entry.json").is_file() and (phase_a / "treatment").is_dir():
        treatment_dir = phase_a / "treatment"
    elif (phase_a / "models" / "extractor" / "best" / "config.json").is_file():
        treatment_dir = phase_a
        phase_a = phase_a.parent
    else:
        raise FileNotFoundError(f"Coverage-only Phase A result not found: {phase_a}")
    adapter = phase_a.parent / f"{phase_a.name}_full_adapter"
    manifest = build_adapter_manifest(
        treatment_dir,
        adapter,
        source="laptop14",
        target="rest15",
        seed=args.seed,
        variant="coverage_only",
    )
    command = build_full_command(
        project_root=Path(__file__).resolve().parent,
        adapter=adapter,
        output_root=Path(args.full_output_root),
        model_path=args.model_path,
        train_batch_size=args.train_batch_size,
        accumulation=args.gradient_accumulation_steps,
        cuda=args.cuda,
        seed=args.seed,
        source_dataset="laptop14",
        target_dataset="rest15",
    )
    record = {"phase_a": str(phase_a), "adapter": manifest, "command": command, "target_test_access": True}
    (phase_a / "coverage_full_command.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        print(" ".join(command))
        return 0
    subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
