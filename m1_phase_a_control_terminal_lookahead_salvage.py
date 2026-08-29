"""只读审计旧 V6 Control 的终态 DataLoader lookahead。

本脚本不恢复、不改写、不删除 V6。它只把一个严格满足条件的
issued-but-unprocessed（已签发但未处理）末尾批次标记为
``lookahead_not_consumed``，供新的 Phase A 入口决定是否允许外部复用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from t5_absa_train import classify_terminal_lookahead, read_dann_audit_journal


PRODUCER_TRAINING_SEMANTICS_COMMIT = "9caba1c508d096a4d360d7940d8c9d9eb4be8333"


def _utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and ".tmp" not in path.name),
        key=lambda path: str(path.relative_to(root)).lower(),
    )
    for path in files:
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def audit_v6_control(v6_run_dir: str | Path, output_path: str | Path | None = None) -> dict:
    source = Path(v6_run_dir).resolve()
    control = source / "control"
    audit_path = control / "dann_batch_audit.json"
    journal_path = control / "dann_batch_audit.journal.jsonl"
    conditions: dict[str, dict[str, Any]] = {}
    report: dict = {}
    journal: dict = {}

    def condition(name: str, passed: bool, detail: Any = None) -> None:
        conditions[name] = {"status": "PASS" if passed else "FAIL", "detail": detail}

    condition("v6_run_exists", source.is_dir(), str(source))
    stage_status_path = source / "stage_status.json"
    stage_status = _read_json(stage_status_path) if stage_status_path.is_file() else {}
    completed_stages = stage_status.get("completed_stages") if isinstance(stage_status, dict) else None
    scope = (stage_status.get("identity") or {}).get("scope") if isinstance(stage_status, dict) else None
    condition(
        "producer_identity",
        isinstance(stage_status, dict)
        and (stage_status.get("identity") or {}).get("code_commit") == PRODUCER_TRAINING_SEMANTICS_COMMIT,
        (stage_status.get("identity") or {}).get("code_commit") if isinstance(stage_status, dict) else None,
    )
    condition(
        "control_only_completed_stage",
        completed_stages == ["control_training"],
        completed_stages,
    )
    condition(
        "target_test_not_accessed",
        isinstance(scope, dict)
        and scope.get("target_test_access") is False
        and (scope.get("forbidden") or {}).get("target_test") is False,
        scope.get("target_test_access") if isinstance(scope, dict) else None,
    )
    if audit_path.is_file():
        try:
            report = _read_json(audit_path)
            condition("dann_audit_readable", isinstance(report, dict), str(audit_path))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            condition("dann_audit_readable", False, str(exc))
    else:
        condition("dann_audit_readable", False, f"missing: {audit_path}")
    if journal_path.is_file():
        try:
            journal = read_dann_audit_journal(journal_path)
            condition("journal_chain_complete", journal.get("chain_valid") is True, journal.get("record_count"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            condition("journal_chain_complete", False, str(exc))
    else:
        condition("journal_chain_complete", False, f"missing: {journal_path}")

    decision = classify_terminal_lookahead(
        report,
        gradient_accumulation_steps=int(report.get("gradient_accumulation_steps") or 16),
        journal_audit=journal or None,
    )
    condition("terminal_lookahead", bool(decision.get("safe") and decision.get("lookahead_not_consumed")), decision)
    condition("fresh_replay_count_zero", journal.get("replay_count") == 0, journal.get("replay_count"))

    global_step = report.get("trainer_global_step")
    max_steps = report.get("trainer_max_steps")
    condition("trainer_reached_max_steps", isinstance(global_step, int) and global_step == max_steps and global_step > 0, {"global_step": global_step, "max_steps": max_steps})

    checkpoint_root = control / "models" / "extractor"
    checkpoint_dirs = []
    if checkpoint_root.is_dir():
        for candidate in checkpoint_root.glob("checkpoint-*"):
            try:
                step = int(candidate.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            if candidate.is_dir():
                checkpoint_dirs.append((step, candidate))
    checkpoint_dirs.sort(key=lambda item: item[0])
    latest_step, latest = checkpoint_dirs[-1] if checkpoint_dirs else (None, None)
    condition("no_checkpoint_after_terminal_step", isinstance(global_step, int) and all(step <= global_step for step, _ in checkpoint_dirs), [(step, str(path)) for step, path in checkpoint_dirs])
    condition("terminal_checkpoint_model_present", latest is not None and latest_step == global_step and any((latest / name).is_file() for name in ("pytorch_model.bin", "model.safetensors")), str(latest) if latest else None)
    gradient_state = None
    if latest is not None and (latest / "dann_gradient_state.pt").is_file():
        try:
            gradient_state = torch.load(latest / "dann_gradient_state.pt", map_location="cpu")
        except (OSError, RuntimeError, EOFError, ValueError):
            gradient_state = None
    condition(
        "no_uncommitted_gradient_state",
        isinstance(gradient_state, dict)
        and gradient_state.get("accumulation_remainder") == 0
        and gradient_state.get("gradients") == {},
        {
            "accumulation_remainder": gradient_state.get("accumulation_remainder") if isinstance(gradient_state, dict) else None,
            "gradient_count": len(gradient_state.get("gradients", {})) if isinstance(gradient_state, dict) and isinstance(gradient_state.get("gradients"), dict) else None,
        },
    )

    model_path = control / "models" / "extractor" / "best"
    condition("best_model_present", model_path.is_dir(), str(model_path))
    result = {
        "schema_version": 1,
        "status": "PASS" if all(item["status"] == "PASS" for item in conditions.values()) else "BLOCKED",
        "classification": decision,
        "source_run_dir": str(source),
        "source_run_remains_blocked": True,
        "dann_audit_path": str(audit_path),
        "dann_audit_sha256": _sha256(audit_path) if audit_path.is_file() else None,
        "journal_path": str(journal_path),
        "journal_sha256": _sha256(journal_path) if journal_path.is_file() else None,
        "model_path": str(model_path),
        "model_tree_sha256": _hash_tree(model_path) if model_path.is_dir() else None,
        "conditions": conditions,
        "producer_training_semantics_commit": PRODUCER_TRAINING_SEMANTICS_COMMIT,
        "orchestration_commit": _git_commit(Path(__file__).resolve().parent),
        "reuse_depth": 1,
        "target_test_access": False,
        "source_artifacts_modified": False,
        "audited_at": _utc_now(),
    }
    if output_path is None:
        output_path = source.parents[2] / "diagnostics" / "m1_phase_a_terminal_lookahead_salvage_v1" / "control_terminal_lookahead_salvage_audit.json"
    _atomic_write_json(Path(output_path).resolve(), result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only V6 terminal lookahead salvage audit")
    parser.add_argument("--v6_control_run_dir", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    result = audit_v6_control(args.v6_control_run_dir, args.output or None)
    print(json.dumps({"status": result["status"], "report": str(Path(args.output).resolve()) if args.output else None}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
