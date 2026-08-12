from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TypeVar


class ReproducibilityError(RuntimeError):
    pass


class GoldenMismatchError(ReproducibilityError):
    pass


RowType = TypeVar("RowType")


def console_safe_text(value: str, encoding: str | None = None) -> str:
    target_encoding = encoding or getattr(sys.stdout, "encoding", None) or "utf-8"
    return value.encode(target_encoding, errors="replace").decode(target_encoding)


def compare_observed_rows(
    stage: str, rows: Sequence[dict], observed_golden_rows: int
) -> dict:
    actual_rows = len(rows)
    return {
        "stage": stage,
        "actual_rows": actual_rows,
        "observed_golden_rows": observed_golden_rows,
        "matched": actual_rows == observed_golden_rows,
    }


def apply_selection_limit(
    rows: Sequence[RowType], selection_limit: int | None
) -> list[RowType]:
    if selection_limit is None:
        return list(rows)
    return list(rows[:selection_limit])


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def semantic_text_label_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for row in read_jsonl(path):
        payload = json.dumps(
            {"label": row["label"], "text": row["text"]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update((payload + "\n").encode("utf-8"))
    return digest.hexdigest().upper()


TRAINING_ROW_FIELDS = (
    "input",
    "target",
    "sample_weight",
    "augmentation",
    "base_id",
    "id",
)


def semantic_training_rows_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for row in read_jsonl(path):
        payload = json.dumps(
            {key: row[key] for key in TRAINING_ROW_FIELDS if key in row},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update((payload + "\n").encode("utf-8"))
    return digest.hexdigest().upper()


def validate_metrics(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    tolerance: float = 1e-12,
) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise GoldenMismatchError(f"metric missing for {key}")
        actual_value = actual[key]
        if isinstance(expected_value, float):
            if abs(float(actual_value) - expected_value) > tolerance:
                raise GoldenMismatchError(
                    f"metric mismatch for {key}: {actual_value} != {expected_value}"
                )
        elif actual_value != expected_value:
            raise GoldenMismatchError(
                f"metric mismatch for {key}: {actual_value} != {expected_value}"
            )


def write_json_atomic(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def count_jsonl_rows(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


@dataclass
class RunContext:
    run_root: Path
    manifest_path: Path
    manifest: dict

    @property
    def commands_path(self) -> Path:
        return self.run_root / "commands.jsonl"

    @property
    def run_command_path(self) -> Path:
        return self.run_root / "run_command.cmd"

    @property
    def run_record_path(self) -> Path:
        return self.run_root / "RUN_RECORD_CN.md"

    @property
    def environment_path(self) -> Path:
        return self.run_root / "environment.json"

    @property
    def stage_status_path(self) -> Path:
        return self.run_root / "stage_status.json"

    def _update_stage_status(self, stage: str, record: Mapping[str, object]) -> None:
        statuses = {}
        if self.stage_status_path.exists():
            statuses = json.loads(self.stage_status_path.read_text(encoding="utf-8"))
        statuses[stage] = dict(record)
        write_json_atomic(self.stage_status_path, statuses)

    def capture_environment(
        self, python_executable: str, model_paths: Iterable[Path]
    ) -> dict:
        packages = {}
        for name in ("torch", "transformers", "accelerate"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None

        cuda = {
            "available": False,
            "runtime": None,
            "cudnn": None,
            "gpu": None,
            "driver": None,
        }
        try:
            import torch

            cuda.update(
                {
                    "available": torch.cuda.is_available(),
                    "runtime": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "gpu": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else None
                    ),
                }
            )
        except Exception as error:
            cuda["torch_error"] = repr(error)
        try:
            driver = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            cuda["driver"] = driver.stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError) as error:
            cuda["driver_error"] = repr(error)

        try:
            pip_freeze = subprocess.run(
                [str(python_executable), "-m", "pip", "freeze"],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout.splitlines()
        except (OSError, subprocess.SubprocessError) as error:
            pip_freeze = [f"ERROR: {error!r}"]

        models = {}
        for model_path in model_paths:
            resolved = Path(model_path).resolve()
            if resolved.is_file():
                models[str(resolved)] = sha256_file(resolved)
            else:
                models[str(resolved)] = None

        snapshot = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "python": {
                "executable": str(python_executable),
                "version": sys.version,
                "platform": platform.platform(),
            },
            "conda": {
                "prefix": os.environ.get("CONDA_PREFIX"),
                "default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            },
            "packages": packages,
            "cuda": cuda,
            "random_environment": {
                name: os.environ.get(name)
                for name in (
                    "PYTHONHASHSEED",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "CUDA_VISIBLE_DEVICES",
                )
            },
            "models": models,
            "pip_freeze": pip_freeze,
        }
        write_json_atomic(self.environment_path, snapshot)
        return snapshot

    def write_user_command(self, command: str) -> None:
        self.run_command_path.write_text(command.rstrip() + "\n", encoding="utf-8")

    def render_run_record_cn(self) -> None:
        user_command = ""
        if self.run_command_path.exists():
            user_command = self.run_command_path.read_text(encoding="utf-8").strip()
        command_events = []
        if self.commands_path.exists():
            command_events = [
                json.loads(line)
                for line in self.commands_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        completed_events = [
            event
            for event in command_events
            if event.get("status") in {"completed", "failed", "dry_run"}
        ]
        lines = [
            "# 实验运行记录",
            "",
            "## 运行身份",
            "",
            f"- 运行编号：`{self.manifest['run_id']}`",
            f"- 配方编号：`{self.manifest['recipe_id']}`",
            f"- Git 提交：`{self.manifest['git_commit']}`",
            f"- Git 分支：`{self.manifest['git_branch']}`",
            f"- 恢复次数：`{self.manifest['resume_count']}`",
            "",
            "## 完整运行命令",
            "",
            "```cmd",
            user_command,
            "```",
            "",
            "## 阶段命令",
            "",
        ]
        if not completed_events:
            lines.append("尚无已结束的阶段命令。")
        for event in completed_events:
            command = subprocess.list2cmdline(event["argv"])
            lines.extend(
                [
                    f"### {event['stage']}",
                    "",
                    f"- 状态：`{event['status']}`",
                    f"- 退出码：`{event['exit_code']}`",
                    f"- 开始时间：`{event['started_at']}`",
                    f"- 结束时间：`{event['finished_at']}`",
                    "",
                    "```cmd",
                    command,
                    "```",
                    "",
                ]
            )
        self.run_record_path.write_text("\n".join(lines), encoding="utf-8")

    def _append_jsonl(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def run_command(
        self,
        stage: str,
        argv: list[str],
        cwd: Path | None = None,
        dry_run: bool = False,
    ) -> subprocess.CompletedProcess:
        command_id = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc).isoformat()
        command = [str(part) for part in argv]
        base_record = {
            "command_id": command_id,
            "stage": stage,
            "argv": command,
            "cwd": str(Path(cwd or self.run_root).resolve()),
            "started_at": started_at,
        }
        self._append_jsonl(
            self.commands_path,
            {**base_record, "status": "running", "finished_at": None, "exit_code": None},
        )
        self._update_stage_status(
            stage,
            {
                "status": "running",
                "command_id": command_id,
                "started_at": started_at,
                "finished_at": None,
                "exit_code": None,
            },
        )
        if dry_run:
            finished_at = datetime.now(timezone.utc).isoformat()
            self._append_jsonl(
                self.commands_path,
                {
                    **base_record,
                    "status": "dry_run",
                    "finished_at": finished_at,
                    "exit_code": 0,
                },
            )
            self._update_stage_status(
                stage,
                {
                    "status": "dry_run",
                    "command_id": command_id,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "exit_code": 0,
                },
            )
            return subprocess.CompletedProcess(command, 0)

        log_path = self.run_root / "logs" / f"{stage}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            command,
            cwd=str(cwd or self.run_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        with log_path.open("a", encoding="utf-8") as log_handle:
            assert process.stdout is not None
            with process.stdout:
                for line in process.stdout:
                    print(console_safe_text(line), end="", flush=True)
                    log_handle.write(line)
                    log_handle.flush()
        exit_code = process.wait()
        finished_at = datetime.now(timezone.utc).isoformat()
        status = "completed" if exit_code == 0 else "failed"
        self._append_jsonl(
            self.commands_path,
            {
                **base_record,
                "status": status,
                "finished_at": finished_at,
                "exit_code": exit_code,
            },
        )
        self._update_stage_status(
            stage,
            {
                "status": status,
                "command_id": command_id,
                "started_at": started_at,
                "finished_at": finished_at,
                "exit_code": exit_code,
            },
        )
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, command)
        return subprocess.CompletedProcess(command, exit_code)

    @classmethod
    def create(
        cls,
        run_root: Path,
        run_id: str,
        recipe_id: str,
        git_commit: str,
        git_branch: str,
    ) -> "RunContext":
        return cls.open_or_create(
            run_root, run_id, recipe_id, git_commit, git_branch
        )

    @classmethod
    def open_or_create(
        cls,
        run_root: Path,
        run_id: str,
        recipe_id: str,
        git_commit: str,
        git_branch: str,
    ) -> "RunContext":
        run_root = Path(run_root).resolve()
        identity = {
            "run_id": run_id,
            "recipe_id": recipe_id,
            "git_commit": git_commit,
            "git_branch": git_branch,
        }
        manifest_path = run_root / "manifest.json"
        if run_root.exists() and not manifest_path.exists() and any(run_root.iterdir()):
            raise ReproducibilityError(
                f"existing run directory has no manifest.json: {run_root}"
            )
        run_root.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key, expected in identity.items():
                if manifest.get(key) != expected:
                    raise ReproducibilityError(
                        f"{key} mismatch: {manifest.get(key)} != {expected}"
                    )
            manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        else:
            manifest = {
                **identity,
                "resume_count": 0,
                "stages": {},
                "artifacts": {},
            }
        write_json_atomic(manifest_path, manifest)
        return cls(
            run_root=run_root,
            manifest_path=manifest_path,
            manifest=manifest,
        )

    def require_internal_artifact(self, path: Path) -> Path:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self.run_root):
            raise ReproducibilityError(
                f"artifact is outside current run root: {resolved}"
            )
        return resolved

    def record_artifact(
        self,
        stage: str,
        path: Path,
        input_hashes: Mapping[str, str] | None = None,
        semantic_hash: str = "",
    ) -> dict:
        resolved = self.require_internal_artifact(path)
        if not resolved.is_file():
            raise ReproducibilityError(f"artifact is missing or not a file: {resolved}")
        record = {
            "producer_stage": stage,
            "path": str(resolved),
            "sha256": sha256_file(resolved),
            "input_hashes": dict(input_hashes or {}),
            "semantic_sha256": semantic_hash,
        }
        self.manifest["artifacts"][str(resolved)] = record
        write_json_atomic(self.manifest_path, self.manifest)
        return record

    def mark_stage_complete(
        self,
        stage: str,
        outputs: Iterable[Path],
        input_hashes: Mapping[str, str] | None = None,
        argv: Iterable[str] = (),
    ) -> None:
        resolved_outputs = [self.require_internal_artifact(path) for path in outputs]
        command = [str(part) for part in argv]
        command_sha256 = hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for output in resolved_outputs:
            self.record_artifact(stage, output, input_hashes)
        self.manifest["stages"][stage] = {
            "status": "completed",
            "outputs": [str(path) for path in resolved_outputs],
            "input_hashes": dict(input_hashes or {}),
            "argv": command,
            "argv_sha256": command_sha256,
        }
        write_json_atomic(self.manifest_path, self.manifest)

    def validate_completed_stage(
        self,
        stage: str,
        outputs: Iterable[Path],
        input_hashes: Mapping[str, str] | None = None,
        argv: Iterable[str] = (),
    ) -> bool:
        stage_record = self.manifest["stages"].get(stage)
        if not stage_record or stage_record.get("status") != "completed":
            return False
        expected_outputs = [str(self.require_internal_artifact(path)) for path in outputs]
        if stage_record.get("outputs") != expected_outputs:
            raise ReproducibilityError(f"stage output mismatch: {stage}")
        expected_argv = [str(part) for part in argv]
        expected_argv_sha256 = hashlib.sha256(
            json.dumps(expected_argv, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if (
            stage_record.get("argv") != expected_argv
            or stage_record.get("argv_sha256") != expected_argv_sha256
        ):
            raise ReproducibilityError(f"stage command mismatch: {stage}")
        expected_input_hashes = dict(input_hashes or {})
        if stage_record.get("input_hashes", {}) != expected_input_hashes:
            raise ReproducibilityError(
                f"input hash mismatch for stage {stage}: "
                f"{stage_record.get('input_hashes', {})} != {expected_input_hashes}"
            )
        for output in expected_outputs:
            path = Path(output)
            artifact = self.manifest["artifacts"].get(output)
            if artifact is None or not path.is_file():
                raise ReproducibilityError(f"stage artifact is missing: {path}")
            actual_hash = sha256_file(path)
            if actual_hash != artifact.get("sha256"):
                raise ReproducibilityError(
                    f"artifact hash mismatch: {path}: "
                    f"{actual_hash} != {artifact.get('sha256')}"
                )
        return True
