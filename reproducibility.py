from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


class ReproducibilityError(RuntimeError):
    pass


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
    ) -> None:
        resolved_outputs = [self.require_internal_artifact(path) for path in outputs]
        for output in resolved_outputs:
            self.record_artifact(stage, output, input_hashes)
        self.manifest["stages"][stage] = {
            "status": "completed",
            "outputs": [str(path) for path in resolved_outputs],
        }
        write_json_atomic(self.manifest_path, self.manifest)

    def validate_completed_stage(
        self, stage: str, outputs: Iterable[Path]
    ) -> bool:
        stage_record = self.manifest["stages"].get(stage)
        if not stage_record or stage_record.get("status") != "completed":
            return False
        expected_outputs = [str(self.require_internal_artifact(path)) for path in outputs]
        if stage_record.get("outputs") != expected_outputs:
            raise ReproducibilityError(f"stage output mismatch: {stage}")
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
