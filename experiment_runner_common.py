from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


def matrix_unit_dir(root: Path, batch_size: int, accumulation: int) -> Path:
    return Path(root) / f"batch{int(batch_size)}_accum{int(accumulation)}"


def assign_gpus(units: Sequence[str], gpus: Sequence[str]) -> list[tuple[str, str]]:
    if not gpus:
        raise ValueError("at least one GPU is required")
    return [(unit, gpus[index % len(gpus)]) for index, unit in enumerate(units)]


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_status(path: Path, status: str, **fields: Any) -> None:
    atomic_write_json(path, {"status": status, **fields})


def run_command(command: Sequence[str], log_path: Path, env: Mapping[str, str] | None = None) -> int:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(map(str, command)) + "\n")
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=dict(env) if env else None, check=False)
    return completed.returncode


def completed_file(path: Path) -> bool:
    return Path(path).is_file() and Path(path).stat().st_size > 0
