from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

from syntactic_graph import (
    ALIGNMENT_POLICY_VERSION,
    DEFAULT_PARSER_DIR,
    EXPECTED_PARSER_SHA256,
    GRAPH_SCHEMA_VERSION,
    GraphCacheError,
    _parser_words_from_doc,
    build_parser_identity,
    build_stanza_pipeline,
    build_tokenizer_identity,
    sha256_file,
    validate_alignment_policy,
)
from t5_aste_pipeline import DATASETS, load_split


SPLITS = ("source_train", "source_dev", "target_unlabeled")
OUTPUT_FILES = (
    "alignment_preflight_summary.json",
    "alignment_suspicious_rows.jsonl",
    "alignment_preflight_CN.md",
)
DEFAULT_MODEL_PATH = Path(r"J:\nlp\models\t5-base-py")
DEFAULT_MAX_SOURCE_LENGTH = 128
DEFAULT_MAX_SUBWORDS_PER_WORD = 8
CODE_IDENTITY_FILES = (
    "m1_alignment_preflight.py",
    "syntactic_graph.py",
    "t5_aste_data.py",
    "t5_aste_pipeline.py",
)
STAT_KEYS = (
    "unaligned_word_count",
    "character_coverage_incomplete_count",
    "non_contiguous_shared_count",
    "cross_space_shared_count",
    "cross_sentence_shared_count",
    "out_of_bounds_count",
    "alignment_policy_violation_count",
    "legal_contiguous_shared_subword_count",
    "legal_contiguous_shared_row_count",
    "truncated_rows",
    "truncation_uncovered_word_count",
    "abnormally_many_subwords_word_count",
    "subword_maps_to_multiple_parser_words_count",
    "word_character_count",
    "covered_character_count",
)
class AlignmentPreflightError(RuntimeError):
    """Raised when preflight state cannot be trusted or resumed safely."""


class AlignmentPreflightInterrupted(AlignmentPreflightError):
    """Raised only by the testable interruption hook after a durable row."""


def _empty_stats() -> dict:
    return {key: 0 for key in STAT_KEYS}


def _empty_split_report(row_count: int) -> dict:
    return {
        "row_count": int(row_count),
        "completed_rows": 0,
        "successful_rows": 0,
        "failed_rows": 0,
        "sentence_count": 0,
        "word_count": 0,
        "subword_count": 0,
        "suspicious_row_count": 0,
        "stats": _empty_stats(),
        "distributions": {
            "word_subword_count": {},
            "subword_parser_word_count": {},
        },
    }


def _increment_distribution(distribution: dict, value: int, amount: int = 1) -> None:
    key = str(int(value))
    distribution[key] = int(distribution.get(key, 0)) + int(amount)


def _merge_row_into_split(split_report: dict, row_result: dict) -> None:
    split_report["completed_rows"] += 1
    if row_result["failed"]:
        split_report["failed_rows"] += 1
    else:
        split_report["successful_rows"] += 1
    split_report["sentence_count"] += int(row_result.get("sentence_count", 0))
    split_report["word_count"] += int(row_result.get("word_count", 0))
    split_report["subword_count"] += int(row_result.get("subword_count", 0))
    split_report["suspicious_row_count"] += len(row_result.get("suspicious_rows", []))
    for key in STAT_KEYS:
        split_report["stats"][key] = int(split_report["stats"].get(key, 0)) + int(
            row_result.get("stats", {}).get(key, 0)
        )
    for name in ("word_subword_count", "subword_parser_word_count"):
        for key, value in row_result.get("distributions", {}).get(name, {}).items():
            split_report["distributions"][name][str(key)] = int(
                split_report["distributions"][name].get(str(key), 0)
            ) + int(value)


def _normalize_split_report(report: dict, row_count: int | None = None) -> dict:
    normalized = _empty_split_report(int(report.get("row_count", row_count or 0)))
    for key in (
        "completed_rows",
        "successful_rows",
        "failed_rows",
        "sentence_count",
        "word_count",
        "subword_count",
        "suspicious_row_count",
    ):
        normalized[key] = int(report.get(key, 0))
    for key in STAT_KEYS:
        normalized["stats"][key] = int(report.get("stats", {}).get(key, 0))
    for name in normalized["distributions"]:
        normalized["distributions"][name] = {
            str(key): int(value) for key, value in report.get("distributions", {}).get(name, {}).items()
        }
    return normalized


def _aggregate_split_reports(split_reports: dict[str, dict]) -> dict:
    total = _empty_split_report(0)
    total["row_count"] = sum(int(report["row_count"]) for report in split_reports.values())
    for report in split_reports.values():
        for key in (
            "completed_rows",
            "successful_rows",
            "failed_rows",
            "sentence_count",
            "word_count",
            "subword_count",
            "suspicious_row_count",
        ):
            total[key] += int(report.get(key, 0))
        for key in STAT_KEYS:
            total["stats"][key] += int(report.get("stats", {}).get(key, 0))
        for name in total["distributions"]:
            for key, value in report.get("distributions", {}).get(name, {}).items():
                total["distributions"][name][str(key)] = int(
                    total["distributions"][name].get(str(key), 0)
                ) + int(value)
    return total


def _coverage_rate(stats: dict) -> float:
    total = int(stats.get("word_character_count", 0))
    covered = int(stats.get("covered_character_count", 0))
    return 1.0 if total == 0 else covered / total


def _gate_values(split_reports: dict[str, dict], target_test_access: bool) -> dict[str, bool]:
    aggregate = _aggregate_split_reports(split_reports)
    stats = aggregate["stats"]
    all_scanned = all(report["completed_rows"] == report["row_count"] for report in split_reports.values())
    return {
        "all_splits_scanned": all_scanned,
        "successful_rows_only": aggregate["failed_rows"] == 0,
        "unaligned_words_zero": stats["unaligned_word_count"] == 0,
        "character_coverage_100_percent": _coverage_rate(stats) == 1.0,
        "non_contiguous_shared_zero": stats["non_contiguous_shared_count"] == 0,
        "cross_space_shared_zero": stats["cross_space_shared_count"] == 0,
        "cross_sentence_shared_zero": stats["cross_sentence_shared_count"] == 0,
        "out_of_bounds_zero": stats["out_of_bounds_count"] == 0,
        "alignment_policy_violation_zero": stats["alignment_policy_violation_count"] == 0,
        "truncation_uncovered_zero": stats["truncation_uncovered_word_count"] == 0,
        "target_test_isolated": target_test_access is False,
    }


def build_preflight_summary(
    split_reports: dict[str, dict],
    identity: dict,
    max_source_length: int,
    runtime_devices: dict | None = None,
) -> dict:
    """Build the machine-readable summary without reading data or writing files."""
    normalized = {
        split: _normalize_split_report(report) for split, report in split_reports.items()
    }
    for split in SPLITS:
        normalized.setdefault(split, _empty_split_report(0))
    aggregate = _aggregate_split_reports(normalized)
    gates = _gate_values(normalized, target_test_access=False)
    complete = all(gates.values())
    summary = {
        "schema_version": 1,
        "status": "PASS" if complete else "BLOCKED",
        "target_test_access": False,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "alignment_policy_version": ALIGNMENT_POLICY_VERSION,
        "runtime_devices": dict(
            runtime_devices
            or {
                "requested_cuda_index": 0,
                "actual_cuda_index": None,
                "parser_device": "unknown",
                "model_device": "unknown",
            }
        ),
        "alignment_strategy": {
            "version": ALIGNMENT_POLICY_VERSION,
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "max_source_length": int(max_source_length),
        },
        "identity": identity,
        "splits": {},
        "totals": {
            "row_count": aggregate["row_count"],
            "completed_rows": aggregate["completed_rows"],
            "successful_rows": aggregate["successful_rows"],
            "failed_rows": aggregate["failed_rows"],
            "sentence_count": aggregate["sentence_count"],
            "word_count": aggregate["word_count"],
            "subword_count": aggregate["subword_count"],
            "suspicious_row_count": aggregate["suspicious_row_count"],
            "stats": aggregate["stats"],
            "character_coverage_rate": _coverage_rate(aggregate["stats"]),
            "distributions": aggregate["distributions"],
        },
        "gates": gates,
    }
    for split, report in normalized.items():
        summary["splits"][split] = {
            **report,
            "character_coverage_rate": _coverage_rate(report["stats"]),
        }
    return summary


def _tokenizer_tokens(tokenizer, input_ids: list[int]) -> list[str]:
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    if converter is None:
        return [str(value) for value in input_ids]
    return [str(value) for value in converter(input_ids)]


def _issue_record(
    split: str,
    row_id: str,
    text: str,
    issue_type: str,
    parser_words: list[dict],
    tokenizer_tokens: list[str],
    character_spans: list[dict],
    shared_token_indices: Iterable[int],
    error_message: str,
) -> dict:
    return {
        "split": split,
        "row_id": str(row_id),
        "text": str(text),
        "issue_type": issue_type,
        "parser_words": parser_words,
        "tokenizer_tokens": tokenizer_tokens,
        "character_spans": character_spans,
        "shared_token_indices": sorted({int(value) for value in shared_token_indices}),
        "error_message": str(error_message),
    }


def _failed_row_result(split: str, row: dict, exc: Exception) -> dict:
    message = f"{type(exc).__name__}: {exc}"
    suspicious = [
        _issue_record(
            split,
            str(row.get("id", "")),
            str(row.get("text", "")),
            "row_failure",
            [],
            [],
            [],
            [],
            message,
        )
    ]
    return {
        "split": split,
        "row_id": str(row.get("id", "")),
        "text": str(row.get("text", "")),
        "failed": True,
        "issue_types": ["row_failure"],
        "sentence_count": 0,
        "word_count": 0,
        "subword_count": 0,
        "stats": _empty_stats(),
        "distributions": {"word_subword_count": {}, "subword_parser_word_count": {}},
        "suspicious_rows": suspicious,
    }


def scan_alignment_row(
    split: str,
    row: dict,
    tokenizer,
    parser,
    max_source_length: int = DEFAULT_MAX_SOURCE_LENGTH,
    max_subwords_per_word: int = DEFAULT_MAX_SUBWORDS_PER_WORD,
) -> dict:
    """Scan one row and return every issue found without raising row-level errors."""
    text = str(row.get("text", ""))
    input_text = str(row.get("input") or text)
    try:
        encoded = tokenizer(
            input_text,
            add_special_tokens=True,
            truncation=True,
            max_length=int(max_source_length),
            return_offsets_mapping=True,
        )
        full_encoded = tokenizer(
            input_text,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        input_ids = [int(value) for value in encoded.get("input_ids", [])]
        offsets = [tuple(pair) for pair in encoded.get("offset_mapping", [])]
        full_ids = list(full_encoded.get("input_ids", []))
        if not offsets:
            raise GraphCacheError("tokenizer did not return offset_mapping")
        if len(input_ids) != len(offsets):
            raise GraphCacheError("tokenizer input_ids and offset_mapping length mismatch")
        full_offsets = [tuple(pair) for pair in full_encoded.get("offset_mapping", [])]
        if len(full_ids) != len(full_offsets):
            raise GraphCacheError("full tokenizer input_ids and offset_mapping length mismatch")
        doc = parser(text)
        words = _parser_words_from_doc(doc)
        sentence_count = len(getattr(doc, "sentences", []))
        tokenizer_tokens = _tokenizer_tokens(tokenizer, input_ids)
        validation = validate_alignment_policy(words, input_text, text, offsets)
        full_validation = validate_alignment_policy(words, input_text, text, full_offsets)
        token_spans = validation["token_spans"]
        token_to_words: dict[int, list[int]] = {
            int(index): list(positions)
            for index, positions in validation["token_to_word_positions"].items()
        }
        character_spans = []
        for token_index, (token_start, token_end) in enumerate(offsets):
            relative_start, relative_end = token_spans.get(token_index, (0, 0))
            character_spans.append(
                {
                    "token_index": token_index,
                    "start": relative_start,
                    "end": relative_end,
                    "raw_start": int(token_start),
                    "raw_end": int(token_end),
                }
            )

        was_truncated = len(full_ids) > len(input_ids)
        stats = _empty_stats()
        distributions = {"word_subword_count": {}, "subword_parser_word_count": {}}
        issue_tokens: dict[str, set[int]] = {}
        issue_messages: dict[str, list[str]] = {}

        def add_issue(issue_type: str, message: str, token_indices: Iterable[int] = ()) -> None:
            issue_tokens.setdefault(issue_type, set()).update(int(value) for value in token_indices)
            issue_messages.setdefault(issue_type, []).append(str(message))

        policy_violations = validation["violations"]
        stats["alignment_policy_violation_count"] = len(policy_violations)
        if policy_violations:
            add_issue(
                "alignment_policy_violation",
                "; ".join(item["message"] for item in policy_violations),
                [
                    token_index
                    for item in policy_violations
                    for token_index in item.get("token_indices", [])
                ],
            )
        for item in policy_violations:
            add_issue(item["issue_type"], item["message"], item.get("token_indices", []))
        stats["out_of_bounds_count"] = sum(
            item["issue_type"] == "out_of_bounds_mapping" for item in policy_violations
        )
        stats["non_contiguous_shared_count"] = sum(
            item["issue_type"] == "non_contiguous_shared_subword" for item in policy_violations
        )
        stats["cross_space_shared_count"] = sum(
            item["issue_type"] == "cross_space_shared_subword" for item in policy_violations
        )
        stats["cross_sentence_shared_count"] = sum(
            item["issue_type"] == "cross_sentence_shared_subword" for item in policy_violations
        )

        coverage_by_position = {
            int(item["word_position"]): item for item in validation["word_coverage"]
        }
        full_coverage_by_position = {
            int(item["word_position"]): item for item in full_validation["word_coverage"]
        }
        for word_position, word in enumerate(words):
            start = int(word["start"])
            end = int(word["end"])
            word_length = max(0, end - start)
            stats["word_character_count"] += word_length
            coverage = coverage_by_position.get(
                word_position,
                {"token_indices": [], "covered_length": 0, "fully_covered": False},
            )
            token_indices = list(coverage.get("token_indices", []))
            _increment_distribution(distributions["word_subword_count"], len(token_indices))
            if len(token_indices) > int(max_subwords_per_word):
                stats["abnormally_many_subwords_word_count"] += 1
                add_issue(
                    "abnormally_many_subwords_for_word",
                    f"parser word maps to {len(token_indices)} subwords, threshold={max_subwords_per_word}: index={word['index']} text={word['text']!r}",
                    token_indices,
                )
            covered_length = int(coverage.get("covered_length", 0))
            fully_covered = bool(coverage.get("fully_covered", False))
            stats["covered_character_count"] += min(word_length, covered_length)
            if not token_indices:
                stats["unaligned_word_count"] += 1
            if not fully_covered:
                stats["character_coverage_incomplete_count"] += 1
                full_coverage = full_coverage_by_position.get(
                    word_position, {"fully_covered": False}
                )
                full_fully_covered = bool(full_coverage.get("fully_covered", False))
                if was_truncated and full_fully_covered and not fully_covered:
                    stats["truncation_uncovered_word_count"] += 1
                    add_issue(
                        "truncation_uncovered_word",
                        f"truncation left parser word uncovered: index={word['index']} text={word['text']!r}",
                        token_indices,
                    )

        for token_index in range(len(offsets)):
            token_to_words.setdefault(token_index, [])
        for token_index in range(len(offsets)):
            _increment_distribution(distributions["subword_parser_word_count"], len(token_to_words[token_index]))

        legal_shared_token_count = 0
        for token_index, word_positions in validation["token_to_word_positions"].items():
            if len(word_positions) <= 1:
                continue
            stats["subword_maps_to_multiple_parser_words_count"] += 1
            add_issue(
                "subword_maps_to_multiple_parser_words",
                f"subword maps to parser word positions={word_positions}: token_index={token_index}",
                [token_index],
            )
        for shared in validation["shared_subwords"]:
            token_index = int(shared["token_index"])
            word_positions = list(shared["word_positions"])
            if shared["legal"]:
                legal_shared_token_count += 1
                issue_type = (
                    "partial_contiguous_shared_subword"
                    if shared["partial_contiguous_shared_subword"]
                    else "legal_contiguous_shared_subword"
                )
                add_issue(
                    issue_type,
                    f"legal contiguous shared subword: token_index={token_index} word_positions={word_positions}",
                    [token_index],
                )

        stats["legal_contiguous_shared_subword_count"] = legal_shared_token_count
        stats["legal_contiguous_shared_row_count"] = int(legal_shared_token_count > 0)
        if was_truncated:
            stats["truncated_rows"] = 1

        suspicious_rows = []
        for issue_type in sorted(issue_tokens):
            suspicious_rows.append(
                _issue_record(
                    split,
                    str(row.get("id", "")),
                    text,
                    issue_type,
                    words,
                    tokenizer_tokens,
                    character_spans,
                    issue_tokens[issue_type],
                    "; ".join(issue_messages[issue_type]),
                )
            )
        return {
            "split": split,
            "row_id": str(row.get("id", "")),
            "text": text,
            "failed": False,
            "issue_types": sorted(issue_tokens),
            "sentence_count": sentence_count,
            "word_count": len(words),
            "subword_count": len(input_ids),
            "stats": stats,
            "distributions": distributions,
            "suspicious_rows": suspicious_rows,
            "legal_shared_token_count": legal_shared_token_count,
        }
    except Exception as exc:
        return _failed_row_result(split, row, exc)


def _dependency_for_row(dependency, index: int):
    if isinstance(dependency, (list, tuple)):
        return dependency[index]
    return dependency


def scan_split_rows(
    split: str,
    rows: list[dict],
    tokenizer,
    parser,
    max_source_length: int = DEFAULT_MAX_SOURCE_LENGTH,
    max_subwords_per_word: int = DEFAULT_MAX_SUBWORDS_PER_WORD,
) -> dict:
    """Scan all rows in a split; row failures are recorded and scanning continues."""
    report = _empty_split_report(len(rows))
    row_results = []
    suspicious_rows = []
    for index, row in enumerate(rows):
        result = scan_alignment_row(
            split,
            row,
            _dependency_for_row(tokenizer, index),
            _dependency_for_row(parser, index),
            max_source_length=max_source_length,
            max_subwords_per_word=max_subwords_per_word,
        )
        _merge_row_into_split(report, result)
        row_results.append(result)
        suspicious_rows.extend(result.get("suspicious_rows", []))
    report["row_results"] = row_results
    report["suspicious_rows"] = suspicious_rows
    return report


def load_preflight_rows(source_dataset: str, target_dataset: str) -> dict[str, list[dict]]:
    """Load only source train/dev and target train; target test is never requested."""
    return {
        "source_train": load_split(source_dataset, "train"),
        "source_dev": load_split(source_dataset, "dev"),
        "target_unlabeled": load_split(target_dataset, "train"),
    }


def _git_commit(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _code_identity(repo_dir: Path) -> dict:
    return {
        "files_sha256": {
            relative: sha256_file(repo_dir / relative) for relative in CODE_IDENTITY_FILES
        }
    }


def _input_identity(rows_by_split: dict[str, list[dict]]) -> dict:
    result = {}
    for split in SPLITS:
        rows = rows_by_split[split]
        paths = sorted({str(row.get("source_path", "")) for row in rows if row.get("source_path")})
        if len(paths) != 1:
            raise AlignmentPreflightError(f"{split} must resolve to exactly one input file")
        path = Path(paths[0])
        if not path.is_file():
            raise AlignmentPreflightError(f"missing {split} input file: {path}")
        result[split] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def _build_identity(
    repo_dir: Path,
    rows_by_split: dict[str, list[dict]],
    tokenizer_identity: dict,
    parser_identity: dict,
    max_source_length: int,
    max_subwords_per_word: int,
    runtime_devices: dict,
) -> dict:
    parser_actual = dict(parser_identity)
    parser_actual.setdefault("expected_sha256", dict(EXPECTED_PARSER_SHA256))
    return {
        "git_commit": _git_commit(repo_dir),
        "input_files": _input_identity(rows_by_split),
        "parser": parser_actual,
        "tokenizer": tokenizer_identity,
        "code": _code_identity(repo_dir),
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "alignment_policy_version": ALIGNMENT_POLICY_VERSION,
        "max_source_length": int(max_source_length),
        "max_subwords_per_word": int(max_subwords_per_word),
        "runtime_devices": dict(runtime_devices),
    }


def _runtime_device_identity(args) -> dict:
    requested = int(getattr(args, "cuda", 0))
    if requested < 0:
        raise AlignmentPreflightError("--cuda must be a non-negative device index")
    if bool(getattr(args, "cpu", False)):
        return {
            "requested_cuda_index": requested,
            "actual_cuda_index": None,
            "parser_device": "cpu",
            "model_device": "cpu",
        }
    try:
        import torch
    except ImportError as exc:
        raise AlignmentPreflightError("torch is required for non-CPU preflight") from exc
    if not torch.cuda.is_available():
        raise AlignmentPreflightError(
            f"requested CUDA device {requested}, but CUDA is unavailable"
        )
    device_count = int(torch.cuda.device_count())
    if requested >= device_count:
        raise AlignmentPreflightError(
            f"requested CUDA device {requested} is out of range; device_count={device_count}"
        )
    torch.cuda.set_device(requested)
    actual = int(torch.cuda.current_device())
    device_name = f"cuda:{actual}"
    return {
        "requested_cuda_index": requested,
        "actual_cuda_index": actual,
        "parser_device": device_name,
        "model_device": device_name,
    }


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AlignmentPreflightError(f"unreadable preflight summary: {path}") from exc


def _read_suspicious_count(path: Path) -> int:
    return len(_read_complete_suspicious_lines(path))


def _read_complete_suspicious_lines(path: Path) -> list[bytes]:
    required = {
        "split",
        "row_id",
        "text",
        "issue_type",
        "parser_words",
        "tokenizer_tokens",
        "character_spans",
        "shared_token_indices",
        "error_message",
    }
    complete_lines: list[bytes] = []
    try:
        raw_lines = path.read_bytes().splitlines(keepends=True)
        nonempty_line_numbers = [
            line_number
            for line_number, raw_line in enumerate(raw_lines, start=1)
            if raw_line.strip()
        ]
        last_nonempty_line = nonempty_line_numbers[-1] if nonempty_line_numbers else 0
        for line_number, raw_line in enumerate(raw_lines, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if line_number == last_nonempty_line:
                    break
                raise AlignmentPreflightError(
                    f"invalid suspicious row before tail at line {line_number}"
                ) from exc
            if not isinstance(record, dict) or not required.issubset(record):
                raise AlignmentPreflightError(f"invalid suspicious row at line {line_number}")
            complete_lines.append(raw_line)
    except OSError as exc:
        raise AlignmentPreflightError(f"unreadable suspicious row file: {path}") from exc
    return complete_lines


def _inspect_output_contract(output_dir: Path) -> tuple[tuple[Path, Path, Path], str]:
    """Inspect output state without creating directories or files."""
    output_dir = Path(output_dir)
    paths = tuple(output_dir / name for name in OUTPUT_FILES)
    if not output_dir.exists():
        return paths, "new"
    if not output_dir.is_dir():
        raise AlignmentPreflightError(f"output path is not a directory: {output_dir}")
    entries = {entry.name for entry in output_dir.iterdir()}
    allowed = set(OUTPUT_FILES) | {OUTPUT_FILES[0] + ".tmp"}
    unexpected = sorted(entries.difference(allowed))
    if unexpected:
        raise AlignmentPreflightError(
            f"output directory contains unexpected files; refusing to mix state: {unexpected}"
        )
    for path in paths:
        if path.exists() and not path.is_file():
            raise AlignmentPreflightError(f"preflight output path is not a file: {path}")
    summary_exists, suspicious_exists, markdown_exists = [path.exists() for path in paths]
    if summary_exists and suspicious_exists:
        return paths, "existing"
    if not summary_exists and suspicious_exists:
        return paths, "orphan_suspicious"
    if not summary_exists and not suspicious_exists and not markdown_exists:
        return paths, "new"
    raise AlignmentPreflightError("preflight output contract is incomplete; refusing resume")


def _ensure_output_contract(output_dir: Path) -> tuple[Path, Path, Path, bool]:
    """Backward-compatible read-only contract inspection.

    Unlike the old implementation, this helper never creates the directory or
    any output file.  Initialization is performed only after identity checks.
    """
    paths, state = _inspect_output_contract(output_dir)
    return paths[0], paths[1], paths[2], state == "existing"


def _initialize_output_contract(
    output_dir: Path,
    paths: tuple[Path, Path, Path],
    state: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = paths[0].with_name(paths[0].name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    if state == "orphan_suspicious":
        paths[1].write_bytes(b"")
    else:
        paths[1].touch()


def _reconcile_suspicious_log(path: Path, expected_count: int) -> None:
    complete_lines = _read_complete_suspicious_lines(path)
    if len(complete_lines) < int(expected_count):
        raise AlignmentPreflightError(
            f"resume suspicious rows are missing committed records: expected={expected_count} actual={len(complete_lines)}"
        )
    committed_bytes = b"".join(complete_lines[: int(expected_count)])
    if path.read_bytes() != committed_bytes:
        path.write_bytes(committed_bytes)


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# 全量词—子词对齐预检报告",
        "",
        f"更新时间：{datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M（北京时间）')}",
        "",
        f"总体状态：`{summary['status']}`",
        "",
        "本预检只读取 source_train、source_dev 和 target_unlabeled 的训练输入；不读取 target_test，不修改模型、训练、损失、标签或正式实验逻辑。",
        "",
        "## 数据划分扫描",
        "",
        "| 数据划分 | 句数 | 词数 | 子词数 | 成功行 | 失败行 | 完成行 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, report in summary["splits"].items():
        lines.append(
            f"| `{split}` | {report['sentence_count']} | {report['word_count']} | {report['subword_count']} | {report['successful_rows']} | {report['failed_rows']} | {report['completed_rows']}/{report['row_count']} |"
        )
    lines.extend(
        [
            "",
            "## 机器可读汇总",
            "",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _persist_progress(summary_path: Path, markdown_path: Path, summary: dict) -> None:
    _atomic_write_json(summary_path, summary)
    _write_markdown(markdown_path, summary)


def _validate_resume_state(
    summary: dict,
    identity: dict,
    rows_by_split: dict[str, list[dict]],
    suspicious_path: Path,
) -> dict[str, dict]:
    if summary.get("target_test_access") is not False:
        raise AlignmentPreflightError("existing summary does not explicitly forbid target test access")
    if summary.get("identity") != identity:
        raise AlignmentPreflightError("resume identity mismatch; old state must not be mixed")
    split_reports = {}
    for split in SPLITS:
        if split not in summary.get("splits", {}):
            raise AlignmentPreflightError(f"existing summary misses split: {split}")
        report = _normalize_split_report(summary["splits"][split], len(rows_by_split[split]))
        if report["row_count"] != len(rows_by_split[split]):
            raise AlignmentPreflightError(f"resume row count mismatch: split={split}")
        if report["completed_rows"] < 0 or report["completed_rows"] > report["row_count"]:
            raise AlignmentPreflightError(f"resume completed row count invalid: split={split}")
        split_reports[split] = report
    expected_suspicious = sum(report["suspicious_row_count"] for report in split_reports.values())
    _reconcile_suspicious_log(suspicious_path, expected_suspicious)
    actual_suspicious = _read_suspicious_count(suspicious_path)
    if actual_suspicious != expected_suspicious:
        raise AlignmentPreflightError(
            f"resume suspicious row count mismatch: expected={expected_suspicious} actual={actual_suspicious}"
        )
    return split_reports


def run_preflight(
    args,
    *,
    rows_by_split: dict[str, list[dict]] | None = None,
    tokenizer=None,
    parser=None,
    tokenizer_identity: dict | None = None,
    parser_identity: dict | None = None,
    repo_dir: str | Path | None = None,
    stop_after_rows: int | None = None,
) -> dict:
    """Run or resume the read-only preflight over all three required splits."""
    output_dir = Path(args.output_dir)
    repo_dir = Path(repo_dir or Path(__file__).resolve().parent)
    max_source_length = int(getattr(args, "max_source_length", DEFAULT_MAX_SOURCE_LENGTH))
    max_subwords_per_word = int(
        getattr(args, "max_subwords_per_word", DEFAULT_MAX_SUBWORDS_PER_WORD)
    )
    if max_source_length <= 0 or max_subwords_per_word <= 0:
        raise AlignmentPreflightError("max_source_length and max_subwords_per_word must be positive")
    rows_by_split = rows_by_split or load_preflight_rows(args.source_dataset, args.target_dataset)
    if set(rows_by_split) != set(SPLITS):
        raise AlignmentPreflightError(f"preflight splits must be exactly {list(SPLITS)}")
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer_identity = tokenizer_identity or build_tokenizer_identity(args.model_path, tokenizer)
    parser_identity = parser_identity or build_parser_identity(args.parser_dir)
    runtime_devices = _runtime_device_identity(args)
    identity = _build_identity(
        repo_dir,
        rows_by_split,
        tokenizer_identity,
        parser_identity,
        max_source_length,
        max_subwords_per_word,
        runtime_devices,
    )
    if parser is None:
        parser = build_stanza_pipeline(args.parser_dir, use_gpu=not bool(getattr(args, "cpu", False)))

    paths, output_state = _inspect_output_contract(output_dir)
    summary_path, suspicious_path, markdown_path = paths
    temporary = summary_path.with_name(summary_path.name + ".tmp")

    if output_state == "existing":
        split_reports = _validate_resume_state(
            _read_json(summary_path), identity, rows_by_split, suspicious_path
        )
        if temporary.exists():
            temporary.unlink()
    else:
        _initialize_output_contract(output_dir, paths, output_state)
        split_reports = {
            split: _empty_split_report(len(rows_by_split[split])) for split in SPLITS
        }
        initial_summary = build_preflight_summary(
            split_reports, identity, max_source_length, runtime_devices
        )
        initial_summary["status"] = "RUNNING"
        initial_summary["alignment_strategy"]["max_subwords_per_word"] = max_subwords_per_word
        _persist_progress(summary_path, markdown_path, initial_summary)

    all_completed = all(report["completed_rows"] == report["row_count"] for report in split_reports.values())
    if all_completed:
        final_summary = build_preflight_summary(
            split_reports, identity, max_source_length, runtime_devices
        )
        final_summary["alignment_strategy"]["max_subwords_per_word"] = max_subwords_per_word
        _persist_progress(summary_path, markdown_path, final_summary)
        return final_summary

    processed = sum(report["completed_rows"] for report in split_reports.values())
    with suspicious_path.open("a", encoding="utf-8", newline="\n") as suspicious_handle:
        from tqdm import tqdm

        for split in SPLITS:
            report = split_reports[split]
            start_index = report["completed_rows"]
            if start_index == report["row_count"]:
                continue
            for index in tqdm(
                range(start_index, report["row_count"]),
                initial=start_index,
                total=report["row_count"],
                desc=f"alignment-preflight:{split}",
            ):
                row = rows_by_split[split][index]
                result = scan_alignment_row(
                    split,
                    row,
                    tokenizer,
                    parser,
                    max_source_length=max_source_length,
                    max_subwords_per_word=max_subwords_per_word,
                )
                _merge_row_into_split(report, result)
                for suspicious_row in result.get("suspicious_rows", []):
                    suspicious_handle.write(
                        json.dumps(suspicious_row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                suspicious_handle.flush()
                os.fsync(suspicious_handle.fileno())
                processed += 1
                progress_summary = build_preflight_summary(
                    split_reports, identity, max_source_length, runtime_devices
                )
                progress_summary["status"] = "RUNNING"
                progress_summary["alignment_strategy"]["max_subwords_per_word"] = max_subwords_per_word
                _persist_progress(summary_path, markdown_path, progress_summary)
                if stop_after_rows is not None and processed >= int(stop_after_rows):
                    raise AlignmentPreflightInterrupted(
                        f"preflight interrupted after {processed} rows"
                    )
    final_summary = build_preflight_summary(
        split_reports, identity, max_source_length, runtime_devices
    )
    final_summary["alignment_strategy"]["max_subwords_per_word"] = max_subwords_per_word
    _persist_progress(summary_path, markdown_path, final_summary)
    return final_summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M1 全量词—子词对齐只读预检")
    parser.add_argument("--source_dataset", default="laptop14", choices=sorted(DATASETS))
    parser.add_argument("--target_dataset", default="rest15", choices=sorted(DATASETS))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--parser_dir", default=str(DEFAULT_PARSER_DIR))
    parser.add_argument("--max_source_length", type=int, default=DEFAULT_MAX_SOURCE_LENGTH)
    parser.add_argument("--max_subwords_per_word", type=int, default=DEFAULT_MAX_SUBWORDS_PER_WORD)
    parser.add_argument("--cuda", type=int, default=0, help="请求的 CUDA 设备编号")
    parser.add_argument("--cpu", action="store_true", help="仅用于 CPU 测试；正式命令不使用此开关")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    try:
        summary = run_preflight(args)
    except AlignmentPreflightInterrupted as exc:
        print(f"alignment preflight interrupted: {exc}", file=sys.stderr)
        raise SystemExit(3)
    except Exception as exc:
        print(f"alignment preflight blocked: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
