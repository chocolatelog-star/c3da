from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable


GRAPH_SCHEMA_VERSION = 1
DEFAULT_PARSER_DIR = Path(r"J:\nlp\models\stanza_resources")
PARSER_PROCESSORS = "tokenize,mwt,pos,lemma,depparse"
PARSER_PACKAGES = {
    "tokenize": "ewt",
    "mwt": "ewt",
    "pos": "ewt_charlm",
    "lemma": "combined_nocharlm",
    "depparse": "ewt_charlm",
}
EXPECTED_PARSER_SHA256 = {
    "resources.json": "4e41c1df152146fa26ed0c006a08feea7a60bb3414bb6d57dbda24ad2e3cb99c",
    "en/tokenize/ewt.pt": "fc2fed0cd74dbaef1620bd3e776141ae76c4e28eb5aeff369b2715c31cc73cba",
    "en/mwt/ewt.pt": "73411a30da7638bbda2ebd9490e017d78feb4e029e90c9f5c9f37e5433292eb0",
    "en/pos/ewt_charlm.pt": "f89696d286c29aff173061fbd4b581c73525257ce38015804be047a5e40f9614",
    "en/lemma/combined_nocharlm.pt": "e3cb21e3c97a514d102fcc95e78fbc2ab838bc7b306a48029022f35caba1aa2c",
    "en/depparse/ewt_charlm.pt": "7386666c2054363f6c4eae702f84ef7d4a11aa4708c2907b82b105e56925d897",
}


class GraphCacheError(RuntimeError):
    """Raised when a graph cache cannot be trusted for the requested input."""


def normalize_graph_text(text: str) -> str:
    return " ".join(str(text).split())


def normalized_text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_graph_text(text).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_row_json(row: dict) -> str:
    return json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _stable_index(value: str, values: Iterable[str]) -> int:
    ordered = sorted(set(values))
    try:
        return ordered.index(value)
    except ValueError as exc:
        raise GraphCacheError(f"unknown graph vocabulary item: {value}") from exc


def _dependency_key(edge: dict) -> str:
    return str(edge.get("dependency_key", "self"))


def _pos_key(edge: dict) -> str:
    return str(edge.get("pos_pair_key", "self"))


def _relation_key(edge: dict) -> str:
    return str(edge["relation_key"])


def _assign_edge_ids(edges: list[dict], relation_vocab: list[str] | None = None) -> list[dict]:
    relation_values = relation_vocab or [_relation_key(edge) for edge in edges]
    dependency_values = [_dependency_key(edge) for edge in edges]
    pos_values = [_pos_key(edge) for edge in edges]
    assigned = []
    for edge in edges:
        current = dict(edge)
        current["relation_id"] = _stable_index(_relation_key(current), relation_values)
        current["dependency_relation_id"] = _stable_index(_dependency_key(current), dependency_values)
        current["pos_pair_id"] = _stable_index(_pos_key(current), pos_values)
        assigned.append(current)
    return assigned


def build_typed_edges(words: list[dict], relation_vocab: list[str] | None = None) -> list[dict]:
    """Build only dependency, POS-neighbor and self-loop edges."""

    edges: list[dict] = []

    def append_edge(src: int, dst: int, kind: str, dependency_key: str, pos_pair_key: str) -> None:
        source = words[src]
        target = words[dst]
        relation_key = f"{kind}|{dependency_key}|{source['upos']}|{target['upos']}"
        edges.append(
            {
                "src": int(src),
                "dst": int(dst),
                "kind": kind,
                "relation_key": relation_key,
                "dependency_key": dependency_key,
                "pos_pair_key": pos_pair_key,
            }
        )

    for word in words:
        index = int(word["index"])
        head = int(word.get("head", -1))
        if head >= 0:
            if head >= len(words):
                raise GraphCacheError(f"dependency head out of range: {head}")
            dependency_key = f"{word.get('deprel', 'dep')}|{words[head]['upos']}|{word['upos']}"
            pos_pair_key = f"{words[head]['upos']}|{word['upos']}"
            append_edge(head, index, "dependency_forward", dependency_key, pos_pair_key)
            reverse_dependency_key = f"reverse:{word.get('deprel', 'dep')}|{word['upos']}|{words[head]['upos']}"
            reverse_pos_pair_key = f"{word['upos']}|{words[head]['upos']}"
            append_edge(index, head, "dependency_reverse", reverse_dependency_key, reverse_pos_pair_key)

    for left, right in zip(words, words[1:]):
        if int(left.get("sentence_index", 0)) != int(right.get("sentence_index", 0)):
            continue
        left_index = int(left["index"])
        right_index = int(right["index"])
        append_edge(
            left_index,
            right_index,
            "pos_neighbor_forward",
            "pos_neighbor",
            f"{left['upos']}|{right['upos']}",
        )
        append_edge(
            right_index,
            left_index,
            "pos_neighbor_reverse",
            "pos_neighbor",
            f"{right['upos']}|{left['upos']}",
        )

    for word in words:
        index = int(word["index"])
        append_edge(index, index, "self_loop", "self", f"{word['upos']}|{word['upos']}")

    return _assign_edge_ids(edges, relation_vocab=relation_vocab)


def _find_graph_text_start(input_text: str, graph_text: str) -> int:
    if input_text == graph_text:
        return 0
    if input_text.endswith(graph_text):
        return len(input_text) - len(graph_text)
    matches = [match.start() for match in re.finditer(re.escape(graph_text), input_text)]
    if len(matches) == 1:
        return matches[0]
    raise GraphCacheError("raw graph text has no unique exact span in the model input")


def align_parser_words_to_subwords(
    words: list[dict],
    input_text: str,
    graph_text: str,
    offset_mapping: list[tuple[int, int]],
) -> list[list[int]]:
    graph_start = _find_graph_text_start(input_text, graph_text)
    aligned: list[list[int]] = []
    used_tokens: set[int] = set()
    for word in words:
        start = int(word["start"])
        end = int(word["end"])
        token_indices = []
        for token_index, (token_start, token_end) in enumerate(offset_mapping):
            token_start = int(token_start) - graph_start
            token_end = int(token_end) - graph_start
            if token_end <= token_start:
                continue
            if token_start >= start and token_end <= end:
                token_indices.append(token_index)
        if not token_indices:
            raise GraphCacheError(
                f"parser word cannot be aligned to subwords: index={word['index']} text={word['text']!r}"
            )
        overlap = used_tokens.intersection(token_indices)
        if overlap:
            raise GraphCacheError(f"subword token is assigned to multiple parser words: {sorted(overlap)}")
        used_tokens.update(token_indices)
        aligned.append(token_indices)
    return aligned


def _parser_words_from_doc(doc) -> list[dict]:
    words: list[dict] = []
    for sentence_index, sentence in enumerate(getattr(doc, "sentences", [])):
        sentence_offset = len(words)
        sentence_words = list(getattr(sentence, "words", []))
        for local_index, word in enumerate(sentence_words):
            start = getattr(word, "start_char", None)
            end = getattr(word, "end_char", None)
            if start is None or end is None:
                raise GraphCacheError(f"parser word has no character span: {getattr(word, 'text', '')!r}")
            head = int(getattr(word, "head", 0) or 0)
            global_head = sentence_offset + head - 1 if head > 0 else -1
            words.append(
                {
                    "index": sentence_offset + local_index,
                    "sentence_index": sentence_index,
                    "text": str(getattr(word, "text", "")),
                    "start": int(start),
                    "end": int(end),
                    "upos": str(getattr(word, "upos", "X") or "X"),
                    "head": global_head,
                    "deprel": str(getattr(word, "deprel", "dep") or "dep"),
                }
            )
    if not words:
        raise GraphCacheError("parser returned no words")
    return words


def build_graph_record(
    row_id: str | int,
    text: str,
    input_text: str,
    parser,
    tokenizer,
    tokenizer_identity: dict,
    parser_identity: dict,
    max_length: int = 128,
    relation_vocab: list[str] | None = None,
) -> dict:
    encoded = tokenizer(
        input_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = [tuple(pair) for pair in encoded.get("offset_mapping", [])]
    if not offsets:
        raise GraphCacheError("tokenizer did not return offset_mapping")
    words = _parser_words_from_doc(parser(text))
    word_to_subword = align_parser_words_to_subwords(words, input_text, text, offsets)
    edges = build_typed_edges(words, relation_vocab=relation_vocab)
    return {
        "row_id": str(row_id),
        "text": str(text),
        "normalized_text_sha256": normalized_text_sha256(text),
        "input_text_sha256": hashlib.sha256(str(input_text).encode("utf-8")).hexdigest(),
        "input_length": len(encoded.get("input_ids", [])),
        "parser_tokens": words,
        "word_to_subword": word_to_subword,
        "edges": edges,
        "parser_identity": parser_identity,
        "tokenizer_identity": tokenizer_identity,
        "alignment": {
            "valid_word_count": len(words),
            "unaligned_word_count": 0,
            "mapped_subword_count": sum(len(indices) for indices in word_to_subword),
        },
    }


def apply_relation_vocab(records: list[dict]) -> list[str]:
    relation_vocab = sorted(
        {
            edge["relation_key"]
            for record in records
            for edge in record["edges"]
        }
    )
    dependency_vocab = sorted(
        {
            edge.get("dependency_key", "self")
            for record in records
            for edge in record["edges"]
        }
    )
    pos_vocab = sorted(
        {
            edge.get("pos_pair_key", "self")
            for record in records
            for edge in record["edges"]
        }
    )
    for record in records:
        record["edges"] = _assign_edge_ids(record["edges"], relation_vocab)
        for edge in record["edges"]:
            edge["dependency_relation_id"] = _stable_index(edge["dependency_key"], dependency_vocab)
            edge["pos_pair_id"] = _stable_index(edge["pos_pair_key"], pos_vocab)
    return relation_vocab


class GraphCache:
    def __init__(
        self,
        records: list[dict],
        relation_vocab: list[str],
        split: str,
        use_task_prefix: bool = True,
    ):
        self.records = records
        self.relation_vocab = relation_vocab
        self.split = split
        self.use_task_prefix = bool(use_task_prefix)
        self._by_id = {str(record["row_id"]): record for record in records}
        if len(self._by_id) != len(records):
            raise GraphCacheError(f"duplicate row_id in {split} graph cache")

    @property
    def relation_vocab_size(self) -> int:
        return max(1, len(self.relation_vocab))

    def get(self, row: dict) -> dict:
        row_id = str(row.get("id", ""))
        if not row_id:
            raise GraphCacheError(f"{self.split} row has no id")
        record = self._by_id.get(row_id)
        if record is None:
            raise GraphCacheError(f"missing graph cache row: split={self.split} row_id={row_id}")
        expected_hash = normalized_text_sha256(row.get("text", ""))
        if record.get("normalized_text_sha256") != expected_hash:
            raise GraphCacheError(f"text hash mismatch: split={self.split} row_id={row_id}")
        input_text = str(
            row.get(
                "input",
                f"extract aste: {row.get('text', '')}" if self.use_task_prefix else row.get("text", ""),
            )
        )
        expected_input_hash = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
        if record.get("input_text_sha256") != expected_input_hash:
            raise GraphCacheError(f"input hash mismatch: split={self.split} row_id={row_id}")
        return {
            "word_to_subword": [list(indices) for indices in record["word_to_subword"]],
            "word_mask": [1] * len(record["word_to_subword"]),
            "edge_src": [int(edge["src"]) for edge in record["edges"]],
            "edge_dst": [int(edge["dst"]) for edge in record["edges"]],
            "relation_id": [int(edge["relation_id"]) for edge in record["edges"]],
            "dependency_relation_id": [int(edge["dependency_relation_id"]) for edge in record["edges"]],
            "pos_pair_id": [int(edge["pos_pair_id"]) for edge in record["edges"]],
            "edge_mask": [1] * len(record["edges"]),
        }


class CompositeGraphCache:
    """Route graph rows from source and target-unlabeled split caches."""

    def __init__(self, caches: dict[str, GraphCache]):
        required = {"source_train", "target_unlabeled"}
        if not required.issubset(caches):
            raise GraphCacheError(f"composite cache requires splits: {sorted(required)}")
        vocabularies = {tuple(cache.relation_vocab) for cache in caches.values()}
        if len(vocabularies) != 1:
            raise GraphCacheError("relation vocabulary mismatch in composite graph cache")
        self.caches = dict(caches)
        self.relation_vocab = list(next(iter(vocabularies)))
        self.relation_vocab_size = max(1, len(self.relation_vocab))

    def get(self, row: dict) -> dict:
        split = "target_unlabeled" if row.get("augmentation") == "target_unlabeled" else "source_train"
        return self.caches[split].get(row)


def load_graph_cache_rows(
    cache_path: str | Path,
    expected_rows: list[dict],
    expected_split: str,
) -> list[dict]:
    path = Path(cache_path)
    if not path.is_file():
        raise GraphCacheError(f"missing graph cache: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphCacheError(f"unreadable graph cache: {path}") from exc
    by_id = {str(row.get("row_id", "")): row for row in rows}
    if len(by_id) != len(rows):
        raise GraphCacheError(f"duplicate row_id in {expected_split} graph cache")
    expected_ids = [str(row.get("id", "")) for row in expected_rows]
    if [str(row.get("row_id", "")) for row in rows] != expected_ids:
        raise GraphCacheError(f"row order mismatch: split={expected_split}")
    for expected in expected_rows:
        row_id = str(expected.get("id", ""))
        cached = by_id[row_id]
        if cached.get("normalized_text_sha256") != normalized_text_sha256(expected.get("text", "")):
            raise GraphCacheError(f"text hash mismatch: split={expected_split} row_id={row_id}")
    return rows


def load_graph_cache_directory(
    cache_dir: str | Path,
    split: str,
    expected_rows: list[dict],
    tokenizer_identity: dict | None = None,
    parser_identity: dict | None = None,
) -> GraphCache:
    root = Path(cache_dir)
    manifest_path = root / "manifest.json"
    vocab_path = root / "relation_vocab.json"
    if not manifest_path.is_file() or not vocab_path.is_file():
        raise GraphCacheError(f"graph cache manifest or vocabulary missing: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        relation_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphCacheError(f"unreadable graph cache metadata: {root}") from exc
    if manifest.get("schema_version") != GRAPH_SCHEMA_VERSION:
        raise GraphCacheError("graph cache schema version mismatch")
    if manifest.get("target_test_access") is not False:
        raise GraphCacheError("graph cache manifest must explicitly forbid target test access")
    if manifest.get("relation_vocab_sha256") != sha256_file(vocab_path):
        raise GraphCacheError("graph relation vocabulary SHA256 mismatch")
    if split not in {"source_train", "source_dev", "target_unlabeled"}:
        raise GraphCacheError(f"target_test graph cache is forbidden: {split}")
    if tokenizer_identity is not None and manifest.get("tokenizer_identity") != tokenizer_identity:
        raise GraphCacheError(f"tokenizer identity mismatch in cache manifest: split={split}")
    if parser_identity is not None and manifest.get("parser_identity") != parser_identity:
        raise GraphCacheError(f"parser identity mismatch in cache manifest: split={split}")
    use_task_prefix = bool(manifest.get("use_task_prefix", True))
    expected_input_sha256 = _split_input_identity(expected_rows, use_task_prefix)
    if manifest.get("input_sha256", {}).get(split) != expected_input_sha256:
        raise GraphCacheError(f"input identity mismatch in cache manifest: split={split}")
    rows = load_graph_cache_rows(root / f"{split}.jsonl", expected_rows, split)
    if int(manifest.get("splits", {}).get(split, -1)) != len(rows):
        raise GraphCacheError(f"cache row count mismatch: split={split}")
    for record in rows:
        if tokenizer_identity is not None and record.get("tokenizer_identity") != tokenizer_identity:
            raise GraphCacheError(f"tokenizer identity mismatch: split={split} row_id={record['row_id']}")
        if parser_identity is not None and record.get("parser_identity") != parser_identity:
            raise GraphCacheError(f"parser identity mismatch: split={split} row_id={record['row_id']}")
        if record.get("normalized_text_sha256") != normalized_text_sha256(
            next(row["text"] for row in expected_rows if str(row.get("id")) == str(record["row_id"]))
        ):
            raise GraphCacheError(f"text identity mismatch: split={split} row_id={record['row_id']}")
    return GraphCache(
        rows,
        list(relation_vocab),
        split,
        use_task_prefix=use_task_prefix,
    )


def build_parser_identity(parser_dir: str | Path = DEFAULT_PARSER_DIR) -> dict:
    parser_dir = Path(parser_dir)
    if not parser_dir.is_dir():
        raise GraphCacheError(f"missing Stanza resource directory: {parser_dir}")
    actual_hashes = {}
    for relative, expected in EXPECTED_PARSER_SHA256.items():
        path = parser_dir / relative
        if not path.is_file():
            raise GraphCacheError(f"missing Stanza model file: {path}")
        actual = sha256_file(path)
        actual_hashes[relative] = actual
        if actual.lower() != expected.lower():
            raise GraphCacheError(f"Stanza model SHA256 mismatch: {relative}")
    try:
        import stanza
    except ImportError as exc:
        raise GraphCacheError("stanza is required to build graph caches") from exc
    return {
        "language": "en",
        "processors": PARSER_PROCESSORS,
        "packages": dict(PARSER_PACKAGES),
        "resource_dir": str(parser_dir),
        "stanza_version": str(getattr(stanza, "__version__", "unknown")),
        "sha256": actual_hashes,
    }


def build_tokenizer_identity(model_path: str | Path, tokenizer) -> dict:
    model_path = Path(model_path)
    files = {}
    for name in ("spiece.model", "tokenizer.json"):
        path = model_path / name
        if not path.is_file():
            raise GraphCacheError(f"missing tokenizer file: {path}")
        files[name] = sha256_file(path)
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise GraphCacheError("formal tokenizer must provide deterministic fast offset mapping")
    return {
        "class": type(tokenizer).__name__,
        "is_fast": True,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "files_sha256": files,
    }


def build_stanza_pipeline(parser_dir: str | Path, use_gpu: bool):
    try:
        import stanza
    except ImportError as exc:
        raise GraphCacheError("stanza is required to build graph caches") from exc
    return stanza.Pipeline(
        "en",
        processors=PARSER_PROCESSORS,
        package=dict(PARSER_PACKAGES),
        dir=str(parser_dir),
        download_method=None,
        use_gpu=bool(use_gpu),
    )


def _read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _split_input_identity(rows: list[dict], use_task_prefix: bool) -> str:
    identities = []
    for row in rows:
        text = str(row.get("text", ""))
        input_text = str(row.get("input") or (f"extract aste: {text}" if use_task_prefix else text))
        identities.append(
            {
                "id": str(row.get("id", "")),
                "text": text,
                "input": input_text,
            }
        )
    payload = "\n".join(canonical_row_json(item) for item in identities).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_progress(path: Path, state: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_resumable_records(
    split: str,
    rows: list[dict],
    output_dir: Path,
    tokenizer: object,
    parser: object,
    tokenizer_identity: dict,
    parser_identity: dict,
    use_task_prefix: bool,
    max_length: int,
    processed_before: int,
    stop_after_rows: int | None,
) -> tuple[list[dict], int]:
    from tqdm import tqdm

    partial_path = output_dir / f"{split}.partial.jsonl"
    progress_path = output_dir / f"{split}.progress.json"
    input_identity = _split_input_identity(rows, use_task_prefix)
    state_identity = {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "split": split,
        "row_count": len(rows),
        "input_sha256": input_identity,
        "tokenizer_identity": tokenizer_identity,
        "parser_identity": parser_identity,
        "use_task_prefix": bool(use_task_prefix),
        "max_length": int(max_length),
    }
    has_partial = partial_path.exists()
    has_progress = progress_path.exists()
    if has_partial != has_progress:
        raise GraphCacheError(f"incomplete resume metadata for graph cache split={split}")
    records: list[dict] = []
    if has_partial:
        try:
            state = json.loads(progress_path.read_text(encoding="utf-8"))
            records = [
                json.loads(line)
                for line in partial_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise GraphCacheError(f"unreadable resume state for split={split}") from exc
        for key, expected in state_identity.items():
            if state.get(key) != expected:
                raise GraphCacheError(f"resume identity mismatch: split={split} key={key}")
        if int(state.get("completed_rows", -1)) != len(records):
            raise GraphCacheError(f"resume row count mismatch: split={split}")
        if len(records) > len(rows):
            raise GraphCacheError(f"resume cache has too many rows: split={split}")
        for index, record in enumerate(records):
            row = rows[index]
            expected_input = str(row.get("input") or (f"extract aste: {row['text']}" if use_task_prefix else row["text"]))
            if str(record.get("row_id")) != str(row.get("id")):
                raise GraphCacheError(f"resume row identity mismatch: split={split} index={index}")
            if record.get("normalized_text_sha256") != normalized_text_sha256(row.get("text", "")):
                raise GraphCacheError(f"resume text identity mismatch: split={split} index={index}")
            if record.get("input_text_sha256") != hashlib.sha256(expected_input.encode("utf-8")).hexdigest():
                raise GraphCacheError(f"resume input identity mismatch: split={split} index={index}")
    if len(records) == len(rows):
        return records, processed_before + len(records)

    with partial_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index in tqdm(
            range(len(records), len(rows)),
            initial=len(records),
            total=len(rows),
            desc=f"graph-cache:{split}",
        ):
            row = rows[index]
            input_text = str(row.get("input") or (f"extract aste: {row['text']}" if use_task_prefix else row["text"]))
            record = build_graph_record(
                row_id=row.get("id"),
                text=row.get("text", ""),
                input_text=input_text,
                parser=parser,
                tokenizer=tokenizer,
                tokenizer_identity=tokenizer_identity,
                parser_identity=parser_identity,
                max_length=max_length,
            )
            handle.write(canonical_row_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            records.append(record)
            _write_progress(
                progress_path,
                {
                    **state_identity,
                    "completed_rows": len(records),
                },
            )
            processed = processed_before + len(records)
            if stop_after_rows is not None and processed >= stop_after_rows:
                raise GraphCacheError(
                    f"interrupted after {processed} rows while building split={split}"
                )
    return records, processed_before + len(records)


def _graph_cache_stats(records_by_split: dict[str, list[dict]]) -> dict:
    records = [record for split in records_by_split.values() for record in split]
    return {
        "parse_coverage": {split: 1.0 for split in records_by_split},
        "alignment_coverage": {split: 1.0 for split in records_by_split},
        "failed_rows": {split: 0 for split in records_by_split},
        "canonical_row_sha256": {
            split: [sha256_bytes(canonical_row_json(record).encode("utf-8")) for record in split_records]
            for split, split_records in records_by_split.items()
        },
        "row_count": len(records),
    }


def build_graph_cache_records(
    split_rows: dict[str, list[dict]],
    output_dir: str | Path,
    tokenizer: object,
    parser: object,
    tokenizer_identity: dict,
    parser_identity: dict,
    use_task_prefix: bool = True,
    max_length: int = 128,
    stop_after_rows: int | None = None,
) -> dict:
    required_splits = {"source_train", "source_dev", "target_unlabeled"}
    if set(split_rows) != required_splits:
        raise GraphCacheError(f"graph cache splits must be exactly {sorted(required_splits)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_split: dict[str, list[dict]] = {}
    processed = 0
    for split in ("source_train", "source_dev", "target_unlabeled"):
        records, processed = _load_resumable_records(
            split,
            split_rows[split],
            output_dir,
            tokenizer,
            parser,
            tokenizer_identity,
            parser_identity,
            use_task_prefix,
            max_length,
            processed,
            stop_after_rows,
        )
        records_by_split[split] = records
    all_records = [record for records in records_by_split.values() for record in records]
    relation_vocab = apply_relation_vocab(all_records)
    (output_dir / "relation_vocab.json").write_text(
        json.dumps(relation_vocab, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    for split, records in records_by_split.items():
        (output_dir / f"{split}.jsonl").write_text(
            "".join(canonical_row_json(record) + "\n" for record in records),
            encoding="utf-8",
        )
    stats = _graph_cache_stats(records_by_split)
    manifest = {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "splits": {split: len(rows) for split, rows in split_rows.items()},
        "input_sha256": {split: _split_input_identity(rows, use_task_prefix) for split, rows in split_rows.items()},
        "parser_identity": parser_identity,
        "tokenizer_identity": tokenizer_identity,
        "relation_vocab_sha256": sha256_file(output_dir / "relation_vocab.json"),
        "relation_vocab_size": len(relation_vocab),
        "use_task_prefix": bool(use_task_prefix),
        "max_length": int(max_length),
        "target_test_access": False,
        "resume": {
            "enabled": True,
            "partial_files": {split: f"{split}.partial.jsonl" for split in split_rows},
            "progress_files": {split: f"{split}.progress.json" for split in split_rows},
        },
        "stats": stats,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_graph_cache(
    source_train_rows: list[dict],
    source_dev_rows: list[dict],
    target_unlabeled_rows: list[dict],
    output_dir: str | Path,
    model_path: str | Path,
    parser_dir: str | Path = DEFAULT_PARSER_DIR,
    use_task_prefix: bool = True,
    max_length: int = 128,
    use_gpu: bool = True,
) -> dict:
    from transformers import AutoTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer_identity = build_tokenizer_identity(model_path, tokenizer)
    parser_identity = build_parser_identity(parser_dir)
    parser = build_stanza_pipeline(parser_dir, use_gpu=use_gpu)
    split_rows = {
        "source_train": source_train_rows,
        "source_dev": source_dev_rows,
        "target_unlabeled": target_unlabeled_rows,
    }
    return build_graph_cache_records(
        split_rows,
        output_dir,
        tokenizer,
        parser,
        tokenizer_identity,
        parser_identity,
        use_task_prefix=use_task_prefix,
        max_length=max_length,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic Stanza EWT graph caches")
    parser.add_argument("build-cache", nargs="?")
    parser.add_argument("--source_train_file", required=True)
    parser.add_argument("--source_dev_file", required=True)
    parser.add_argument("--target_unlabeled_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default=r"J:\nlp\models\t5-base-py")
    parser.add_argument("--parser_dir", default=str(DEFAULT_PARSER_DIR))
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--no_task_prefix", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    build_graph_cache(
        source_train_rows=_read_jsonl(args.source_train_file),
        source_dev_rows=_read_jsonl(args.source_dev_file),
        target_unlabeled_rows=_read_jsonl(args.target_unlabeled_file),
        output_dir=args.output_dir,
        model_path=args.model_path,
        parser_dir=args.parser_dir,
        use_task_prefix=not args.no_task_prefix,
        max_length=args.max_length,
        use_gpu=not args.cpu,
    )


if __name__ == "__main__":
    main()
