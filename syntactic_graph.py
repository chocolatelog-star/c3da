from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable


GRAPH_SCHEMA_VERSION = 2
COMPOSITIONAL_RELATION_SCHEMA_VERSION = 1
ALIGNMENT_POLICY_VERSION = "overlap-contiguous-contained-sharing-v3"
DEFAULT_PARSER_DIR = Path(r"J:\nlp\models\stanza_resources")
PARSER_PROCESSORS = "tokenize,mwt,pos,lemma,depparse"
PARSER_PACKAGES = {
    "tokenize": "ewt",
    "mwt": "ewt",
    "pos": "ewt_charlm",
    "lemma": "combined_nocharlm",
    "depparse": "ewt_charlm",
}
UNK_RELATION_KEY = "__UNK_REL__"
UNK_DEP = "__UNK_DEP__"
UNK_POS = "__UNK_POS__"
GRAPH_DIRECTIONS = ("forward", "reverse", "self")
GRAPH_DEPENDENCY_VOCAB = (
    "self", "pos_neighbor", "acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp", "clf", "compound", "conj", "cop", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark", "nmod", "nsubj", "nummod", "obj", "obl", "orphan", "parataxis", "punct", "reparandum", "root", "vocative", "xcomp", UNK_DEP,
)
GRAPH_POS_VOCAB = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", UNK_POS,
)
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


def _compositional_parts(edge: dict) -> tuple[str, str, str, str]:
    """Return deterministic dependency, direction and endpoint POS categories."""
    kind = str(edge.get("kind", ""))
    direction = "reverse" if "reverse" in kind else ("self" if "self_loop" in kind else "forward")
    dep = str(edge.get("deprel", edge.get("dependency_key", "dep"))).split("|")[0]
    if dep.startswith("reverse:"):
        dep = dep.removeprefix("reverse:")
    dep = dep if dep in GRAPH_DEPENDENCY_VOCAB else ("pos_neighbor" if dep == "pos_neighbor" else UNK_DEP)
    pos_pair = str(edge.get("pos_pair_key", "" )).split("|")
    src_pos = pos_pair[0] if pos_pair and pos_pair[0] in GRAPH_POS_VOCAB else UNK_POS
    dst_pos = pos_pair[1] if len(pos_pair) > 1 and pos_pair[1] in GRAPH_POS_VOCAB else UNK_POS
    return dep, direction, src_pos, dst_pos


def _assign_edge_ids(edges: list[dict], relation_vocab: list[str] | None = None) -> list[dict]:
    relation_values = relation_vocab or sorted(set([_relation_key(edge) for edge in edges] + [UNK_RELATION_KEY]))
    relation_ids = {value: index for index, value in enumerate(relation_values)}
    unknown_id = relation_ids.get(UNK_RELATION_KEY)
    dependency_values = [_dependency_key(edge) for edge in edges]
    pos_values = [_pos_key(edge) for edge in edges]
    assigned = []
    for edge in edges:
        current = dict(edge)
        relation_key = _relation_key(current)
        if relation_key not in relation_ids:
            if unknown_id is None:
                raise GraphCacheError(f"unknown graph vocabulary item: {relation_key}")
            current["relation_id"] = unknown_id
        else:
            current["relation_id"] = relation_ids[relation_key]
        current["dependency_relation_id"] = _stable_index(_dependency_key(current), dependency_values)
        current["pos_pair_id"] = _stable_index(_pos_key(current), pos_values)
        dep, direction, src_pos, dst_pos = _compositional_parts(current)
        current["compositional_dependency"] = dep
        current["compositional_direction"] = direction
        current["compositional_src_pos"] = src_pos
        current["compositional_dst_pos"] = dst_pos
        current["compositional_dependency_id"] = list(GRAPH_DEPENDENCY_VOCAB).index(dep)
        current["compositional_direction_id"] = GRAPH_DIRECTIONS.index(direction)
        current["compositional_src_pos_id"] = list(GRAPH_POS_VOCAB).index(src_pos)
        current["compositional_dst_pos_id"] = list(GRAPH_POS_VOCAB).index(dst_pos)
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


def _alignment_policy_violation(
    issue_type: str,
    message: str,
    token_indices: Iterable[int] = (),
    word_positions: Iterable[int] = (),
) -> dict:
    return {
        "issue_type": str(issue_type),
        "message": str(message),
        "token_indices": [int(index) for index in token_indices],
        "word_positions": [int(position) for position in word_positions],
    }


def validate_alignment_policy(
    words: list[dict],
    input_text: str,
    graph_text: str,
    offset_mapping: list[tuple[int, int]],
) -> dict:
    """Validate and describe the single formal word-to-subword alignment policy.

    The function is deliberately side-effect free.  It returns the alignment and
    every policy violation so diagnostics can report failures without maintaining
    a second acceptance rule.  ``align_parser_words_to_subwords`` below is the
    strict adapter used by graph-cache construction.
    """
    violations: list[dict] = []
    try:
        graph_start = _find_graph_text_start(input_text, graph_text)
    except GraphCacheError as exc:
        violation = _alignment_policy_violation("graph_text_span_invalid", str(exc))
        return {
            "valid": False,
            "aligned": [],
            "word_to_subword": [],
            "token_to_word_positions": {},
            "token_spans": {},
            "word_coverage": [],
            "shared_subwords": [],
            "violations": [violation],
            "error_message": violation["message"],
        }

    token_spans: dict[int, tuple[int, int]] = {}
    for token_index, offset in enumerate(offset_mapping):
        try:
            token_start, token_end = (int(value) for value in offset)
        except (TypeError, ValueError) as exc:
            violation = _alignment_policy_violation(
                "out_of_bounds_mapping",
                f"token offset is invalid: token_index={token_index} offset={offset!r}",
                [token_index],
            )
            violations.append(violation)
            continue
        relative_start = token_start - graph_start
        relative_end = token_end - graph_start
        if relative_end > relative_start:
            token_spans[token_index] = (relative_start, relative_end)

    aligned: list[list[int]] = []
    word_coverage: list[dict] = []
    for word_position, word in enumerate(words):
        start = int(word["start"])
        end = int(word["end"])
        word_length = max(0, end - start)
        out_of_bounds_word = start < 0 or end < start or end > len(graph_text)
        if out_of_bounds_word:
            violations.append(
                _alignment_policy_violation(
                    "out_of_bounds_mapping",
                    f"parser word span outside graph text: index={word['index']} span=({start},{end}) text_length={len(graph_text)}",
                    word_positions=[word_position],
                )
            )
        token_indices = [
            token_index
            for token_index, (token_start, token_end) in token_spans.items()
            if token_start < end and token_end > start
        ]
        aligned.append(token_indices)
        if not token_indices:
            violations.append(
                _alignment_policy_violation(
                    "unaligned_parser_word",
                    f"parser word cannot be aligned to subwords: index={word['index']} text={word['text']!r}",
                    word_positions=[word_position],
                )
            )
            word_coverage.append(
                {
                    "word_position": word_position,
                    "token_indices": [],
                    "covered_length": 0,
                    "word_length": word_length,
                    "fully_covered": False,
                }
            )
            continue

        covered_until = start
        covered_ranges: list[tuple[int, int]] = []
        for token_index in token_indices:
            token_start, token_end = token_spans[token_index]
            clipped_start = max(start, token_start)
            clipped_end = min(end, token_end)
            if clipped_end > clipped_start:
                covered_ranges.append((clipped_start, clipped_end))
            if clipped_start > covered_until:
                violations.append(
                    _alignment_policy_violation(
                        "incomplete_character_coverage",
                        f"parser word has an uncovered character gap: index={word['index']} text={word['text']!r}",
                        token_indices,
                        [word_position],
                    )
                )
                break
            covered_until = max(covered_until, clipped_end)
        if covered_until < end and not any(
            item["issue_type"] == "incomplete_character_coverage"
            and word_position in item["word_positions"]
            for item in violations
        ):
            violations.append(
                _alignment_policy_violation(
                    "incomplete_character_coverage",
                    f"parser word is only partially aligned: index={word['index']} text={word['text']!r}",
                    token_indices,
                    [word_position],
                )
            )
        ordered_ranges = sorted(covered_ranges)
        merged_ranges: list[tuple[int, int]] = []
        for range_start, range_end in ordered_ranges:
            if not merged_ranges or range_start > merged_ranges[-1][1]:
                merged_ranges.append((range_start, range_end))
            else:
                merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], range_end))
        covered_length = sum(range_end - range_start for range_start, range_end in merged_ranges)
        fully_covered = (
            not out_of_bounds_word
            and bool(merged_ranges)
            and merged_ranges[0][0] <= start
            and merged_ranges[-1][1] >= end
            and covered_length >= word_length
            and all(
                right[0] <= left[1]
                for left, right in zip(merged_ranges, merged_ranges[1:])
            )
        )
        if not fully_covered and not any(
            item["issue_type"] == "incomplete_character_coverage"
            and word_position in item["word_positions"]
            for item in violations
        ):
            violations.append(
                _alignment_policy_violation(
                    "incomplete_character_coverage",
                    f"parser word character coverage incomplete: index={word['index']} text={word['text']!r}",
                    token_indices,
                    [word_position],
                )
            )
        word_coverage.append(
            {
                "word_position": word_position,
                "token_indices": token_indices,
                "covered_length": covered_length,
                "word_length": word_length,
                "fully_covered": fully_covered,
            }
        )

    token_to_word_positions: dict[int, list[int]] = {}
    for word_position, token_indices in enumerate(aligned):
        for token_index in token_indices:
            token_to_word_positions.setdefault(token_index, []).append(word_position)

    shared_subwords: list[dict] = []
    for token_index, word_positions in token_to_word_positions.items():
        token_start, token_end = token_spans[token_index]
        if token_start < 0 or token_end > len(graph_text) or token_end <= token_start:
            violations.append(
                _alignment_policy_violation(
                    "out_of_bounds_mapping",
                    f"mapped subword span outside graph text: token_index={token_index} span=({token_start},{token_end})",
                    [token_index],
                    word_positions,
                )
            )
        if len(word_positions) == 1:
            word = words[word_positions[0]]
            if token_start < int(word["start"]) or token_end > int(word["end"]):
                violations.append(
                    _alignment_policy_violation(
                        "token_span_exceeds_parser_word",
                        f"mapped subword exceeds parser word span: token_index={token_index} token_span=({token_start},{token_end}) word_index={word['index']} word_span=({word['start']},{word['end']})",
                        [token_index],
                        word_positions,
                    )
                )
            continue

        first = word_positions[0]
        last = word_positions[-1]
        shared_words = words[first : last + 1]
        positions_are_contiguous = word_positions == list(range(first, last + 1))
        spans_are_contiguous = all(
            int(left["end"]) == int(right["start"])
            for left, right in zip(shared_words, shared_words[1:])
        )
        same_sentence = len({int(word.get("sentence_index", 0)) for word in shared_words}) == 1
        exact_union = (
            int(shared_words[0]["start"]) == token_start
            and int(shared_words[-1]["end"]) == token_end
        )
        contained_in_parser_union = (
            token_start >= int(shared_words[0]["start"])
            and token_end <= int(shared_words[-1]["end"])
        )
        gaps_are_spaces = any(
            int(left["end"]) < int(right["start"])
            and graph_text[int(left["end"]) : int(right["start"])].isspace()
            for left, right in zip(shared_words, shared_words[1:])
        )
        overlaps_all_recorded_words = all(
            token_start < int(word["end"]) and token_end > int(word["start"])
            for word in (words[position] for position in word_positions)
        )
        legal = (
            positions_are_contiguous
            and spans_are_contiguous
            and same_sentence
            and contained_in_parser_union
            and overlaps_all_recorded_words
        )
        partial_contiguous_shared_subword = legal and not exact_union
        shared = {
            "token_index": token_index,
            "word_positions": list(word_positions),
            "word_indices": [int(word["index"]) for word in shared_words],
            "positions_are_contiguous": positions_are_contiguous,
            "spans_are_contiguous": spans_are_contiguous,
            "same_sentence": same_sentence,
            "exact_union": exact_union,
            "contained_in_parser_union": contained_in_parser_union,
            "gaps_are_spaces": gaps_are_spaces,
            "overlaps_all_recorded_words": overlaps_all_recorded_words,
            "partial_contiguous_shared_subword": partial_contiguous_shared_subword,
            "legal": legal,
        }
        shared_subwords.append(shared)
        if not positions_are_contiguous:
            violations.append(
                _alignment_policy_violation(
                    "non_contiguous_shared_subword",
                    "non_contiguous_shared_subword: shared subword skips parser words: "
                    f"token_index={token_index} word_positions={word_positions}",
                    [token_index],
                    word_positions,
                )
            )
        elif not spans_are_contiguous and gaps_are_spaces:
            violations.append(
                _alignment_policy_violation(
                    "cross_space_shared_subword",
                    "cross_space_shared_subword: shared subword crosses whitespace: "
                    f"token_index={token_index} word_positions={word_positions}",
                    [token_index],
                    word_positions,
                )
            )
        elif not spans_are_contiguous:
            violations.append(
                _alignment_policy_violation(
                    "non_contiguous_shared_subword",
                    "non_contiguous_shared_subword: parser word spans are not contiguous: "
                    f"token_index={token_index} word_positions={word_positions}",
                    [token_index],
                    word_positions,
                )
            )
        if not same_sentence:
            violations.append(
                _alignment_policy_violation(
                    "cross_sentence_shared_subword",
                    "cross_sentence_shared_subword: shared subword crosses parser sentences: "
                    f"token_index={token_index} word_positions={word_positions}",
                    [token_index],
                    word_positions,
                )
            )
        if not contained_in_parser_union:
            violations.append(
                _alignment_policy_violation(
                    "shared_subword_outside_parser_union",
                    "shared_subword_outside_parser_union: shared subword exceeds parser word union: "
                    f"token_index={token_index} word_positions={word_positions} "
                    f"token_span=({token_start},{token_end}) "
                    f"parser_union=({shared_words[0]['start']},{shared_words[-1]['end']})",
                    [token_index],
                    word_positions,
                )
            )
        if not overlaps_all_recorded_words:
            violations.append(
                _alignment_policy_violation(
                    "non_contiguous_shared_subword",
                    "non_contiguous_shared_subword: shared subword does not overlap every recorded parser word: "
                    f"token_index={token_index} word_positions={word_positions}",
                    [token_index],
                    word_positions,
                )
            )

    return {
        "valid": not violations,
        "aligned": aligned,
        "word_to_subword": aligned,
        "token_to_word_positions": token_to_word_positions,
        "token_spans": token_spans,
        "word_coverage": word_coverage,
        "shared_subwords": shared_subwords,
        "violations": violations,
        "error_message": violations[0]["message"] if violations else "",
    }


def align_parser_words_to_subwords(
    words: list[dict],
    input_text: str,
    graph_text: str,
    offset_mapping: list[tuple[int, int]],
) -> list[list[int]]:
    validation = validate_alignment_policy(words, input_text, graph_text, offset_mapping)
    if not validation["valid"]:
        raise GraphCacheError(validation["error_message"])
    return validation["aligned"]


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
    subword_assignment_counts: dict[int, int] = {}
    for token_indices in word_to_subword:
        for token_index in token_indices:
            subword_assignment_counts[token_index] = subword_assignment_counts.get(token_index, 0) + 1
    shared_subword_count = sum(count > 1 for count in subword_assignment_counts.values())
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
        "alignment_policy_version": ALIGNMENT_POLICY_VERSION,
        "alignment": {
            "valid_word_count": len(words),
            "unaligned_word_count": 0,
            "mapped_subword_count": sum(len(indices) for indices in word_to_subword),
            "unique_mapped_subword_count": len(subword_assignment_counts),
            "shared_subword_count": shared_subword_count,
            "shared_subword_assignments": sum(
                count - 1 for count in subword_assignment_counts.values() if count > 1
            ),
        },
    }


def apply_relation_vocab(records: list[dict], relation_vocab: list[str] | None = None) -> list[str]:
    relation_vocab = relation_vocab or sorted(
        {
            edge["relation_key"]
            for record in records
            for edge in record["edges"]
        }
    )
    if UNK_RELATION_KEY not in relation_vocab:
        relation_vocab = [*relation_vocab, UNK_RELATION_KEY]
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
        record = self.get_record(row)
        return {
            "word_to_subword": [list(indices) for indices in record["word_to_subword"]],
            "word_mask": [1] * len(record["word_to_subword"]),
            "edge_src": [int(edge["src"]) for edge in record["edges"]],
            "edge_dst": [int(edge["dst"]) for edge in record["edges"]],
            "relation_id": [int(edge["relation_id"]) for edge in record["edges"]],
            "dependency_relation_id": [int(edge["dependency_relation_id"]) for edge in record["edges"]],
            "pos_pair_id": [int(edge["pos_pair_id"]) for edge in record["edges"]],
            "compositional_dependency_id": [int(edge["compositional_dependency_id"]) for edge in record["edges"]],
            "compositional_direction_id": [int(edge["compositional_direction_id"]) for edge in record["edges"]],
            "compositional_src_pos_id": [int(edge["compositional_src_pos_id"]) for edge in record["edges"]],
            "compositional_dst_pos_id": [int(edge["compositional_dst_pos_id"]) for edge in record["edges"]],
            "edge_mask": [1] * len(record["edges"]),
        }

    def get_record(self, row: dict) -> dict:
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
        return record

    def get_parser_tokens(self, row: dict) -> list[dict]:
        return [dict(token) for token in self.get_record(row)["parser_tokens"]]


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

    def get_parser_tokens(self, row: dict) -> list[dict]:
        split = "target_unlabeled" if row.get("augmentation") == "target_unlabeled" else "source_train"
        return self.caches[split].get_parser_tokens(row)


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


def _parser_identity_matches(observed: dict | None, expected: dict | None) -> bool:
    if observed is None or expected is None:
        return observed == expected
    observed_norm = dict(observed)
    expected_norm = dict(expected)
    observed_norm.pop("resource_dir", None)
    expected_norm.pop("resource_dir", None)
    return observed_norm == expected_norm


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
    if manifest.get("alignment_policy_version") != ALIGNMENT_POLICY_VERSION:
        raise GraphCacheError("graph cache alignment policy version mismatch")
    if manifest.get("target_test_access") is not False:
        raise GraphCacheError("graph cache manifest must explicitly forbid target test access")
    if manifest.get("relation_vocab_sha256") != sha256_file(vocab_path):
        raise GraphCacheError("graph relation vocabulary SHA256 mismatch")
    if split not in {"source_train", "source_dev", "target_unlabeled"}:
        raise GraphCacheError(f"target_test graph cache is forbidden: {split}")
    if tokenizer_identity is not None and manifest.get("tokenizer_identity") != tokenizer_identity:
        raise GraphCacheError(f"tokenizer identity mismatch in cache manifest: split={split}")
    if parser_identity is not None and not _parser_identity_matches(manifest.get("parser_identity"), parser_identity):
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
        if parser_identity is not None and not _parser_identity_matches(record.get("parser_identity"), parser_identity):
            raise GraphCacheError(f"parser identity mismatch: split={split} row_id={record['row_id']}")
        if record.get("alignment_policy_version") != ALIGNMENT_POLICY_VERSION:
            raise GraphCacheError(f"alignment policy mismatch: split={split} row_id={record['row_id']}")
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
        "alignment_policy_version": ALIGNMENT_POLICY_VERSION,
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


def verify_tokenizer_input_equivalence(
    reference_tokenizer,
    runtime_tokenizer,
    rows: list[dict],
    *,
    use_task_prefix: bool,
    max_length: int,
) -> dict:
    """Prove that task-output tokens did not alter graph input alignment."""
    for index, row in enumerate(rows):
        row_id = str(row.get("id", index))
        text = str(row.get("text", ""))
        input_text = f"extract aste: {text}" if use_task_prefix else text
        common = {
            "add_special_tokens": True,
            "return_offsets_mapping": True,
            "truncation": True,
            "max_length": int(max_length),
        }
        reference = reference_tokenizer(input_text, **common)
        runtime = runtime_tokenizer(input_text, **common)
        if (
            list(reference.get("input_ids", [])) != list(runtime.get("input_ids", []))
            or list(reference.get("offset_mapping", [])) != list(runtime.get("offset_mapping", []))
        ):
            raise GraphCacheError(
                f"runtime tokenizer changes graph input tokenization: row_id={row_id}"
            )
    return {"rows_checked": len(rows), "differences": 0}


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
    relation_vocab: list[str] | None = None,
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
    relation_vocab = apply_relation_vocab(all_records, relation_vocab=relation_vocab)
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
        "alignment_policy_version": ALIGNMENT_POLICY_VERSION,
        "splits": {split: len(rows) for split, rows in split_rows.items()},
        "input_sha256": {split: _split_input_identity(rows, use_task_prefix) for split, rows in split_rows.items()},
        "parser_identity": parser_identity,
        "tokenizer_identity": tokenizer_identity,
        "relation_vocab_sha256": sha256_file(output_dir / "relation_vocab.json"),
        "relation_vocab_size": len(relation_vocab),
        "compositional_relation_schema": {
            "version": COMPOSITIONAL_RELATION_SCHEMA_VERSION,
            "dependency_vocab": list(GRAPH_DEPENDENCY_VOCAB),
            "direction_vocab": list(GRAPH_DIRECTIONS),
            "pos_vocab": list(GRAPH_POS_VOCAB),
            "combined_relation_lookup": "disabled_for_new_adapter",
        },
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
    relation_vocab: list[str] | None = None,
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
        relation_vocab=relation_vocab,
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
    parser.add_argument("--relation_vocab_file", default="")
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
        relation_vocab=json.loads(Path(args.relation_vocab_file).read_text(encoding="utf-8")) if args.relation_vocab_file else None,
    )


if __name__ == "__main__":
    main()
