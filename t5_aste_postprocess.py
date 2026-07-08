from __future__ import annotations

import string

from t5_aste_data import canonicalize_triplet_text, micro_f1, parse_triplet_text_list, triplets_to_text


def _levenshtein(left: str, right: str) -> int:
    left = left.lower()
    right = right.lower()
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _normalize_for_match(text: str) -> str:
    return " ".join(text.lower().split())


def _clean_token(token: str) -> str:
    return token.strip(string.punctuation).lower()


def _is_simple_inflection(left: str, right: str) -> bool:
    if left == right:
        return True
    pairs = [(left, right), (right, left)]
    for shorter, longer in pairs:
        if longer == f"{shorter}s" or longer == f"{shorter}es":
            return True
        if shorter.endswith("y") and longer == f"{shorter[:-1]}ies":
            return True
    return False


def _exact_token_sequence_match(term_tokens: list[str], sentence_tokens: list[str]) -> bool:
    if not term_tokens:
        return False
    for start in range(0, len(sentence_tokens) - len(term_tokens) + 1):
        if sentence_tokens[start : start + len(term_tokens)] == term_tokens:
            return True
    return False


def _find_token_sequence(term_tokens: list[str], sentence_tokens: list[str]) -> int | None:
    if not term_tokens:
        return None
    for start in range(0, len(sentence_tokens) - len(term_tokens) + 1):
        if sentence_tokens[start : start + len(term_tokens)] == term_tokens:
            return start
    return None


def recover_terms_with_editdistance(original_term: str, sentence_tokens: list[str]) -> str:
    """BGCA-style word-level span recovery with light punctuation cleanup."""
    term = _normalize_for_match(original_term)
    if not term or not sentence_tokens:
        return term

    cleaned_sentence_tokens = [_clean_token(token) for token in sentence_tokens]
    new_words = []
    for word in term.split():
        cleaned_word = _clean_token(word)
        if not cleaned_word:
            continue
        distances = []
        for token in cleaned_sentence_tokens:
            distance = _levenshtein(cleaned_word, token)
            if _is_simple_inflection(cleaned_word, token):
                distance = 0
            distances.append(distance)
        smallest_idx = distances.index(min(distances))
        new_words.append(cleaned_sentence_tokens[smallest_idx])

    return " ".join(word for word in new_words if word)


def _fix_term(term: str, sentence_tokens: list[str]) -> str:
    normalized = _normalize_for_match(term)
    cleaned_sentence_tokens = [_clean_token(token) for token in sentence_tokens]
    term_tokens = [_clean_token(token) for token in normalized.split() if _clean_token(token)]
    if _exact_token_sequence_match(term_tokens, cleaned_sentence_tokens):
        return normalized
    return recover_terms_with_editdistance(normalized, sentence_tokens)


_SAFE_LEADING_ASPECT_MODIFIERS = {
    "ergonomic",
    "frozen",
    "large",
    "longer",
    "textured",
}

_SAFE_LEFT_ASPECT_EXPANDERS = {
    "ati",
    "built-in",
    "command",
    "inbuilt",
    "island",
    "left",
    "right",
    "usb",
    "windows",
}

_SAFE_RIGHT_ASPECT_EXPANDERS = {
    "applications",
    "buttons",
    "card",
    "graphics",
    "key",
    "life",
    "ports",
    "port",
    "quality",
    "size",
    "software",
    "support",
    "system",
    "updates",
}

def _maybe_contract_aspect_modifier(aspect: str, opinion: str, sentence_tokens: list[str]) -> str:
    aspect_tokens = [_clean_token(token) for token in aspect.split() if _clean_token(token)]
    if len(aspect_tokens) < 2:
        return aspect

    head_tokens = aspect_tokens[1:]
    if _find_token_sequence(head_tokens, sentence_tokens) is None:
        return aspect

    first = aspect_tokens[0]
    opinion_tokens = {_clean_token(token) for token in opinion.split() if _clean_token(token)}
    if first in opinion_tokens or first in _SAFE_LEADING_ASPECT_MODIFIERS:
        return " ".join(head_tokens)
    return aspect


def _maybe_expand_aspect_boundary(aspect: str, sentence_tokens: list[str]) -> str:
    aspect_tokens = [_clean_token(token) for token in aspect.split() if _clean_token(token)]
    start = _find_token_sequence(aspect_tokens, sentence_tokens)
    if start is None:
        return aspect

    end = start + len(aspect_tokens)
    expanded = list(aspect_tokens)

    if start > 0 and sentence_tokens[start - 1] in _SAFE_LEFT_ASPECT_EXPANDERS:
        expanded.insert(0, sentence_tokens[start - 1])

    if end < len(sentence_tokens):
        next_token = sentence_tokens[end]
        if next_token in _SAFE_RIGHT_ASPECT_EXPANDERS:
            expanded.append(next_token)

    return " ".join(expanded)


def _fix_aspect_boundary(aspect: str, opinion: str, sentence_tokens: list[str]) -> str:
    contracted = _maybe_contract_aspect_modifier(aspect, opinion, sentence_tokens)
    return _maybe_expand_aspect_boundary(contracted, sentence_tokens)


def _maybe_fix_opinion_boundary(opinion: str, sentence_tokens: list[str]) -> str:
    tokens = [_clean_token(token) for token in opinion.split() if _clean_token(token)]
    if len(tokens) < 2:
        return opinion

    if tokens[0] == "too" and len(tokens) == 2:
        return tokens[1]

    if tokens[:2] == ["not", "available"] and len(tokens) > 2:
        return "not available"

    if len(tokens) >= 3 and tokens[1:3] == ["enough", "to"]:
        return tokens[0]

    if len(tokens) >= 3 and tokens[1] == "to" and tokens[0] in {"easy", "simple"}:
        return tokens[0]

    if len(tokens) >= 2 and tokens[-1] == "out":
        start = _find_token_sequence(tokens, sentence_tokens)
        if start is not None:
            return " ".join(tokens)

    return opinion


def _maybe_expand_opinion_boundary(opinion: str, sentence_tokens: list[str]) -> str:
    tokens = [_clean_token(token) for token in opinion.split() if _clean_token(token)]
    start = _find_token_sequence(tokens, sentence_tokens)
    if start is None:
        return opinion

    end = start + len(tokens)
    if end < len(sentence_tokens) and sentence_tokens[end] == "out" and tokens[-1] in {"shorting", "crapped"}:
        return " ".join(tokens + ["out"])
    return opinion


def _fix_opinion_boundary(opinion: str, sentence_tokens: list[str]) -> str:
    contracted = _maybe_fix_opinion_boundary(opinion, sentence_tokens)
    return _maybe_expand_opinion_boundary(contracted, sentence_tokens)


def fix_pred_triplets(pred_label: str, sentence: str) -> str:
    fixed_triplets = []
    sentence_tokens = [_clean_token(token) for token in sentence.lower().split()]
    for aspect, opinion, sentiment in parse_triplet_text_list(pred_label):
        fixed_aspect = _fix_term(aspect, sentence_tokens)
        fixed_opinion = _fix_term(opinion, sentence_tokens)
        fixed_aspect = _fix_aspect_boundary(fixed_aspect, fixed_opinion, sentence_tokens)
        fixed_opinion = _fix_opinion_boundary(fixed_opinion, sentence_tokens)
        fixed_triplets.append((fixed_aspect, fixed_opinion, sentiment))
    return triplets_to_text(fixed_triplets)


def evaluate_raw_and_fixed(rows: list[dict]) -> dict:
    predictions = []
    raw_preds = []
    fixed_preds = []
    golds = []

    for row in rows:
        text = row["text"]
        gold = canonicalize_triplet_text(row.get("gold", ""))
        pred_raw = canonicalize_triplet_text(row.get("pred", ""))
        pred_fixed = canonicalize_triplet_text(fix_pred_triplets(pred_raw, text))

        raw_preds.append(pred_raw)
        fixed_preds.append(pred_fixed)
        golds.append(gold)
        predictions.append(
            {
                "text": text,
                "gold": gold,
                "pred_raw": pred_raw,
                "pred_fixed": pred_fixed,
                "raw_triplets": parse_triplet_text_list(pred_raw),
                "fixed_triplets": parse_triplet_text_list(pred_fixed),
                "gold_triplets": parse_triplet_text_list(gold),
                "fixed_changed": pred_raw != pred_fixed,
            }
        )

    return {
        "raw_scores": micro_f1(raw_preds, golds),
        "fixed_scores": micro_f1(fixed_preds, golds),
        "predictions": predictions,
    }
