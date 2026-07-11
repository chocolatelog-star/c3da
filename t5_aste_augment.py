from __future__ import annotations

import random
import re
import math
from collections import Counter, defaultdict
from typing import Iterable

from t5_aste_data import canonicalize_triplet_text, parse_triplet_text, triplets_to_text


PROMPT_STYLES = {
    "concept",
    "legacy",
    "masked_mutual",
    "rewrite_aspect",
    "label_composition",
    "label_to_text",
    "rsda_t5_label_composition",
    "sentence_fusion_composition",
}
CHANNEL_MODES = {"all", "aspect", "opinion"}
OPINION_REPLACEMENT_MODES = {"coupled_random", "semantic_same_sentiment", "sentiment_vector"}
GENERIC_ASPECTS = {
    "about",
    "all",
    "application",
    "applications",
    "anything",
    "computer",
    "device",
    "devices",
    "item",
    "machine",
    "product",
    "something",
    "stuff",
    "thing",
    "things",
    "unit",
    "use",
}
ACTION_LIKE_SUFFIXES = ("ing",)
ACTION_LIKE_TERMS = {
    "navigate",
    "runs",
}
OPINION_LIKE_TERMS = {
    "amazing",
    "bad",
    "beautiful",
    "best",
    "better",
    "bright",
    "cheap",
    "clean",
    "comfortable",
    "easy",
    "excellent",
    "expensive",
    "fast",
    "good",
    "great",
    "hard",
    "horrible",
    "long",
    "nice",
    "poor",
    "quick",
    "responsive",
    "rude",
    "slow",
    "terrible",
    "visual",
    "worst",
}
NOISY_LABEL_COMPOSITION_OPINIONS = {
    "anything",
    "backlit",
}
BAD_SPECIFIC_ASPECTS = {
    "abilitiy",
    "kernal",
}
BAD_SPECIFIC_SUFFIXES = {
    " out",
}
ASPECT_TYPE_KEYWORDS = {
    "battery": {"battery", "battery life"},
    "display": {"display", "screen", "monitor", "resolution", "graphics", "video"},
    "input": {"keyboard", "key board", "keys", "trackpad", "touchpad", "mouse pad", "buttons"},
    "performance": {"speed", "performance", "memory", "ram", "processor", "cpu", "hard drive", "hard disc", "storage"},
    "software": {"software", "program", "application", "applications", "os", "operating system", "bios", "imovie", "iphoto", "itunes"},
    "service": {"service", "customer service", "customer support", "support", "warranty", "extended warranty", "sales associate"},
    "connectivity": {"internet", "internet speed", "wifi", "wi-fi", "bluetooth", "usb", "port", "ports", "hdmi"},
    "device": {"laptop", "computer", "machine", "macbook", "notebook", "desktop", "netbook"},
    "design": {"design", "build", "case", "unibody design", "size", "weight", "look", "looks"},
}
OPINION_TYPE_HINTS = {
    "battery": {"long", "short", "hours", "hour", "lasts", "lasted", "charge", "charged"},
    "display": {"bright", "clear", "glossy", "dark", "sharp", "crisp"},
    "input": {"responsive", "comfortable", "sticky", "stuck", "click", "typing", "omitted"},
    "performance": {"fast", "slow", "responsive", "speedy", "powerful", "crashed", "crashes", "freeze", "frozen"},
    "software": {"easy", "useful", "stable", "crashed", "crashes", "working", "installed", "preloaded"},
    "service": {"rude", "helpful", "friendly", "knowledgeable", "terrible", "spotty", "slow"},
    "connectivity": {"connected", "working", "fast", "slow", "locked", "froze"},
    "design": {"beautiful", "slick", "light", "heavy", "thin", "durable"},
}


def _normalize_fragment(fragment: str) -> str:
    return " ".join(fragment.lower().split())


def infer_aspect_type(aspect: str) -> str:
    normalized = _normalize_fragment(aspect)
    for aspect_type, keywords in ASPECT_TYPE_KEYWORDS.items():
        if normalized in keywords:
            return aspect_type
    for aspect_type, keywords in ASPECT_TYPE_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return aspect_type
    return "other"


def _fragment_in_text(fragment: str, text: str) -> bool:
    return _normalize_fragment(fragment) in _normalize_fragment(text)


def build_label_composition_prompt(triplets: list[tuple[str, str, str]]) -> str:
    label = canonicalize_triplet_text(triplets_to_text(triplets))
    return f"generate aste sentence: {label}"


def build_sentence_fusion_prompt(left_text: str, right_text: str, triplets: list[tuple[str, str, str]]) -> str:
    label = canonicalize_triplet_text(triplets_to_text(triplets))
    return f"fuse aste sentences: sentence 1: {left_text} sentence 2: {right_text} labels: {label}"


def _label_text_for_triplet(triplet: tuple[str, str, str]) -> str:
    return canonicalize_triplet_text(triplets_to_text([triplet]))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(_normalize_fragment(left).split())
    right_tokens = set(_normalize_fragment(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def rsda_composition_pair_compatible(
    left: tuple[str, str, str],
    right: tuple[str, str, str],
    similarity: float,
    min_similarity: float = 0.35,
    max_similarity: float = 0.95,
) -> bool:
    left_aspect, left_opinion, left_sentiment = left
    right_aspect, right_opinion, right_sentiment = right
    normalized_left = _normalize_fragment(left_aspect)
    normalized_right = _normalize_fragment(right_aspect)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return False
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return False
    if _token_jaccard(normalized_left, normalized_right) >= 0.5:
        return False
    if similarity < min_similarity or similarity > max_similarity:
        return False
    if (
        left_sentiment == right_sentiment
        and _normalize_fragment(left_opinion) == _normalize_fragment(right_opinion)
        and similarity > 0.9
    ):
        return False
    return True


def _valid_aspect(fragment: str) -> bool:
    normalized = _normalize_fragment(fragment)
    if not normalized or normalized in GENERIC_ASPECTS:
        return False
    words = normalized.split()
    return 1 <= len(words) <= 4


def _aspect_reject_reason(fragment: str, text: str = "") -> str:
    normalized = _normalize_fragment(fragment)
    if not normalized or normalized in GENERIC_ASPECTS:
        return "generic_or_bad_shape"
    words = normalized.split()
    if not 1 <= len(words) <= 4:
        return "generic_or_bad_shape"
    if normalized in ACTION_LIKE_TERMS or (len(words) == 1 and normalized.endswith(ACTION_LIKE_SUFFIXES)):
        return "event_or_action_like"
    if normalized in OPINION_LIKE_TERMS:
        return "opinion_like"
    if text and not _fragment_in_text(normalized, text):
        return "not_in_text"
    return ""


def _specific_aspect_reject_reason(aspect: str, stats: dict, min_pseudo_weight: float) -> str:
    normalized = _normalize_fragment(aspect)
    if normalized in BAD_SPECIFIC_ASPECTS or any(normalized.endswith(suffix) for suffix in BAD_SPECIFIC_SUFFIXES):
        return "bad_specific_shape"
    if stats["target_tf"] < 2:
        return "low_frequency"
    if stats["mean_pseudo_weight"] < 0.645:
        return "low_mean_weight"
    if stats["domain_score"] <= 0:
        return "not_target_specific"
    return ""


def _valid_opinion(fragment: str) -> bool:
    normalized = _normalize_fragment(fragment)
    words = normalized.split()
    return bool(normalized) and 1 <= len(words) <= 5


def _valid_label_composition_triplet(text: str, triplet: tuple[str, str, str]) -> bool:
    aspect, opinion, sentiment = triplet
    aspect_reason = _aspect_reject_reason(aspect, text)
    normalized_aspect = _normalize_fragment(aspect)
    normalized_opinion = _normalize_fragment(opinion)
    if aspect_reason:
        return False
    if normalized_aspect in ACTION_LIKE_TERMS or normalized_aspect in OPINION_LIKE_TERMS:
        return False
    if not _valid_opinion(opinion):
        return False
    if normalized_opinion in NOISY_LABEL_COMPOSITION_OPINIONS:
        return False
    if normalized_opinion in GENERIC_ASPECTS or normalized_opinion in ACTION_LIKE_TERMS:
        return False
    if sentiment not in {"pos", "neg", "neu"}:
        return False
    if not _fragment_in_text(aspect, text) or not _fragment_in_text(opinion, text):
        return False
    return True


def build_domain_memory(pseudo_rows: Iterable[dict], min_pseudo_weight: float = 0.6) -> dict:
    return build_target_memory(pseudo_rows, min_pseudo_weight=min_pseudo_weight)


def _empty_rejected_reasons() -> dict[str, list[str]]:
    return {
        "kept": [],
        "low_weight": [],
        "generic_or_bad_shape": [],
        "event_or_action_like": [],
        "opinion_like": [],
        "not_in_text": [],
    }


def build_source_memory(source_rows: Iterable[dict]) -> dict:
    rows = list(source_rows)
    aspect_counts: Counter[str] = Counter()
    aspect_doc_counts: Counter[str] = Counter()
    opinion_counts: dict[str, Counter[str]] = defaultdict(Counter)
    triplet_counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        seen_aspects_in_row: set[str] = set()
        for aspect, opinion, sentiment in parse_triplet_text(row.get("label", "")):
            if _valid_aspect(aspect):
                aspect_counts[aspect] += 1
                seen_aspects_in_row.add(aspect)
            if _valid_opinion(opinion):
                opinion_counts[sentiment][opinion] += 1
            if _valid_aspect(aspect) and _valid_opinion(opinion):
                triplet_counts[(aspect, opinion, sentiment)] += 1
        for aspect in seen_aspects_in_row:
            aspect_doc_counts[aspect] += 1
    return {
        "aspects": sorted(aspect_counts),
        "aspect_counts": dict(aspect_counts),
        "aspect_doc_counts": dict(aspect_doc_counts),
        "opinions_by_sentiment": {
            sentiment: sorted(counter) for sentiment, counter in sorted(opinion_counts.items())
        },
        "opinion_counts_by_sentiment": {
            sentiment: dict(counter) for sentiment, counter in sorted(opinion_counts.items())
        },
        "triplets": [list(triplet) for triplet in sorted(triplet_counts)],
        "triplet_counts": {"|".join(triplet): count for triplet, count in sorted(triplet_counts.items())},
    }


def build_target_memory(
    pseudo_rows: Iterable[dict],
    min_pseudo_weight: float = 0.6,
    source_memory: dict | None = None,
    source_row_count: int = 0,
) -> dict:
    rows = list(pseudo_rows)
    aspect_counts: Counter[str] = Counter()
    aspect_doc_counts: Counter[str] = Counter()
    aspect_weight_sums: Counter[str] = Counter()
    aspect_in_text_counts: Counter[str] = Counter()
    opinion_counts: dict[str, Counter[str]] = defaultdict(Counter)
    triplet_counts: Counter[tuple[str, str, str]] = Counter()
    opinion_aspect_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rejected = _empty_rejected_reasons()
    for row in rows:
        row_weight = float(row.get("sample_weight", 0.0))
        low_weight = row_weight < min_pseudo_weight
        if low_weight:
            for aspect, _opinion, _sentiment in parse_triplet_text(row.get("label", "")):
                rejected["low_weight"].append(aspect)
            continue
        text = row.get("text", "")
        seen_aspects_in_row: set[str] = set()
        for aspect, opinion, sentiment in parse_triplet_text(row.get("label", "")):
            aspect_reject_reason = _aspect_reject_reason(aspect, text)
            aspect_ok = not aspect_reject_reason
            aspect_in_text = _fragment_in_text(aspect, text)
            if aspect_ok:
                aspect_counts[aspect] += 1
                aspect_weight_sums[aspect] += row_weight
                aspect_in_text_counts[aspect] += 1
                seen_aspects_in_row.add(aspect)
                rejected["kept"].append(aspect)
            else:
                rejected[aspect_reject_reason].append(aspect)
            if _valid_opinion(opinion) and _fragment_in_text(opinion, text):
                opinion_counts[sentiment][opinion] += 1
            if aspect_ok and _valid_opinion(opinion) and _fragment_in_text(opinion, text):
                triplet_counts[(aspect, opinion, sentiment)] += 1
                opinion_aspect_counts[f"{opinion}|{sentiment}"][aspect] += 1
        for aspect in seen_aspects_in_row:
            aspect_doc_counts[aspect] += 1

    source_aspect_counts = Counter()
    source_aspect_doc_counts = Counter()
    if source_memory:
        source_aspect_counts.update(source_memory.get("aspect_counts", {}))
        source_aspect_doc_counts.update(source_memory.get("aspect_doc_counts", {}))
    target_row_count = len(rows)
    aspect_stats = {}
    for aspect in sorted(aspect_counts):
        target_tf = aspect_counts[aspect]
        target_df = aspect_doc_counts[aspect]
        source_tf = int(source_aspect_counts.get(aspect, 0))
        source_df = int(source_aspect_doc_counts.get(aspect, 0))
        target_rate = (target_df + 1) / max(1, target_row_count)
        source_rate = (source_df + 1) / max(1, source_row_count)
        domain_score = math.log(target_rate) - math.log(source_rate)
        aspect_stats[aspect] = {
            "target_tf": int(target_tf),
            "target_df": int(target_df),
            "source_tf": source_tf,
            "source_df": source_df,
            "mean_pseudo_weight": round(aspect_weight_sums[aspect] / max(1, target_tf), 6),
            "in_text_ratio": round(aspect_in_text_counts[aspect] / max(1, target_tf), 6),
            "domain_score": round(domain_score, 6),
        }

    core_target_aspects = sorted(
        aspect
        for aspect, stats in aspect_stats.items()
        if stats["target_df"] >= 2 and stats["mean_pseudo_weight"] >= min_pseudo_weight
    )
    specific_candidate_aspects = sorted(
        aspect
        for aspect, stats in aspect_stats.items()
        if aspect not in core_target_aspects and stats["domain_score"] > 0 and stats["mean_pseudo_weight"] >= min_pseudo_weight
    )
    rejected_specific: dict[str, list[str]] = {
        "bad_specific_shape": [],
        "low_frequency": [],
        "low_mean_weight": [],
        "not_target_specific": [],
    }
    specific_target_aspects = []
    for aspect in specific_candidate_aspects:
        reason = _specific_aspect_reject_reason(aspect, aspect_stats[aspect], min_pseudo_weight)
        if reason:
            rejected_specific[reason].append(aspect)
            continue
        specific_target_aspects.append(aspect)
    specific_target_aspects = sorted(specific_target_aspects)
    return {
        "aspects": sorted(aspect_counts),
        "core_target_aspects": core_target_aspects,
        "specific_candidate_aspects": specific_candidate_aspects,
        "specific_target_aspects": specific_target_aspects,
        "aspect_stats": aspect_stats,
        "aspect_types": {aspect: infer_aspect_type(aspect) for aspect in sorted(aspect_counts)},
        "aspect_counts": dict(aspect_counts),
        "opinions_by_sentiment": {
            sentiment: sorted(counter) for sentiment, counter in sorted(opinion_counts.items())
        },
        "opinion_counts_by_sentiment": {
            sentiment: dict(counter) for sentiment, counter in sorted(opinion_counts.items())
        },
        "triplets": [list(triplet) for triplet in sorted(triplet_counts)],
        "triplet_counts": {"|".join(triplet): count for triplet, count in sorted(triplet_counts.items())},
        "opinion_aspect_counts": {
            key: dict(counter) for key, counter in sorted(opinion_aspect_counts.items())
        },
        "rejected_aspects_by_reason": {reason: sorted(set(values)) for reason, values in rejected.items()},
        "rejected_specific_aspects_by_reason": {
            reason: sorted(set(values)) for reason, values in rejected_specific.items()
        },
        "min_pseudo_weight": min_pseudo_weight,
    }


def build_cross_domain_memory(
    source_memory: dict,
    target_memory: dict,
    max_candidate_opinions_per_sentiment: int = 30,
) -> dict:
    preferred_aspects = set(target_memory.get("core_target_aspects", []))
    preferred_aspects.update(target_memory.get("specific_target_aspects", []))
    target_aspects = sorted(preferred_aspects) if preferred_aspects else list(target_memory.get("aspects", []))
    opinions_by_sentiment: dict[str, list[str]] = {}
    candidate_opinions_by_sentiment: dict[str, list[str]] = {}
    sentiments = set(source_memory.get("opinions_by_sentiment", {})) | set(target_memory.get("opinions_by_sentiment", {}))
    for sentiment in sorted(sentiments):
        merged = set(source_memory.get("opinions_by_sentiment", {}).get(sentiment, []))
        merged.update(target_memory.get("opinions_by_sentiment", {}).get(sentiment, []))
        opinions_by_sentiment[sentiment] = sorted(merged)

        target_counts = Counter(target_memory.get("opinion_counts_by_sentiment", {}).get(sentiment, {}))
        source_counts = Counter(source_memory.get("opinion_counts_by_sentiment", {}).get(sentiment, {}))
        ranked = sorted(
            merged,
            key=lambda opinion: (
                -target_counts.get(opinion, 0),
                -source_counts.get(opinion, 0),
                opinion,
            ),
        )
        candidate_opinions_by_sentiment[sentiment] = ranked[:max_candidate_opinions_per_sentiment]

    candidate_triplets = []
    observed_target_triplets = {tuple(triplet) for triplet in target_memory.get("triplets", [])}
    for aspect in target_aspects:
        for sentiment, opinions in candidate_opinions_by_sentiment.items():
            for opinion in opinions:
                candidate_triplets.append([aspect, opinion, sentiment])
    for aspect, opinion, sentiment in sorted(observed_target_triplets):
        if aspect in target_aspects:
            triplet = [aspect, opinion, sentiment]
            if triplet not in candidate_triplets:
                candidate_triplets.append(triplet)
    return {
        "target_aspects": target_aspects,
        "aspect_types": {
            aspect: target_memory.get("aspect_types", {}).get(aspect, infer_aspect_type(aspect))
            for aspect in target_aspects
        },
        "opinions_by_sentiment": opinions_by_sentiment,
        "candidate_opinions_by_sentiment": candidate_opinions_by_sentiment,
        "candidate_triplets": candidate_triplets,
        "target_triplet_counts": target_memory.get("triplet_counts", {}),
        "opinion_aspect_counts": target_memory.get("opinion_aspect_counts", {}),
        "candidate_generation": {
            "max_candidate_opinions_per_sentiment": max_candidate_opinions_per_sentiment,
            "uses_target_observed_triplets": True,
        },
        "source_triplets": source_memory.get("triplets", []),
        "target_triplets": target_memory.get("triplets", []),
    }


def build_target_aspect_bank(pseudo_rows: Iterable[dict]) -> list[str]:
    bank = set()
    for row in pseudo_rows:
        for aspect, _opinion, _sentiment in parse_triplet_text(row.get("label", "")):
            bank.add(aspect)
    return sorted(bank)


def build_opinion_sentiment_bank(source_rows: Iterable[dict], pseudo_rows: Iterable[dict]) -> dict[str, list[str]]:
    bank: dict[str, set[str]] = defaultdict(set)
    for row in list(source_rows) + list(pseudo_rows):
        for _aspect, opinion, sentiment in parse_triplet_text(row.get("label", "")):
            bank[sentiment].add(opinion)
    return {sentiment: sorted(opinions) for sentiment, opinions in bank.items()}


POLARITY_HINTS = {
    "pos": ["so good", "excellent", "great"],
    "neg": ["so bad", "terrible", "poor"],
    "neu": ["average", "okay", "normal"],
}


def label_terms(label: str) -> tuple[list[str], list[str]]:
    aspects: list[str] = []
    opinions: list[str] = []
    seen_aspects: set[str] = set()
    seen_opinions: set[str] = set()
    for aspect, opinion, _sentiment in parse_triplet_text(label):
        if aspect and aspect not in seen_aspects:
            seen_aspects.add(aspect)
            aspects.append(aspect)
        if opinion and opinion not in seen_opinions:
            seen_opinions.add(opinion)
            opinions.append(opinion)
    return aspects, opinions


def build_label_invariant_prompt(label: str) -> str:
    return build_concept_prompt(label)


def build_legacy_label_invariant_prompt(label: str) -> str:
    aspects, opinions = label_terms(label)
    aspect_text = " ; ".join(aspects)
    opinion_text = " ; ".join(opinions)
    return (
        "paraphrase with label terms: "
        f"aspect terms: {aspect_text} ; "
        f"opinion terms: {opinion_text} ; "
        f"label: {label}"
    )


def build_concept_prompt(label: str) -> str:
    aspects, opinions = label_terms(label)
    concepts = " ; ".join(aspects + opinions)
    if concepts:
        return concepts
    return canonicalize_triplet_text(label)


def format_domain_prefix(domain_name: str = "", style: str = "none") -> str:
    domain = (domain_name or "").strip()
    if style in {"", "none"} or not domain:
        return ""
    if style == "text":
        return f"target domain: [{domain}] ; "
    if style == "bracket":
        return f"[{domain}] "
    raise ValueError("domain_prefix_style must be one of none, text, bracket")


def apply_domain_prefix(prompt: str, domain_name: str = "", style: str = "none") -> str:
    return f"{format_domain_prefix(domain_name, style)}{prompt}"


def _mask_first_fragment(text: str, fragment: str, mask_token: str) -> str:
    pattern = re.compile(re.escape(fragment), flags=re.IGNORECASE)
    masked, count = pattern.subn(mask_token, text, count=1)
    if count:
        return masked
    return f"{mask_token} {text}"


def build_masked_aspect_prompt(
    text: str,
    old_triplet: tuple[str, str, str],
    new_triplet: tuple[str, str, str],
    domain_name: str = "",
    domain_prefix_style: str = "none",
) -> str:
    old_aspect, _old_opinion, _old_sentiment = old_triplet
    new_aspect, opinion, sentiment = new_triplet
    masked_text = _mask_first_fragment(text, old_aspect, "[ASP]")
    prompt = (
        f"masked aspect edit: {masked_text} ; "
        f"new aspect: {new_aspect} ; "
        f"opinion: {opinion} ; "
        f"sentiment: {sentiment}"
    )
    return apply_domain_prefix(prompt, domain_name, domain_prefix_style)


def build_aspect_rewrite_prompt(text: str, old_triplet: tuple[str, str, str], new_triplet: tuple[str, str, str]) -> str:
    old_aspect, _old_opinion, _old_sentiment = old_triplet
    new_aspect, opinion, sentiment = new_triplet
    return (
        f"rewrite aspect: {text} ; "
        f"old aspect: {old_aspect} ; "
        f"new aspect: {new_aspect} ; "
        f"opinion: {opinion} ; "
        f"sentiment: {sentiment}"
    )


def build_masked_opinion_sentiment_prompt(
    text: str,
    old_triplet: tuple[str, str, str],
    new_triplet: tuple[str, str, str],
    domain_name: str = "",
    domain_prefix_style: str = "none",
) -> str:
    aspect, old_opinion, _old_sentiment = old_triplet
    _new_aspect, new_opinion, new_sentiment = new_triplet
    masked_text = _mask_first_fragment(text, old_opinion, "[OPI]")
    prompt = (
        f"masked opinion-sentiment edit: {masked_text} ; "
        f"aspect: {aspect} ; "
        f"new opinion: {new_opinion} ; "
        f"new sentiment: {new_sentiment}"
    )
    return apply_domain_prefix(prompt, domain_name, domain_prefix_style)


def _replace_first_fragment(text: str, old_fragment: str, new_fragment: str) -> str:
    pattern = re.compile(re.escape(old_fragment), flags=re.IGNORECASE)
    replaced, count = pattern.subn(new_fragment, text, count=1)
    if count:
        return replaced
    return text


def _choose_replacement_aspect(
    rng: random.Random,
    aspect_pool: list[tuple[str, str]],
    old_aspect: str,
    sentiment: str,
) -> str:
    same_sentiment = [
        aspect for aspect, aspect_sentiment in aspect_pool if aspect != old_aspect and aspect_sentiment == sentiment
    ]
    if same_sentiment:
        return rng.choice(sorted(set(same_sentiment)))
    different = [aspect for aspect, _sentiment in aspect_pool if aspect != old_aspect]
    if different:
        return rng.choice(sorted(set(different)))
    return old_aspect


def _choose_replacement_opinion(
    rng: random.Random,
    opinion_bank: dict[str, list[str]],
    old_opinion: str,
    sentiment: str,
) -> tuple[str, str]:
    sentiment_choices = [s for s, opinions in opinion_bank.items() if opinions and s != sentiment]
    if not sentiment_choices:
        sentiment_choices = [s for s, opinions in opinion_bank.items() if opinions]
    if not sentiment_choices:
        return old_opinion, sentiment
    new_sentiment = rng.choice(sentiment_choices)
    candidate_opinions = [opinion for opinion in opinion_bank.get(new_sentiment, []) if opinion != old_opinion]
    if candidate_opinions:
        return rng.choice(sorted(set(candidate_opinions))), new_sentiment
    fallback_opinions = opinion_bank.get(sentiment, [])
    if fallback_opinions:
        return rng.choice(sorted(set(fallback_opinions))), sentiment
    return old_opinion, sentiment


PROMPT_LEAK_PATTERNS = [
    r"\blabel\s*:",
    r"\baspect\s+terms?\b",
    r"\bopinion\s+terms?\b",
    r"\bparaphrase\b",
    r"\bwith\s+label\b",
    r"\bwith\s+terms?\b",
    r"\btarget\s+domain\b",
    r"\[(rest14|rest15|rest16|laptop14)\]",
    r"\bpos\s*>",
    r"\bneg\s*>",
    r"\bneu\s*>",
    r"\bopinion\s*>",
    r"<\s*/?\s*(pos|neg|neu|opinion)\s*>",
]


def is_prompt_leak(text: str) -> bool:
    lowered = text.lower()
    if lowered.count(";") >= 3:
        return True
    return any(re.search(pattern, lowered) for pattern in PROMPT_LEAK_PATTERNS)


def filter_augmented_text_quality(text: str) -> tuple[bool, str]:
    stripped = " ".join(text.split())
    if not stripped:
        return False, "empty"
    if is_prompt_leak(stripped):
        return False, "prompt_leak"
    words = stripped.split()
    if len(words) < 4:
        return False, "too_short"
    return True, ""


PLURAL_HINTS = {"they", "them", "their", "these", "those"}
SINGULAR_BAD_WITH_PLURAL_CONTEXT = {
    "aero",
    "bios",
    "build",
    "customer support",
    "desktop",
    "fan",
    "fit",
    "hard disc",
    "mac osx",
    "motherboard",
    "operating system",
    "processor",
    "powerpoint",
    "system",
    "unibody design",
    "warranty",
}
ADJECTIVE_SLOT_BLOCKLIST = {
    "build",
    "customer support",
    "desktop",
    "games",
    "power supply",
    "ram",
    "unibody design",
    "warranty",
}
ACTIVITY_SLOT_BLOCKLIST = {
    "desktop",
    "games",
    "ram",
    "warranty",
}
GOOD_SLOT_ALLOWED = {
    "battery",
    "battery life",
    "display",
    "features",
    "keyboard",
    "laptop",
    "memory",
    "performance",
    "price",
    "screen",
    "system",
    "trackpad",
}
BAD_ARTICLE_ASPECTS = {
    "aero",
    "imovie",
    "internet",
    "internet speed",
    "macbook",
    "photoshop",
    "powerpoint",
    "software",
}
PERSON_OR_SERVICE_ASPECTS = {
    "customer service",
    "customer support",
    "sales associate",
    "service",
    "support",
    "tech support",
}
PERSON_OR_SERVICE_SLOT_ALLOWED = {
    "customer service",
    "customer support",
    "service",
    "support",
    "tech support",
}
CRASH_SLOT_BLOCKLIST = {
    "customer support",
    "desktop",
    "games",
    "hard disc",
    "power supply",
    "ram",
    "unibody design",
    "warranty",
}
QUANTITY_SLOT_ALLOWED_BY_RIGHT = {
    "gigabytes": {"hard disc", "memory", "ram"},
    "gb": {"hard disc", "memory", "ram"},
    "hours": {"battery", "battery life"},
    "hour": {"battery", "battery life"},
}
PURCHASE_OR_PAYMENT_SLOT_BLOCKLIST = {
    "bios",
    "powerpoint",
}
COUNTRY_SLOT_BLOCKLIST = {
    "hard disc",
    "keyboard",
    "motherboard",
    "power supply",
}


def _context_window(text: str, fragment: str, width: int = 5) -> tuple[list[str], list[str]]:
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    frag_words = _normalize_fragment(fragment).split()
    if not words or not frag_words:
        return [], []
    for idx in range(0, len(words) - len(frag_words) + 1):
        if words[idx : idx + len(frag_words)] == frag_words:
            left = words[max(0, idx - width) : idx]
            right = words[idx + len(frag_words) : idx + len(frag_words) + width]
            return left, right
    return [], []


def aspect_replacement_compatible(text: str, old_aspect: str, new_aspect: str) -> bool:
    normalized_new = _normalize_fragment(new_aspect)
    normalized_old = _normalize_fragment(old_aspect)
    left, right = _context_window(text, old_aspect)
    local = set(left + right)
    immediate_left = left[-1] if left else ""
    immediate_right = right[0] if right else ""
    local_text = " ".join(left + [normalized_old] + right)

    if local & PLURAL_HINTS and normalized_new in SINGULAR_BAD_WITH_PLURAL_CONTEXT:
        return False
    if immediate_left in {"good", "great"}:
        return normalized_new in GOOD_SLOT_ALLOWED
    if immediate_left in {"painless", "easy", "fast", "slow", "loud"} and normalized_new in ADJECTIVE_SLOT_BLOCKLIST:
        return False
    if immediate_left in {"a", "an"} and normalized_new in BAD_ARTICLE_ASPECTS:
        return False
    if immediate_left in {"knowledgeable", "helpful", "rude", "friendly"}:
        return normalized_new in PERSON_OR_SERVICE_SLOT_ALLOWED
    if immediate_right in {"works", "work", "working", "running", "connected"} and normalized_new in ACTIVITY_SLOT_BLOCKLIST:
        return False
    if immediate_right in {"crapped", "died", "crashed", "crashes", "crashing"} and normalized_new in CRASH_SLOT_BLOCKLIST:
        return False
    for quantity_term, allowed_aspects in QUANTITY_SLOT_ALLOWED_BY_RIGHT.items():
        if quantity_term in local:
            return normalized_new in allowed_aspects
    if "10 hour" in local_text or "10 hours" in local_text:
        return normalized_new in QUANTITY_SLOT_ALLOWED_BY_RIGHT["hour"]
    if {"paying", "paid", "pay", "purchase", "purchased", "bought"} & local and normalized_new in PURCHASE_OR_PAYMENT_SLOT_BLOCKLIST:
        return False
    if {"country", "overseas"} & local and normalized_new in COUNTRY_SLOT_BLOCKLIST:
        return False
    if normalized_old in PERSON_OR_SERVICE_ASPECTS and normalized_new not in PERSON_OR_SERVICE_SLOT_ALLOWED:
        return False
    return True


def rank_replacement_aspects(
    text: str,
    old_aspect: str,
    opinion: str,
    sentiment: str,
    aspect_bank: list[str],
    domain_memory: dict | None = None,
) -> list[dict]:
    memory = domain_memory or {}
    aspect_types = memory.get("aspect_types", {})
    opinion_aspect_counts = memory.get("opinion_aspect_counts", {})
    target_triplet_counts = memory.get("target_triplet_counts", {})
    old_type = aspect_types.get(old_aspect, infer_aspect_type(old_aspect))
    opinion_key = f"{opinion}|{sentiment}"
    opinion_hint_types = {
        aspect_type
        for aspect_type, hints in OPINION_TYPE_HINTS.items()
        if _normalize_fragment(opinion) in hints
    }
    ranked = []
    for aspect in aspect_bank:
        if aspect == old_aspect:
            continue
        if not aspect_replacement_compatible(text, old_aspect, aspect):
            continue
        new_type = aspect_types.get(aspect, infer_aspect_type(aspect))
        opinion_aspect_count = int(opinion_aspect_counts.get(opinion_key, {}).get(aspect, 0))
        target_triplet_count = int(target_triplet_counts.get(f"{aspect}|{opinion}|{sentiment}", 0))
        type_match = new_type == old_type and new_type != "other"
        opinion_type_match = new_type in opinion_hint_types
        score = 1.0
        score += 2.5 * min(opinion_aspect_count, 3)
        score += 3.0 * min(target_triplet_count, 2)
        if type_match:
            score += 1.2
        if opinion_type_match:
            score += 1.0
        if new_type == "other":
            score -= 0.5
        ranked.append(
            {
                "aspect": aspect,
                "score": round(score, 6),
                "features": {
                    "old_type": old_type,
                    "new_type": new_type,
                    "type_match": type_match,
                    "opinion_type_match": opinion_type_match,
                    "opinion_aspect_count": opinion_aspect_count,
                    "target_triplet_count": target_triplet_count,
                },
            }
        )
    return sorted(ranked, key=lambda item: (item["score"], item["aspect"]), reverse=True)


def rank_replacement_opinions(
    aspect: str,
    old_opinion: str,
    sentiment: str,
    opinion_bank: dict[str, list[str]],
    domain_memory: dict | None = None,
) -> list[dict]:
    memory = domain_memory or {}
    opinion_counts = Counter(memory.get("opinion_counts_by_sentiment", {}).get(sentiment, {}))
    opinion_aspect_counts = memory.get("opinion_aspect_counts", {})
    target_triplet_counts = memory.get("target_triplet_counts", {})
    candidates = opinion_bank.get(sentiment, [])
    ranked = []
    normalized_old = _normalize_fragment(old_opinion)
    for opinion in candidates:
        normalized_opinion = _normalize_fragment(opinion)
        if not normalized_opinion or normalized_opinion == normalized_old:
            continue
        if not _valid_opinion(opinion):
            continue
        opinion_key = f"{opinion}|{sentiment}"
        opinion_count = int(opinion_counts.get(opinion, 0))
        opinion_aspect_count = int(opinion_aspect_counts.get(opinion_key, {}).get(aspect, 0))
        target_triplet_count = int(target_triplet_counts.get(f"{aspect}|{opinion}|{sentiment}", 0))
        lexical_similarity = _token_jaccard(old_opinion, opinion)
        score = 1.0
        score += 2.5 * min(opinion_aspect_count, 3)
        score += 3.0 * min(target_triplet_count, 2)
        score += 0.5 * min(opinion_count, 4)
        score += lexical_similarity
        ranked.append(
            {
                "opinion": opinion,
                "sentiment": sentiment,
                "score": round(score, 6),
                "features": {
                    "opinion_count": opinion_count,
                    "opinion_aspect_count": opinion_aspect_count,
                    "target_triplet_count": target_triplet_count,
                    "lexical_similarity": round(lexical_similarity, 6),
                },
            }
        )
    return sorted(ranked, key=lambda item: (item["score"], item["opinion"]), reverse=True)


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    if width == 0:
        return []
    return [sum(vector[idx] for vector in vectors if len(vector) == width) / len(vectors) for idx in range(width)]


def build_sentiment_centroids(
    opinion_bank: dict[str, list[str]],
    opinion_embeddings: dict[str, list[float]],
) -> dict[str, list[float]]:
    centroids: dict[str, list[float]] = {}
    normalized_embeddings = {
        _normalize_fragment(opinion): vector for opinion, vector in opinion_embeddings.items()
    }
    for sentiment, opinions in opinion_bank.items():
        vectors = [
            normalized_embeddings[_normalize_fragment(opinion)]
            for opinion in opinions
            if _normalize_fragment(opinion) in normalized_embeddings
        ]
        centroid = _mean_vector(vectors)
        if centroid:
            centroids[sentiment] = centroid
    return centroids


def rank_sentiment_vector_replacement_opinions(
    aspect: str,
    old_opinion: str,
    sentiment: str,
    opinion_bank: dict[str, list[str]],
    domain_memory: dict | None = None,
    min_margin: float = 0.05,
    use_polarity_axis: bool = False,
    min_old_similarity: float = 0.35,
    no_cooccurrence_min_similarity: float = 0.50,
) -> list[dict]:
    memory = domain_memory or {}
    opinion_embeddings = {
        _normalize_fragment(opinion): vector
        for opinion, vector in (memory.get("opinion_embeddings") or {}).items()
    }
    if not opinion_embeddings:
        return []
    centroids = memory.get("sentiment_centroids") or build_sentiment_centroids(opinion_bank, opinion_embeddings)
    target_centroid = centroids.get(sentiment)
    polarity_axis = memory.get("sentiment_polarity_axis") or []
    polarity_thresholds = memory.get("sentiment_polarity_thresholds") or {}
    old_vector = opinion_embeddings.get(_normalize_fragment(old_opinion))
    if not target_centroid:
        return []

    opinion_counts = Counter(memory.get("opinion_counts_by_sentiment", {}).get(sentiment, {}))
    opinion_aspect_counts = memory.get("opinion_aspect_counts", {})
    target_triplet_counts = memory.get("target_triplet_counts", {})
    ranked = []
    normalized_old = _normalize_fragment(old_opinion)
    for opinion in opinion_bank.get(sentiment, []):
        normalized_opinion = _normalize_fragment(opinion)
        if not normalized_opinion or normalized_opinion == normalized_old:
            continue
        if not _valid_opinion(opinion):
            continue
        candidate_vector = opinion_embeddings.get(normalized_opinion)
        if not candidate_vector:
            continue
        target_similarity = _cosine_similarity(candidate_vector, target_centroid)
        other_similarity = max(
            [
                _cosine_similarity(candidate_vector, centroid)
                for other_sentiment, centroid in centroids.items()
                if other_sentiment != sentiment
            ]
            or [0.0]
        )
        sentiment_margin = target_similarity - other_similarity
        polarity_score = _cosine_similarity(candidate_vector, polarity_axis) if polarity_axis else 0.0
        polarity_passed = (
            (sentiment == "pos" and polarity_score >= float(polarity_thresholds.get("pos", 0.0)))
            or (sentiment == "neg" and polarity_score <= float(polarity_thresholds.get("neg", 0.0)))
            or (sentiment == "neu" and abs(polarity_score) <= float(polarity_thresholds.get("neu_abs", 0.15)))
        )
        if use_polarity_axis and (not polarity_axis or not polarity_passed):
            continue
        if not use_polarity_axis and sentiment_margin < min_margin:
            continue
        old_similarity = _cosine_similarity(candidate_vector, old_vector) if old_vector else 0.0
        opinion_key = f"{opinion}|{sentiment}"
        opinion_count = int(opinion_counts.get(opinion, 0))
        opinion_aspect_count = int(opinion_aspect_counts.get(opinion_key, {}).get(aspect, 0))
        target_triplet_count = int(target_triplet_counts.get(f"{aspect}|{opinion}|{sentiment}", 0))
        required_old_similarity = (
            min_old_similarity if opinion_aspect_count > 0 or target_triplet_count > 0
            else no_cooccurrence_min_similarity
        )
        if use_polarity_axis and (not old_vector or old_similarity < required_old_similarity):
            continue
        score = 1.5 * target_similarity
        score += 1.0 * sentiment_margin
        score += 0.8 * old_similarity
        score += 0.5 * min(opinion_count, 4)
        score += 2.0 * min(opinion_aspect_count, 3)
        score += 2.5 * min(target_triplet_count, 2)
        ranked.append(
            {
                "opinion": opinion,
                "sentiment": sentiment,
                "score": round(score, 6),
                "features": {
                    "target_similarity": round(target_similarity, 6),
                    "other_similarity": round(other_similarity, 6),
                    "sentiment_margin": round(sentiment_margin, 6),
                    "polarity_score": round(polarity_score, 6),
                    "polarity_passed": polarity_passed,
                    "old_similarity": round(old_similarity, 6),
                    "required_old_similarity": round(required_old_similarity, 6),
                    "opinion_count": opinion_count,
                    "opinion_aspect_count": opinion_aspect_count,
                    "target_triplet_count": target_triplet_count,
                },
            }
        )
    return sorted(ranked, key=lambda item: (item["score"], item["opinion"]), reverse=True)


def build_augmentation_requests(
    source_rows: list[dict],
    pseudo_rows: list[dict],
    per_row: int,
    seed: int,
    prompt_style: str = "concept",
    domain_memory: dict | None = None,
    channel_mode: str = "all",
    label_embeddings: dict[str, list[float]] | None = None,
    label_similarity_top_k: int = 4,
    composition_source_rows: list[dict] | None = None,
    target_domain_name: str = "",
    domain_prefix_style: str = "none",
    opinion_replacement_mode: str = "coupled_random",
    sentiment_vector_min_margin: float = 0.05,
    sentiment_vector_use_polarity_axis: bool = False,
    sentiment_vector_min_old_similarity: float = 0.35,
    sentiment_vector_no_cooccurrence_min_similarity: float = 0.50,
) -> list[dict]:
    """Build C3DA-style generation prompts for ASTE augmentation.

    Channel 1 appends a target-domain aspect to the source sentence.
    Channel 2 appends an opinion-sentiment hint, never sentiment alone.
    """
    rng = random.Random(seed)
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(
            "prompt_style must be one of concept, legacy, masked_mutual, rewrite_aspect, "
            "label_composition, label_to_text, rsda_t5_label_composition, sentence_fusion_composition"
        )
    if channel_mode not in CHANNEL_MODES:
        raise ValueError("channel_mode must be one of all, aspect, opinion")
    if opinion_replacement_mode not in OPINION_REPLACEMENT_MODES:
        raise ValueError("opinion_replacement_mode must be one of coupled_random, semantic_same_sentiment, sentiment_vector")
    memory = domain_memory or build_domain_memory(pseudo_rows)
    domain_prefix = format_domain_prefix(target_domain_name, domain_prefix_style)
    aspect_bank = (
        memory.get("target_aspects")
        or memory.get("core_target_aspects")
        or memory.get("aspects")
        or build_target_aspect_bank(pseudo_rows)
    )
    opinion_bank = (
        memory.get("candidate_opinions_by_sentiment")
        or memory.get("opinions_by_sentiment")
        or build_opinion_sentiment_bank(source_rows, pseudo_rows)
    )
    preferred_opinion_bank = {sentiment: list(opinions) for sentiment, opinions in opinion_bank.items()}
    merged_opinion_bank = build_opinion_sentiment_bank(source_rows, pseudo_rows)
    for sentiment, opinions in opinion_bank.items():
        merged = set(merged_opinion_bank.get(sentiment, []))
        merged.update(opinions)
        merged_opinion_bank[sentiment] = sorted(merged)
    opinion_bank = merged_opinion_bank
    aspect_bank = list(aspect_bank)
    requests: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add_request(
        base: dict,
        triplets: list[tuple[str, str, str]],
        channel: str,
        prompt: str,
        old_triplet: tuple[str, str, str] | None = None,
        new_triplet: tuple[str, str, str] | None = None,
        replacement_rank: dict | None = None,
    ) -> None:
        label = canonicalize_triplet_text(triplets_to_text(triplets))
        if not label:
            return
        key = (prompt.lower(), label.lower(), channel)
        if key in seen:
            return
        seen.add(key)
        request = {
            "input": prompt,
            "label": label,
            "channel": channel,
            "base_text": base["text"],
            "base_id": base.get("id"),
            "prompt_style": prompt_style,
        }
        if domain_prefix:
            request["domain_name"] = target_domain_name
            request["domain_prefix_style"] = domain_prefix_style
            request["domain_prefix"] = domain_prefix
        if old_triplet is not None and new_triplet is not None:
            request["old_triplet"] = list(old_triplet)
            request["new_triplet"] = list(new_triplet)
        if replacement_rank is not None:
            request["replacement_rank"] = replacement_rank
        requests.append(request)

    if prompt_style == "sentence_fusion_composition":
        source_rows_for_composition = composition_source_rows or []
        candidates: list[tuple[dict, tuple[str, str, str]]] = []
        for row in source_rows_for_composition:
            text = row.get("text", "")
            triplets = sorted(parse_triplet_text(row.get("label", "")))
            if len(triplets) != 1:
                continue
            triplet = triplets[0]
            if not _valid_label_composition_triplet(text, triplet):
                continue
            candidates.append((row, triplet))
        max_requests = max(1, len(source_rows_for_composition) * max(1, per_row) // 2)
        attempts = 0
        while len(requests) < max_requests and attempts < max_requests * 20:
            attempts += 1
            if len(candidates) < 2:
                break
            left_row, left_triplet = rng.choice(candidates)
            right_row, right_triplet = rng.choice(candidates)
            if left_row.get("id") == right_row.get("id") and left_row.get("text") == right_row.get("text"):
                continue
            if left_triplet[0] == right_triplet[0]:
                continue
            new_triplets = sorted({left_triplet, right_triplet})
            if len(new_triplets) < 2:
                continue
            prompt = apply_domain_prefix(
                build_sentence_fusion_prompt(left_row.get("text", ""), right_row.get("text", ""), new_triplets),
                target_domain_name,
                domain_prefix_style,
            )
            base_text = f"{left_row.get('text', '')} {right_row.get('text', '')}".strip()
            before_count = len(requests)
            add_request(
                {
                    "text": base_text,
                    "id": f"{left_row.get('id', left_row.get('text', ''))}|{right_row.get('id', right_row.get('text', ''))}",
                },
                new_triplets,
                "sentence_fusion_composition_channel",
                prompt,
            )
            if len(requests) > before_count:
                requests[-1]["new_triplets"] = [list(triplet) for triplet in new_triplets]
                requests[-1]["source_texts"] = [left_row.get("text", ""), right_row.get("text", "")]
        return requests

    if prompt_style in {"label_composition", "label_to_text", "rsda_t5_label_composition"}:
        candidates: list[tuple[dict, tuple[str, str, str]]] = []
        for row in pseudo_rows:
            if float(row.get("sample_weight", 0.0)) < 0.6:
                continue
            text = row.get("text", "")
            for triplet in sorted(parse_triplet_text(row.get("label", ""))):
                if not _valid_label_composition_triplet(text, triplet):
                    continue
                candidates.append((row, triplet))
        if len(candidates) < 2:
            return requests
        if prompt_style == "label_to_text":
            max_requests = max(1, len(pseudo_rows) * max(1, per_row))
            for row, triplet in candidates:
                if len(requests) >= max_requests:
                    break
                prompt = apply_domain_prefix(
                    build_label_composition_prompt([triplet]),
                    target_domain_name,
                    domain_prefix_style,
                )
                add_request(
                    row,
                    [triplet],
                    "label_to_text_channel",
                    prompt,
                )
                if requests:
                    requests[-1]["new_triplets"] = [list(triplet)]
            return requests
        if prompt_style == "rsda_t5_label_composition":
            max_requests = max(1, len(pseudo_rows) * max(1, per_row))
            ranked_pairs: list[tuple[float, str, int, int]] = []
            for left_idx, (left_row, left_triplet) in enumerate(candidates):
                left_label = _label_text_for_triplet(left_triplet)
                left_embedding = (label_embeddings or {}).get(left_label)
                if left_embedding is None:
                    continue
                scored: list[tuple[float, str, int]] = []
                for right_idx, (right_row, right_triplet) in enumerate(candidates):
                    if left_idx == right_idx:
                        continue
                    if left_row.get("id") == right_row.get("id") and left_row.get("text") == right_row.get("text"):
                        continue
                    if left_triplet[0] == right_triplet[0]:
                        continue
                    right_label = _label_text_for_triplet(right_triplet)
                    right_embedding = (label_embeddings or {}).get(right_label)
                    if right_embedding is None:
                        continue
                    similarity = _cosine_similarity(left_embedding, right_embedding)
                    if not rsda_composition_pair_compatible(left_triplet, right_triplet, similarity):
                        continue
                    pair_key = "|".join(sorted([left_label, right_label]))
                    scored.append((similarity, pair_key, right_idx))
                for similarity, pair_key, right_idx in sorted(scored, reverse=True)[: max(1, label_similarity_top_k)]:
                    ranked_pairs.append((similarity, pair_key, left_idx, right_idx))
            for similarity, _pair_key, left_idx, right_idx in sorted(ranked_pairs, reverse=True):
                if len(requests) >= max_requests:
                    break
                left_row, left_triplet = candidates[left_idx]
                right_row, right_triplet = candidates[right_idx]
                new_triplets = sorted({left_triplet, right_triplet})
                if len(new_triplets) < 2:
                    continue
                prompt = apply_domain_prefix(
                    build_label_composition_prompt(new_triplets),
                    target_domain_name,
                    domain_prefix_style,
                )
                base_text = f"{left_row.get('text', '')} {right_row.get('text', '')}".strip()
                before_count = len(requests)
                add_request(
                    {
                        "text": base_text,
                        "id": f"{left_row.get('id', left_row.get('text', ''))}|{right_row.get('id', right_row.get('text', ''))}",
                    },
                    new_triplets,
                    "rsda_t5_label_composition_channel",
                    prompt,
                )
                if len(requests) > before_count:
                    requests[-1]["new_triplets"] = [list(triplet) for triplet in new_triplets]
                    requests[-1]["label_similarity"] = round(similarity, 6)
            return requests
        max_requests = max(1, len(pseudo_rows) * max(1, per_row))
        attempts = 0
        while len(requests) < max_requests and attempts < max_requests * 12:
            attempts += 1
            left_row, left_triplet = rng.choice(candidates)
            right_row, right_triplet = rng.choice(candidates)
            if left_row.get("id") == right_row.get("id") and left_row.get("text") == right_row.get("text"):
                continue
            if left_triplet[0] == right_triplet[0]:
                continue
            new_triplets = sorted({left_triplet, right_triplet})
            prompt = apply_domain_prefix(
                build_label_composition_prompt(new_triplets),
                target_domain_name,
                domain_prefix_style,
            )
            base_text = f"{left_row.get('text', '')} {right_row.get('text', '')}".strip()
            before_count = len(requests)
            add_request(
                {
                    "text": base_text,
                    "id": f"{left_row.get('id', left_row.get('text', ''))}|{right_row.get('id', right_row.get('text', ''))}",
                },
                new_triplets,
                "label_composition_channel",
                prompt,
            )
            if len(requests) > before_count:
                requests[-1]["new_triplets"] = [list(triplet) for triplet in new_triplets]
        return requests

    for row in pseudo_rows:
        triplets = sorted(parse_triplet_text(row.get("label", "")))
        if not triplets:
            continue
        for _ in range(max(1, per_row)):
            # Aspect-aware channel: replace one aspect with a target-domain aspect.
            if channel_mode in {"all", "aspect"} and aspect_bank:
                new_triplets = list(triplets)
                idx = rng.randrange(len(new_triplets))
                old_triplet = new_triplets[idx]
                _aspect, opinion, sentiment = old_triplet
                ranked_aspects = rank_replacement_aspects(
                    text=row["text"],
                    old_aspect=_aspect,
                    opinion=opinion,
                    sentiment=sentiment,
                    aspect_bank=aspect_bank,
                    domain_memory=memory,
                )
                if not ranked_aspects:
                    continue
                top_k = ranked_aspects[: min(3, len(ranked_aspects))]
                replacement_rank = rng.choice(top_k)
                new_aspect = replacement_rank["aspect"]
                new_triplet = (new_aspect, opinion, sentiment)
                new_triplets[idx] = new_triplet
                if prompt_style == "rewrite_aspect":
                    prompt = apply_domain_prefix(
                        build_aspect_rewrite_prompt(row["text"], old_triplet, new_triplet),
                        target_domain_name,
                        domain_prefix_style,
                    )
                    channel = "rewrite_aspect_channel"
                elif prompt_style == "masked_mutual":
                    prompt = build_masked_aspect_prompt(
                        row["text"],
                        old_triplet,
                        new_triplet,
                        target_domain_name,
                        domain_prefix_style,
                    )
                    channel = "masked_aspect_channel"
                elif prompt_style == "legacy":
                    prompt = apply_domain_prefix(
                        build_legacy_label_invariant_prompt(canonicalize_triplet_text(triplets_to_text(new_triplets))),
                        target_domain_name,
                        domain_prefix_style,
                    )
                    channel = "aspect_channel"
                else:
                    prompt = apply_domain_prefix(
                        build_concept_prompt(canonicalize_triplet_text(triplets_to_text(new_triplets))),
                        target_domain_name,
                        domain_prefix_style,
                    )
                    channel = "aspect_channel"
                add_request(row, new_triplets, channel, prompt, old_triplet, new_triplet, replacement_rank)

            # Opinion-sentiment channel: replace opinion and sentiment as a coupled pair.
            sentiment_choices = [s for s, opinions in opinion_bank.items() if opinions]
            if channel_mode in {"all", "opinion"} and sentiment_choices:
                new_triplets = list(triplets)
                idx = rng.randrange(len(new_triplets))
                old_triplet = new_triplets[idx]
                aspect, _opinion, _sentiment = old_triplet
                replacement_rank = None
                if opinion_replacement_mode in {"semantic_same_sentiment", "sentiment_vector"}:
                    if opinion_replacement_mode == "sentiment_vector":
                        ranked_opinions = rank_sentiment_vector_replacement_opinions(
                            aspect=aspect,
                            old_opinion=_opinion,
                            sentiment=_sentiment,
                            opinion_bank=preferred_opinion_bank,
                            domain_memory=memory,
                            min_margin=sentiment_vector_min_margin,
                            use_polarity_axis=sentiment_vector_use_polarity_axis,
                            min_old_similarity=sentiment_vector_min_old_similarity,
                            no_cooccurrence_min_similarity=sentiment_vector_no_cooccurrence_min_similarity,
                        )
                    else:
                        ranked_opinions = rank_replacement_opinions(
                            aspect=aspect,
                            old_opinion=_opinion,
                            sentiment=_sentiment,
                            opinion_bank=preferred_opinion_bank,
                            domain_memory=memory,
                        )
                    if not ranked_opinions:
                        if opinion_replacement_mode == "sentiment_vector":
                            ranked_opinions = rank_sentiment_vector_replacement_opinions(
                                aspect=aspect,
                                old_opinion=_opinion,
                                sentiment=_sentiment,
                                opinion_bank=opinion_bank,
                                domain_memory=memory,
                                min_margin=sentiment_vector_min_margin,
                                use_polarity_axis=sentiment_vector_use_polarity_axis,
                                min_old_similarity=sentiment_vector_min_old_similarity,
                                no_cooccurrence_min_similarity=sentiment_vector_no_cooccurrence_min_similarity,
                            )
                        else:
                            ranked_opinions = rank_replacement_opinions(
                                aspect=aspect,
                                old_opinion=_opinion,
                                sentiment=_sentiment,
                                opinion_bank=opinion_bank,
                                domain_memory=memory,
                            )
                    if not ranked_opinions:
                        continue
                    top_k = ranked_opinions[: min(3, len(ranked_opinions))]
                    replacement_rank = rng.choice(top_k)
                    new_opinion = replacement_rank["opinion"]
                    new_sentiment = replacement_rank["sentiment"]
                else:
                    preferred_opinion, preferred_sentiment = _choose_replacement_opinion(
                        rng, preferred_opinion_bank, _opinion, _sentiment
                    )
                    if (preferred_opinion, preferred_sentiment) != (_opinion, _sentiment):
                        new_opinion, new_sentiment = preferred_opinion, preferred_sentiment
                    else:
                        new_opinion, new_sentiment = _choose_replacement_opinion(rng, opinion_bank, _opinion, _sentiment)
                    if (new_opinion, new_sentiment) == (_opinion, _sentiment):
                        same_sentiment_candidates = [
                            opinion for opinion in opinion_bank.get(_sentiment, []) if opinion != _opinion
                        ]
                        if same_sentiment_candidates:
                            new_opinion = rng.choice(sorted(set(same_sentiment_candidates)))
                        else:
                            continue
                new_triplet = (aspect, new_opinion, new_sentiment)
                new_triplets[idx] = new_triplet
                if prompt_style == "masked_mutual":
                    prompt = build_masked_opinion_sentiment_prompt(
                        row["text"],
                        old_triplet,
                        new_triplet,
                        target_domain_name,
                        domain_prefix_style,
                    )
                    channel = "masked_opinion_sentiment_channel"
                elif prompt_style == "legacy":
                    prompt = apply_domain_prefix(
                        build_legacy_label_invariant_prompt(canonicalize_triplet_text(triplets_to_text(new_triplets))),
                        target_domain_name,
                        domain_prefix_style,
                    )
                    channel = "opinion_sentiment_channel"
                else:
                    prompt = apply_domain_prefix(
                        build_concept_prompt(canonicalize_triplet_text(triplets_to_text(new_triplets))),
                        target_domain_name,
                        domain_prefix_style,
                    )
                    channel = "opinion_sentiment_channel"
                before_count = len(requests)
                add_request(row, new_triplets, channel, prompt, old_triplet, new_triplet, replacement_rank)
                if len(requests) > before_count:
                    requests[-1]["opinion_replacement_mode"] = opinion_replacement_mode
                    if replacement_rank is not None:
                        requests[-1]["opinion_replacement_rank"] = replacement_rank

    return requests


def build_generator_training_rows(
    rows: list[dict],
    seed: int,
    prompt_style: str = "concept",
    channel_mode: str = "all",
    domain_name: str = "",
    domain_prefix_style: str = "none",
) -> list[dict]:
    train_rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    rng = random.Random(seed)
    domain_prefix = format_domain_prefix(domain_name, domain_prefix_style)
    if prompt_style not in PROMPT_STYLES:
        raise ValueError("prompt_style must be one of concept, legacy, masked_mutual, rewrite_aspect, label_composition, label_to_text")
    if channel_mode not in CHANNEL_MODES:
        raise ValueError("channel_mode must be one of all, aspect, opinion")
    aspect_pool = [
        (aspect, sentiment)
        for row in rows
        for aspect, _opinion, sentiment in parse_triplet_text(row.get("label", ""))
        if _valid_aspect(aspect)
    ]
    opinion_bank: dict[str, list[str]] = build_opinion_sentiment_bank(rows, rows)
    for row in rows:
        label = canonicalize_triplet_text(row.get("label", ""))
        if not label:
            continue
        if prompt_style in {"label_composition", "label_to_text"}:
            triplets = sorted(parse_triplet_text(label))
            if not triplets:
                continue
            prompt = apply_domain_prefix(build_label_composition_prompt(triplets), domain_name, domain_prefix_style)
            channel = "label_to_text_generator" if prompt_style == "label_to_text" else "label_composition_generator"
            key = (prompt.lower(), row["text"].lower(), channel)
            if key in seen:
                continue
            seen.add(key)
            train_row = {
                "input": prompt,
                "target": row["text"],
                "label": label,
                "channel": channel,
                "new_triplets": [list(triplet) for triplet in triplets],
            }
            if domain_prefix:
                train_row["domain_name"] = domain_name
                train_row["domain_prefix_style"] = domain_prefix_style
                train_row["domain_prefix"] = domain_prefix
            train_rows.append(train_row)
            continue
        if prompt_style in {"masked_mutual", "rewrite_aspect"}:
            for triplet in parse_triplet_text(label):
                aspect, opinion, sentiment = triplet
                channel_rows = []
                if channel_mode in {"all", "aspect"}:
                    new_aspect = _choose_replacement_aspect(rng, aspect_pool, aspect, sentiment)
                    aspect_triplet = (new_aspect, opinion, sentiment)
                    if prompt_style == "rewrite_aspect":
                        aspect_prompt = apply_domain_prefix(
                            build_aspect_rewrite_prompt(row["text"], triplet, aspect_triplet),
                            domain_name,
                            domain_prefix_style,
                        )
                        aspect_channel = "rewrite_aspect_editor"
                    else:
                        aspect_prompt = build_masked_aspect_prompt(
                            row["text"],
                            triplet,
                            aspect_triplet,
                            domain_name,
                            domain_prefix_style,
                        )
                        aspect_channel = "masked_aspect_editor"
                    aspect_target = _replace_first_fragment(row["text"], aspect, new_aspect)
                    channel_rows.append((aspect_prompt, aspect_target, aspect_channel, triplet, aspect_triplet))
                for prompt, target, channel, old_triplet, new_triplet in channel_rows:
                    key = (prompt.lower(), target.lower(), channel)
                    if key in seen:
                        continue
                    seen.add(key)
                    new_label = canonicalize_triplet_text(triplets_to_text([new_triplet]))
                    train_row = {
                        "input": prompt,
                        "target": target,
                        "label": new_label,
                        "channel": channel,
                        "old_triplet": list(old_triplet),
                        "new_triplet": list(new_triplet),
                    }
                    if domain_prefix:
                        train_row["domain_name"] = domain_name
                        train_row["domain_prefix_style"] = domain_prefix_style
                        train_row["domain_prefix"] = domain_prefix
                    train_rows.append(train_row)
                if channel_mode in {"all", "opinion"} and prompt_style == "masked_mutual":
                    new_opinion, new_sentiment = _choose_replacement_opinion(rng, opinion_bank, opinion, sentiment)
                    opinion_triplet = (aspect, new_opinion, new_sentiment)
                    opinion_prompt = build_masked_opinion_sentiment_prompt(
                        row["text"],
                        triplet,
                        opinion_triplet,
                        domain_name,
                        domain_prefix_style,
                    )
                    opinion_target = _replace_first_fragment(row["text"], opinion, new_opinion)
                    if opinion_target == row["text"] and new_opinion != opinion:
                        continue
                    key = (opinion_prompt.lower(), opinion_target.lower(), "masked_opinion_sentiment_editor")
                    if key in seen:
                        continue
                    seen.add(key)
                    train_row = {
                        "input": opinion_prompt,
                        "target": opinion_target,
                        "label": canonicalize_triplet_text(triplets_to_text([opinion_triplet])),
                        "channel": "masked_opinion_sentiment_editor",
                        "old_triplet": list(triplet),
                        "new_triplet": list(opinion_triplet),
                    }
                    if domain_prefix:
                        train_row["domain_name"] = domain_name
                        train_row["domain_prefix_style"] = domain_prefix_style
                        train_row["domain_prefix"] = domain_prefix
                    train_rows.append(train_row)
        else:
            if prompt_style == "legacy":
                prompt = apply_domain_prefix(build_legacy_label_invariant_prompt(label), domain_name, domain_prefix_style)
            else:
                prompt = apply_domain_prefix(build_concept_prompt(label), domain_name, domain_prefix_style)
            key = (prompt.lower(), row["text"].lower())
            if key in seen:
                continue
            seen.add(key)
            train_row = {
                "input": prompt,
                "target": row["text"],
                "label": label,
                "channel": "label_to_text_generation",
            }
            if domain_prefix:
                train_row["domain_name"] = domain_name
                train_row["domain_prefix_style"] = domain_prefix_style
                train_row["domain_prefix"] = domain_prefix
            train_rows.append(train_row)
    return train_rows


def is_consistent_with_label(text: str, label: str) -> bool:
    lowered = text.lower()
    triplets = parse_triplet_text(label)
    if not triplets:
        return False
    for aspect, opinion, _sentiment in triplets:
        if aspect not in lowered or opinion not in lowered:
            return False
    return True
