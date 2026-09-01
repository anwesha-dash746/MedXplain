"""Clinical-note preprocessing.

Every transformation here is deliberately *conservative and documented*, because
this module is the "Data & Preprocessing" section of the paper in executable
form. Two rules govern the design:

1.  **No lossy normalisation before explanation.** The text the model sees is
    the text the UI highlights, so we never silently delete content that a
    clinician would expect to see highlighted. Cleaning is limited to
    de-identification placeholders, whitespace, and header artefacts.
2.  **Optional steps stay optional.** Abbreviation expansion changes token
    boundaries and therefore explanations, so it is off by default and reported
    as an ablation rather than baked in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from medxplain.data.clinical_lexicon import ABBREVIATIONS

# MIMIC-style de-identification placeholders, e.g. [**2145-3-6**], [**Hospital1 18**]
DEID_PATTERN = re.compile(r"\[\*\*(.*?)\*\*\]", flags=re.DOTALL)
# category of the placeholder, used to pick a readable replacement token
_DEID_CATEGORIES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$"), "<DATE>"),
    (re.compile(r"date", re.I), "<DATE>"),
    (re.compile(r"hospital|location|address|state|country|university", re.I), "<PLACE>"),
    (re.compile(r"name|doctor|attending|initial", re.I), "<NAME>"),
    (re.compile(r"telephone|pager|number|numeric", re.I), "<ID>"),
    (re.compile(r"age", re.I), "<AGE>"),
)

#: Administrative header lines that carry no clinical signal but consume a large
#: share of the encoder's token budget (roughly 60 wordpieces in a MIMIC-style
#: discharge summary). Dropping them is a documented preprocessing decision, not
#: a silent one: it is what buys the truncation window enough room to reach the
#: hospital course.
HEADER_LINE = re.compile(
    r"^\s*(admission date|discharge date|date of birth|sex|service|allergies|"
    r"attending|dictated by|medquist|job#|signed electronically by)\s*:.*$",
    flags=re.IGNORECASE | re.MULTILINE,
)

MULTISPACE = re.compile(r"[ \t]{2,}")
MULTINEWLINE = re.compile(r"\n{3,}")
UNDERSCORE_RULE = re.compile(r"_{3,}")
NON_PRINTABLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass
class PreprocessConfig:
    """Toggles for :func:`clean_note`, serialised into every results file."""

    replace_deid: bool = True
    collapse_whitespace: bool = True
    strip_rule_lines: bool = True
    strip_header: bool = True
    lowercase: bool = False
    expand_abbreviations: bool = False
    max_chars: int | None = None


def _deid_token(inner: str) -> str:
    for pattern, token in _DEID_CATEGORIES:
        if pattern.search(inner.strip()):
            return token
    return "<DEID>"


def replace_deid_placeholders(text: str) -> str:
    """Turn ``[**Hospital1 18**]`` into ``<PLACE>`` etc.

    Kept as a single token rather than deleted: the placeholder occupies a real
    position in the clinical narrative, and removing it would shift every
    downstream character offset used by the explanation payload.
    """
    return DEID_PATTERN.sub(lambda m: _deid_token(m.group(1)), text)


def normalise_abbreviations(text: str, table: dict[str, str] | None = None) -> str:
    """Expand common clinical abbreviations (whole-token, case-insensitive).

    Off by default in :class:`PreprocessConfig`; exposed so the effect of
    abbreviation expansion can be reported as an ablation.
    """
    table = table or ABBREVIATIONS
    # longest first so 'dm2' wins over a hypothetical 'dm'
    keys = sorted(table, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(" + "|".join(re.escape(k) for k in keys) + r")(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    return pattern.sub(lambda m: table[m.group(1).lower()], text)


def clean_note(text: str, cfg: PreprocessConfig | None = None) -> str:
    """Apply the configured cleaning pipeline to a single note."""
    cfg = cfg or PreprocessConfig()
    out = NON_PRINTABLE.sub(" ", text)
    if cfg.replace_deid:
        out = replace_deid_placeholders(out)
    if cfg.strip_rule_lines:
        out = UNDERSCORE_RULE.sub(" ", out)
    if cfg.strip_header:
        out = HEADER_LINE.sub("", out)
    if cfg.expand_abbreviations:
        out = normalise_abbreviations(out)
    if cfg.collapse_whitespace:
        out = MULTISPACE.sub(" ", out)
        out = MULTINEWLINE.sub("\n\n", out)
        out = "\n".join(line.rstrip() for line in out.split("\n"))
        out = out.strip()
    if cfg.lowercase:
        out = out.lower()
    if cfg.max_chars is not None:
        out = out[: cfg.max_chars]
    return out


def offset_preserving_clean(text: str) -> str:
    """Cleaning variant that never changes string length.

    Used when gold rationale character spans must stay valid: replacements are
    padded so downstream offsets computed on the raw note remain correct.
    """

    def pad(match: re.Match[str]) -> str:
        token = _deid_token(match.group(1))
        span = match.end() - match.start()
        return token.ljust(span)[:span] if len(token) <= span else "X" * span

    return DEID_PATTERN.sub(pad, text)


# --------------------------------------------------------------------------------------
# Long-document handling
# --------------------------------------------------------------------------------------


def chunk_text(
    text: str,
    tokenizer,
    max_length: int = 512,
    stride: int = 128,
    max_chunks: int | None = None,
) -> list[dict]:
    """Split a note into overlapping windows of at most ``max_length`` tokens.

    Discharge summaries routinely exceed BERT's 512-token limit, and naive
    truncation drops the hospital course, which is exactly where the mortality
    signal lives. We use a sliding window with ``stride`` tokens of overlap so
    no sentence falls entirely in a seam; predictions over chunks are pooled at
    inference time (see :mod:`medxplain.models.transformer`).

    Returns a list of dicts with ``input_ids``, ``attention_mask``, and the
    character offsets of each chunk in the original string.
    """
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    # room for [CLS] and [SEP]
    body = max_length - 2
    step = max(1, body - stride)

    chunks: list[dict] = []
    start = 0
    while start < len(ids) or not chunks:
        window_ids = ids[start : start + body]
        window_offsets = offsets[start : start + body]
        if not window_ids:
            window_ids, window_offsets = [], []
        chunks.append(
            {
                "input_ids": [tokenizer.cls_token_id] + window_ids + [tokenizer.sep_token_id],
                "attention_mask": [1] * (len(window_ids) + 2),
                "offsets": [(0, 0)] + [tuple(o) for o in window_offsets] + [(0, 0)],
                "token_start": start,
            }
        )
        if start + body >= len(ids):
            break
        start += step
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
    return chunks


def note_length_stats(texts: list[str], tokenizer, sample: int = 500) -> dict:
    """Token-length distribution, reported in the paper's data section."""
    import numpy as np

    subset = texts[:sample]
    lengths = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in subset]
    arr = np.asarray(lengths)
    return {
        "n_sampled": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(arr.max()),
        "frac_over_512": float((arr > 512).mean()),
        "frac_over_256": float((arr > 256).mean()),
    }
