#%% [markdown]
# Akkadian Data Preprocessing Pipeline (Improved)
#
# - Focus: robust normalization + sentence-level alignment
# - Keeps multiple views (raw / normalized / tagged)
# - Avoids lossy Sumerogram replacement
#
# Run:
#   uv run python src/data_preprocessing.py --data-dir data --out-dir src/outputs --plot

#%%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import json
import re
import unicodedata

import numpy as np
import pandas as pd

#%%
# -----------------------------
# Config
# -----------------------------


@dataclass
class NormalizeConfig:
    unicode_form: str = "NFC"
    normalize_h: bool = True  # Ḫ/ḫ -> H/h for test compatibility
    normalize_subscripts: bool = True
    normalize_gaps: bool = True
    strip_editorial: bool = True
    replace_unknown_x: bool = True
    tag_determinatives: bool = False
    tag_sumerograms: bool = False


@dataclass
class AlignConfig:
    min_src_chars: int = 10
    min_tgt_chars: int = 5
    min_ratio: float = 0.3
    max_ratio: float = 3.0
    min_ratio_short: float = 0.2
    max_ratio_short: float = 5.0
    short_token_threshold: int = 6
    target_ratio: float = 1.5  # average tgt/src token ratio
    max_tgt_merge: int = 5
    max_src_merge: int = 3


@dataclass
class QualityConfig:
    min_quality: float = 0.0
    max_gap_ratio: float = 0.5
    max_unk_ratio: float = 0.3
    length_weight: float = 0.3
    gap_weight: float = 0.4
    unk_weight: float = 0.2
    number_penalty: float = 0.1
    annotated_bonus: float = 0.05


#%%
# -----------------------------
# Helpers
# -----------------------------


_SUBSCRIPT_MAP = str.maketrans({
    "\u2080": "0",
    "\u2081": "1",
    "\u2082": "2",
    "\u2083": "3",
    "\u2084": "4",
    "\u2085": "5",
    "\u2086": "6",
    "\u2087": "7",
    "\u2088": "8",
    "\u2089": "9",
    "\u2093": "x",  # subscript x
})


def _nfc(text: str, form: str) -> str:
    return unicodedata.normalize(form, text) if form else text


def _strip_diacritics(text: str) -> str:
    # Insight: use NFD to separate base+diacritics for ASCII-safe tags.
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_sumerogram(token: str) -> bool:
    # Heuristic: uppercase-only tokens (with dots/digits) in transliteration.
    has_alpha = any(ch.isalpha() for ch in token)
    return has_alpha and token == token.upper()


def _sumerogram_tag(token: str) -> str:
    base = _strip_diacritics(token)
    base = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")
    return f"<SUM_{base}>" if base else "<SUM>"


def _determinative_tag(token: str) -> str:
    # token like {d}, {ki}
    inner = token.strip("{}").strip()
    inner = _strip_diacritics(inner)
    inner = re.sub(r"[^A-Za-z0-9]+", "_", inner).strip("_")
    return f"<DET_{inner}>" if inner else "<DET>"


def _normalize_token_for_match(token: str) -> str:
    # Insight: align by removing orthographic variability.
    t = unicodedata.normalize("NFD", token)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.translate(_SUBSCRIPT_MAP)
    t = t.replace("-", "").replace(".", "")
    t = t.replace("{", "").replace("}", "")
    t = t.lower()
    return t


#%%
# -----------------------------
# Normalization
# -----------------------------


def normalize_transliteration(text: str, cfg: NormalizeConfig) -> str:
    text = _nfc(text, cfg.unicode_form)
    if cfg.normalize_h:
        # Ḫ (U+1E2A), ḫ (U+1E2B) -> H/h for test compatibility
        text = text.replace("\u1E2A", "H").replace("\u1E2B", "h")

    if cfg.normalize_subscripts:
        text = text.translate(_SUBSCRIPT_MAP)

    if cfg.normalize_gaps:
        # Normalize ellipsis to <big_gap>
        text = text.replace("\u2026", " <big_gap> ")
        text = re.sub(r"\.\.\.+", " <big_gap> ", text)

        # [x] or [x x] -> <gap>, [ ... ] -> <big_gap>
        def _gap_bracket(m: re.Match) -> str:
            inner = m.group(1).strip()
            if re.fullmatch(r"[xX\s]+", inner):
                return " <gap> "
            if "..." in inner or "\u2026" in inner or len(inner.split()) > 1:
                return " <big_gap> "
            return f" {inner} "

        text = re.sub(r"\[([^\]]+)\]", _gap_bracket, text)

    if cfg.strip_editorial:
        # Remove editorial markers but keep their content
        text = re.sub(r"[!?/]", " ", text)
        text = re.sub(r"<([^>]+)>", r" \1 ", text)
        text = re.sub(r"[\u02F9\u02FA]", "", text)  # half-brackets
        text = text.replace("[", " ").replace("]", " ")

    if cfg.replace_unknown_x:
        # Replace standalone x token with <unk_sign>
        text = re.sub(r"\bx\b", " <unk_sign> ", text)

    text = _collapse_ws(text)
    return text


def normalize_translation(text: str, cfg: NormalizeConfig) -> str:
    text = _nfc(text, cfg.unicode_form)
    if cfg.normalize_gaps:
        text = text.replace("\u2026", " <big_gap> ")
        text = re.sub(r"\.\.\.+", " <big_gap> ", text)
        text = re.sub(r"\[([^\]]*)\]", " <gap> ", text)

    if cfg.strip_editorial:
        text = re.sub(r"<([^>]+)>", r" \1 ", text)

    text = _collapse_ws(text)
    return text


def tag_tokens(text: str, cfg: NormalizeConfig) -> str:
    if not (cfg.tag_sumerograms or cfg.tag_determinatives):
        return text

    tokens = text.split()
    tagged: List[str] = []
    for tok in tokens:
        tagged.append(tok)
        if cfg.tag_sumerograms and _is_sumerogram(tok):
            tagged.append(_sumerogram_tag(tok))
        if cfg.tag_determinatives and tok.startswith("{") and tok.endswith("}"):
            tagged.append(_determinative_tag(tok))
    return " ".join(tagged)


#%%
# -----------------------------
# Sentence segmentation & alignment
# -----------------------------


SENTENCE_BOUNDARIES = [
    r"um-ma\s+[\w\-]+\-ma",      # quote introduction
    r"a-na\s+[\w\-]+\s+q[i\u00ED]-bi",   # "to X say"
    r"q[i\u00ED]-bi(?:\u2084)?-ma",       # say
    r"\bIGI\b",                   # witness list
    r"\bKI\u0160IB\b",            # seal list
    r"li-mu-um",                  # eponymy
    r"ITU\.KAM",                  # month marker
]


def detect_boundaries(text: str, patterns: Sequence[str]) -> List[int]:
    boundaries = [0]
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            boundaries.append(m.start())
    boundaries.append(len(text))
    return sorted(set(boundaries))


def split_by_boundaries(text: str, boundaries: Sequence[int]) -> List[str]:
    segs: List[str] = []
    for i in range(len(boundaries) - 1):
        seg = text[boundaries[i]:boundaries[i + 1]].strip()
        if seg:
            segs.append(seg)
    return segs


def split_translation_into_sentences(text: str) -> List[str]:
    # Simple sentence splitter with punctuation + key headings
    text = text.replace("\n", " ")
    chunks = re.split(r"(?<=[\.;:!?])\s+", text)
    return [c.strip() for c in chunks if c.strip()]


def align_by_length(src_segs: List[str], tgt_segs: List[str], cfg: AlignConfig) -> List[Tuple[str, str]]:
    # Insight: monotonic alignment with local merges is stable for noisy OCR.
    pairs: List[Tuple[str, str]] = []
    i, j = 0, 0
    while i < len(src_segs) and j < len(tgt_segs):
        src_accum = src_segs[i]
        tgt_accum = tgt_segs[j]
        src_i, tgt_i = i, j
        src_merge = 0
        tgt_merge = 0

        while True:
            src_tokens = len(src_accum.split()) or 1
            tgt_tokens = len(tgt_accum.split()) or 1
            ratio = tgt_tokens / src_tokens

            if ratio > cfg.max_ratio and src_i + 1 < len(src_segs) and src_merge < cfg.max_src_merge:
                src_i += 1
                src_merge += 1
                src_accum = src_accum + " " + src_segs[src_i]
                continue

            if ratio < cfg.min_ratio and tgt_i + 1 < len(tgt_segs) and tgt_merge < cfg.max_tgt_merge:
                tgt_i += 1
                tgt_merge += 1
                tgt_accum = tgt_accum + " " + tgt_segs[tgt_i]
                continue

            # Optional fine-tune: try one more tgt merge if it improves target_ratio and keeps bounds
            if tgt_i + 1 < len(tgt_segs) and tgt_merge < cfg.max_tgt_merge:
                candidate = tgt_accum + " " + tgt_segs[tgt_i + 1]
                cand_ratio = (len(candidate.split()) or 1) / (len(src_accum.split()) or 1)
                if cfg.min_ratio <= cand_ratio <= cfg.max_ratio:
                    if abs(cand_ratio - cfg.target_ratio) < abs(ratio - cfg.target_ratio):
                        tgt_i += 1
                        tgt_merge += 1
                        tgt_accum = candidate
                        continue

            break

        pairs.append((src_accum, tgt_accum))
        i = src_i + 1
        j = tgt_i + 1

    return pairs


def extract_pairs_from_annotations(
    doc_id: str,
    transliteration: str,
    sentences_df: pd.DataFrame,
    cfg: AlignConfig,
) -> List[Tuple[str, str]]:
    rows = sentences_df[sentences_df["text_uuid"] == doc_id].copy()
    if rows.empty:
        return []

    # Order by sentence position (fallback to line_number)
    sort_cols = [c for c in ["sentence_obj_in_text", "line_number"] if c in rows.columns]
    if sort_cols:
        rows = rows.sort_values(sort_cols)

    first_words = rows["first_word_transcription"].fillna("").tolist()
    translations = rows["translation"].fillna("").tolist()

    # Tokenize transliteration with positions for robust matching
    tokens: List[Tuple[str, int]] = []
    cursor = 0
    for tok in transliteration.split():
        pos = transliteration.find(tok, cursor)
        if pos == -1:
            pos = cursor
        tokens.append((tok, pos))
        cursor = pos + len(tok)

    norm_tokens = [(_normalize_token_for_match(t), pos) for t, pos in tokens]

    starts: List[Optional[int]] = []
    for fw in first_words:
        if not fw:
            starts.append(None)
            continue
        fw_norm = _normalize_token_for_match(fw)
        match_pos = None
        for tok_norm, pos in norm_tokens:
            if tok_norm == fw_norm:
                match_pos = pos
                break
        if match_pos is None:
            # fallback: partial match (prefix/suffix)
            for tok_norm, pos in norm_tokens:
                if tok_norm.startswith(fw_norm) or fw_norm.startswith(tok_norm):
                    match_pos = pos
                    break
        starts.append(match_pos)

    # Build segments using found boundaries only (high precision)
    pairs: List[Tuple[str, str]] = []
    for idx, start in enumerate(starts):
        if start is None:
            continue
        # find next valid boundary
        end = None
        for j in range(idx + 1, len(starts)):
            if starts[j] is not None:
                end = starts[j]
                break
        if end is None:
            end = len(transliteration)
        src = transliteration[start:end].strip()
        tgt = translations[idx].strip()
        if len(src) < cfg.min_src_chars or len(tgt) < cfg.min_tgt_chars:
            continue
        pairs.append((src, tgt))

    return pairs


def segment_document_rule_based(
    transliteration: str,
    translation: str,
    cfg: AlignConfig,
) -> List[Tuple[str, str]]:
    src_bounds = detect_boundaries(transliteration, SENTENCE_BOUNDARIES)
    src_segs = split_by_boundaries(transliteration, src_bounds)
    tgt_segs = split_translation_into_sentences(translation)

    if not src_segs or not tgt_segs:
        return []

    return align_by_length(src_segs, tgt_segs, cfg)


#%%
# -----------------------------
# Quality checks
# -----------------------------


def extract_numbers(text: str) -> List[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def check_length_ratio(src: str, tgt: str, cfg: AlignConfig) -> bool:
    if len(tgt) == 0:
        return False
    src_tokens = src.split()
    tgt_tokens = tgt.split()
    if not src_tokens or not tgt_tokens:
        return False
    ratio = len(tgt_tokens) / max(1, len(src_tokens))
    if len(src_tokens) <= cfg.short_token_threshold:
        return cfg.min_ratio_short <= ratio <= cfg.max_ratio_short
    return cfg.min_ratio <= ratio <= cfg.max_ratio


def check_number_consistency(src: str, tgt: str) -> bool:
    src_nums = set(extract_numbers(src))
    tgt_nums = set(extract_numbers(tgt))
    if not src_nums and not tgt_nums:
        return True
    overlap = len(src_nums & tgt_nums) / max(len(src_nums), 1)
    return overlap >= 0.5


def validate_pair(src: str, tgt: str, cfg: AlignConfig) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    fatal = False

    if len(src.strip()) < cfg.min_src_chars:
        issues.append("src_too_short")
        fatal = True
    if len(tgt.strip()) < cfg.min_tgt_chars:
        issues.append("tgt_too_short")
        fatal = True
    if not check_length_ratio(src, tgt, cfg):
        issues.append("length_ratio")
        fatal = True
    if not check_number_consistency(src, tgt):
        # Keep as a warning; not fatal because numbers are often spelled out.
        issues.append("number_mismatch")

    return not fatal, issues


#%%
# -----------------------------
# Pipeline
# -----------------------------


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype_backend="pyarrow", keep_default_na=False)
    except Exception:
        return pd.read_csv(path, keep_default_na=False)


def create_sentence_pairs(
    train_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
    align_cfg: AlignConfig,
) -> pd.DataFrame:
    pairs: List[Dict[str, str]] = []
    annotated_ids = set(sentences_df["text_uuid"]) if "text_uuid" in sentences_df.columns else set()

    for _, row in train_df.iterrows():
        doc_id = row["oare_id"]
        src = row["transliteration"]
        tgt = row["translation"]

        if doc_id in annotated_ids:
            segs = extract_pairs_from_annotations(doc_id, src, sentences_df, align_cfg)
            source_tag = "annotated"
        else:
            segs = segment_document_rule_based(src, tgt, align_cfg)
            source_tag = "rule_based"

        for s, t in segs:
            pairs.append({
                "oare_id": doc_id,
                "transliteration": s,
                "translation": t,
                "source": source_tag,
            })

    return pd.DataFrame(pairs)


def preprocess_pairs(
    df: pd.DataFrame,
    norm_cfg: NormalizeConfig,
    align_cfg: AlignConfig,
    quality_cfg: QualityConfig,
) -> pd.DataFrame:
    df = df.copy()
    df["src_raw"] = df["transliteration"]
    df["tgt_raw"] = df["translation"]

    df["src_norm"] = df["transliteration"].apply(lambda x: normalize_transliteration(x, norm_cfg))
    df["tgt_norm"] = df["translation"].apply(lambda x: normalize_translation(x, norm_cfg))

    if norm_cfg.tag_sumerograms or norm_cfg.tag_determinatives:
        df["src_tagged"] = df["src_norm"].apply(lambda x: tag_tokens(x, norm_cfg))
    else:
        df["src_tagged"] = df["src_norm"]

    validations = df.apply(
        lambda r: validate_pair(r["src_norm"], r["tgt_norm"], align_cfg), axis=1
    )
    df["is_valid"] = [v[0] for v in validations]
    df["issues"] = [v[1] for v in validations]

    df["length_ratio_tok"] = df.apply(
        lambda r: _token_ratio(r["src_norm"], r["tgt_norm"]), axis=1
    )
    df["gap_ratio"] = df["src_norm"].apply(_gap_ratio)
    df["unk_ratio"] = df["src_norm"].apply(_unk_ratio)

    df["quality_score"] = df.apply(
        lambda r: score_pair(
            r["src_norm"],
            r["tgt_norm"],
            align_cfg,
            quality_cfg,
            r["source"],
            r["issues"],
        ),
        axis=1,
    )

    df["quality_pass"] = df["quality_score"] >= quality_cfg.min_quality
    df["quality_pass"] &= df["gap_ratio"] <= quality_cfg.max_gap_ratio
    df["quality_pass"] &= df["unk_ratio"] <= quality_cfg.max_unk_ratio

    return df


def _token_ratio(src: str, tgt: str) -> float:
    src_tokens = src.split()
    tgt_tokens = tgt.split()
    return len(tgt_tokens) / max(1, len(src_tokens))


def _gap_ratio(src: str) -> float:
    tokens = src.split()
    if not tokens:
        return 0.0
    gap = sum(1 for t in tokens if t in {"<gap>", "<big_gap>"})
    return gap / len(tokens)


def _unk_ratio(src: str) -> float:
    tokens = src.split()
    if not tokens:
        return 0.0
    unk = sum(1 for t in tokens if t == "<unk_sign>")
    return unk / len(tokens)


def score_pair(
    src: str,
    tgt: str,
    align_cfg: AlignConfig,
    quality_cfg: QualityConfig,
    source: str,
    issues: List[str],
) -> float:
    # Score in [0, 1], higher is better.
    ratio = _token_ratio(src, tgt)
    ratio_dev = abs(ratio - align_cfg.target_ratio) / max(align_cfg.target_ratio, 1e-6)
    gap_r = _gap_ratio(src)
    unk_r = _unk_ratio(src)

    score = 1.0
    score -= min(1.0, ratio_dev) * quality_cfg.length_weight
    score -= min(1.0, gap_r / max(quality_cfg.max_gap_ratio, 1e-6)) * quality_cfg.gap_weight
    score -= min(1.0, unk_r / max(quality_cfg.max_unk_ratio, 1e-6)) * quality_cfg.unk_weight

    if "number_mismatch" in issues:
        score -= quality_cfg.number_penalty
    if source == "annotated":
        score += quality_cfg.annotated_bonus

    return max(0.0, min(1.0, score))


#%%
# -----------------------------
# Visualization (optional)
# -----------------------------


def plot_length_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(df["src_norm"].str.split().str.len(), bins=40)
    axes[0].set_title("Src token length")
    axes[0].set_xlabel("tokens")

    axes[1].hist(df["tgt_norm"].str.split().str.len(), bins=40)
    axes[1].set_title("Tgt token length")
    axes[1].set_xlabel("tokens")

    if "quality_score" in df.columns:
        axes[2].hist(df["quality_score"], bins=40)
        axes[2].set_title("Quality score")
        axes[2].set_xlabel("score")

    fig.tight_layout()
    fig.savefig(out_dir / "length_distributions.png", dpi=150)
    plt.close(fig)


#%%
# -----------------------------
# CLI
# -----------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Akkadian preprocessing pipeline")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--tag-sumerograms", action="store_true")
    p.add_argument("--tag-determinatives", action="store_true")
    p.add_argument("--no-normalize-h", action="store_true")
    p.add_argument("--no-normalize-subscripts", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--min-quality", type=float, default=0.0)
    p.add_argument("--max-gap-ratio", type=float, default=0.5)
    p.add_argument("--max-unk-ratio", type=float, default=0.3)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_csv(data_dir / "train.csv")
    sentences_df = read_csv(data_dir / "Sentences_Oare_FirstWord_LinNum.csv")

    # Shape safety
    assert set(train_df.columns) >= {"oare_id", "transliteration", "translation"}
    assert "text_uuid" in sentences_df.columns

    norm_cfg = NormalizeConfig(
        tag_sumerograms=args.tag_sumerograms,
        tag_determinatives=args.tag_determinatives,
        normalize_h=not args.no_normalize_h,
        normalize_subscripts=not args.no_normalize_subscripts,
    )
    align_cfg = AlignConfig()
    quality_cfg = QualityConfig(
        min_quality=args.min_quality,
        max_gap_ratio=args.max_gap_ratio,
        max_unk_ratio=args.max_unk_ratio,
    )

    sentence_df = create_sentence_pairs(train_df, sentences_df, align_cfg)
    processed_df = preprocess_pairs(sentence_df, norm_cfg, align_cfg, quality_cfg)

    processed_df.to_csv(out_dir / "sentence_pairs.csv", index=False)
    valid_df = processed_df[processed_df["is_valid"] & processed_df["quality_pass"]]
    valid_df.to_csv(out_dir / "sentence_pairs_valid.csv", index=False)

    # Save configs for reproducibility
    with open(out_dir / "preprocess_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "normalize_config": norm_cfg.__dict__,
            "align_config": align_cfg.__dict__,
            "quality_config": quality_cfg.__dict__,
        }, f, ensure_ascii=True, indent=2)

    if args.plot:
        plot_length_distributions(valid_df, out_dir)

    valid_count = len(valid_df)
    print(f"Total pairs: {len(processed_df)}")
    print(f"Valid pairs: {valid_count} ({valid_count/len(processed_df)*100:.1f}%)")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
