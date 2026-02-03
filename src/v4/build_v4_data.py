#%% [markdown]
# # Build V4 Dataset (OG data -> data/v4)
#
# Creates document-level and sentence-level datasets using V4b normalization.
# Optionally mines publications for additional (noisy) translation pairs.
#
# Usage:
#   uv run python src/v4/build_v4_data.py --data-dir data --out-dir data/v4
#   uv run python src/v4/build_v4_data.py --data-dir data --out-dir data/v4 --use-publications

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Reuse sentence alignment helpers from data_preprocessing.py
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_preprocessing import (
    AlignConfig,
    extract_pairs_from_annotations,
    segment_document_rule_based,
)


# -----------------------------
# V4b Normalization (match train/infer)
# -----------------------------

_VOWEL_MAP = {
    "\u00e0": "a", "\u00e1": "a", "\u00e2": "a", "\u0101": "a", "\u00e4": "a",
    "\u00c0": "A", "\u00c1": "A", "\u00c2": "A", "\u0100": "A", "\u00c4": "A",
    "\u00e8": "e", "\u00e9": "e", "\u00ea": "e", "\u0113": "e", "\u00eb": "e",
    "\u00c8": "E", "\u00c9": "E", "\u00ca": "E", "\u0112": "E", "\u00cb": "E",
    "\u00ec": "i", "\u00ed": "i", "\u00ee": "i", "\u012b": "i", "\u00ef": "i",
    "\u00cc": "I", "\u00cd": "I", "\u00ce": "I", "\u012a": "I", "\u00cf": "I",
    "\u00f2": "o", "\u00f3": "o", "\u00f4": "o", "\u014d": "o", "\u00f6": "o",
    "\u00d2": "O", "\u00d3": "O", "\u00d4": "O", "\u014c": "O", "\u00d6": "O",
    "\u00f9": "u", "\u00fa": "u", "\u00fb": "u", "\u016b": "u", "\u00fc": "u",
    "\u00d9": "U", "\u00da": "U", "\u00db": "U", "\u016a": "U", "\u00dc": "U",
}

_CONSONANT_MAP = {
    "\u0161": "s", "\u0160": "S",
    "\u1e63": "s", "\u1e62": "S",
    "\u1e6d": "t", "\u1e6c": "T",
    "\u1e2b": "h", "\u1e2a": "H",
}

_QUOTE_MAP = {
    "\u201e": '"', "\u201c": '"', "\u201d": '"',
    "\u2018": "'", "\u2019": "'", "\u201a": "'",
    "\u02be": "'", "\u02bf": "'",
}

_SUBSCRIPT_MAP = str.maketrans({
    "\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u2084": "4",
    "\u2085": "5", "\u2086": "6", "\u2087": "7", "\u2088": "8", "\u2089": "9",
    "\u2093": "x",
})

_FULL_MAP = str.maketrans({**_VOWEL_MAP, **_CONSONANT_MAP, **_QUOTE_MAP})


def normalize_transliteration(text: str) -> str:
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)

    # Protect literal gap tokens before removing <content> blocks
    text = text.replace("<gap>", "__LIT_GAP__")
    text = text.replace("<big_gap>", "__LIT_BIG_GAP__")

    # Remove only apostrophe-style line numbers (e.g., 1', 1'')
    text = re.sub(r"\b\d+'{1,2}\b", " ", text)

    # 1. Handle <content> -> content FIRST
    text = re.sub(r"<<([^>]+)>>", r"\1", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)

    # 2. Large gaps
    text = re.sub(r"\[\s*…+\s*…*\s*\]", " __BIG_GAP__ ", text)
    text = re.sub(r"\[\s*\.\.\.+\s*\.\.\.+\s*\]", " __BIG_GAP__ ", text)

    # 3. Ellipsis
    text = text.replace("\u2026", " __BIG_GAP__ ")
    text = re.sub(r"\.\.\.+", " __BIG_GAP__ ", text)

    # 4. [x]
    text = re.sub(r"\[\s*x\s*\]", " __GAP__ ", text, flags=re.IGNORECASE)

    # 5. [content] -> content
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)

    # 6. Half brackets
    text = text.replace("\u2039", "").replace("\u203a", "")
    text = text.replace("\u2308", "").replace("\u2309", "")
    text = text.replace("\u230a", "").replace("\u230b", "")
    text = text.replace("˹", "").replace("˺", "")

    # 7. Character maps
    text = text.translate(_FULL_MAP)
    text = text.translate(_SUBSCRIPT_MAP)

    # 8. Scribal notations / word dividers
    text = re.sub(r"[!?/]", " ", text)
    text = re.sub(r"\s*:\s*", " ", text)

    # 9. Standalone x
    text = re.sub(r"\bx\b", " __GAP__ ", text, flags=re.IGNORECASE)

    # 10. Convert placeholders
    text = text.replace("__GAP__", "<gap>")
    text = text.replace("__BIG_GAP__", "<big_gap>")

    # Restore literal tokens
    text = text.replace("__LIT_GAP__", "<gap>")
    text = text.replace("__LIT_BIG_GAP__", "<big_gap>")

    # 11. Cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_translation(text: str) -> str:
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Utilities
# -----------------------------


def split_by_oare_id(df: pd.DataFrame, val_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    ids = df["oare_id"].dropna().unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val])
    train_df = df[~df["oare_id"].isin(val_ids)].reset_index(drop=True)
    val_df = df[df["oare_id"].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df, sorted(val_ids)


def filter_pairs(df: pd.DataFrame, min_src: int, min_tgt: int, min_ratio: float, max_ratio: float) -> pd.DataFrame:
    df = df.copy()
    df = df[(df["src"].str.len() >= min_src) & (df["tgt"].str.len() >= min_tgt)]
    src_len = df["src"].str.split().str.len().clip(lower=1)
    tgt_len = df["tgt"].str.split().str.len().clip(lower=1)
    ratio = tgt_len / src_len
    df = df[(ratio >= min_ratio) & (ratio <= max_ratio)]
    return df.reset_index(drop=True)


# -----------------------------
# Sentence-level pairs
# -----------------------------


def build_sentence_pairs(train_df: pd.DataFrame, sentences_df: pd.DataFrame, use_rule_based: bool, align_cfg: AlignConfig) -> pd.DataFrame:
    pairs: List[Dict[str, str]] = []
    annotated_ids = set(sentences_df["text_uuid"].unique())

    for _, row in train_df.iterrows():
        doc_id = row["oare_id"]
        src_raw = row["transliteration"]
        tgt_raw = row["translation"]

        segs: List[Tuple[str, str]] = []
        source_tag = "rule_based"

        if doc_id in annotated_ids:
            # Fill missing first_word_transcription with spelling
            rows = sentences_df[sentences_df["text_uuid"] == doc_id].copy()
            if "first_word_transcription" in rows.columns:
                rows["first_word_transcription"] = rows["first_word_transcription"].fillna("")
            if "first_word_spelling" in rows.columns:
                rows.loc[rows["first_word_transcription"].eq(""), "first_word_transcription"] = rows["first_word_spelling"].fillna("")

            segs = extract_pairs_from_annotations(doc_id, src_raw, rows, align_cfg)
            source_tag = "annotated"
        elif use_rule_based:
            segs = segment_document_rule_based(src_raw, tgt_raw, align_cfg)
            source_tag = "rule_based"

        for s, t in segs:
            pairs.append({
                "oare_id": doc_id,
                "src_raw": s,
                "tgt_raw": t,
                "source": source_tag,
            })

    df = pd.DataFrame(pairs)
    if df.empty:
        return df

    df["src"] = df["src_raw"].apply(normalize_transliteration)
    df["tgt"] = df["tgt_raw"].apply(normalize_translation)
    return df


# -----------------------------
# Publications mining (optional, noisy)
# -----------------------------


ID_PATTERNS = [
    r"kt\s*[\w/]+\s*\d+[a-z]?",
    r"ick\s*\d+\s*\d+[a-z]?",
    r"cct\s*\d+\s*\d+[a-z]?",
    r"bin\s*\d+\s*\d+[a-z]?",
    r"tc\s*\d+\s*\d+[a-z]?",
    r"akt\s*\d+[a-z]?\s*\d+[a-z]?",
    r"poat\s*\d+",
    r"vs\s*\d+\s*\d+[a-z]?",
    r"kts\s*\d+\s*\d+[a-z]?",
    r"kt\s*\d+",
]


def normalize_id(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"[\[\](){}.,:;]", "", t)
    return t


def build_alias_map(published_df: pd.DataFrame) -> Dict[str, List[Tuple[str, str]]]:
    alias_cols = [
        "aliases",
        "label",
        "publication_catalog",
        "inventory_position",
        "cdli_id",
        "oatp_key",
        "excavation_no",
    ]
    alias_map: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for _, row in published_df.iterrows():
        oare_id = row.get("oare_id", "")
        for col in alias_cols:
            val = row.get(col, "")
            if not isinstance(val, str) or not val.strip():
                continue
            parts = re.split(r"[|;,]", val)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                norm = normalize_id(p)
                if not norm:
                    continue
                alias_map[norm].append((oare_id, p))
    return alias_map


def extract_translation_candidates(page_text: str) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []

    # Line range patterns: "1-3) ..."
    for m in re.finditer(r"(?:^|\n)\s*\d+\s*[-–]\s*\d+\)\s*([^\n]{20,})", page_text):
        candidates.append((m.group(1).strip(), "line_range"))

    # Translation markers
    for m in re.finditer(r"(?:Translation|\u00dcbersetzung|Traduction|Traduzione)\s*[:\-]\s*([^\n]{20,})", page_text, flags=re.IGNORECASE):
        candidates.append((m.group(1).strip(), "marker"))

    # Quoted English-like text
    for m in re.finditer(r'"([A-Z][^"]{20,})"', page_text):
        candidates.append((m.group(1).strip(), "quoted"))

    # Deduplicate
    seen = set()
    uniq: List[Tuple[str, str]] = []
    for text, kind in candidates:
        key = (text, kind)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((text, kind))
    return uniq


COMMON_EN = {
    "the", "and", "of", "to", "in", "that", "this", "from", "for", "is", "are", "be", "with",
}
COMMON_DE = {"der", "die", "das", "und", "ist", "nicht", "ein", "eine", "mit", "von"}
COMMON_FR = {"le", "la", "les", "et", "est", "une", "des", "dans", "pour"}


def guess_lang(text: str) -> str:
    lowered = re.sub(r"[^a-z\s]", " ", text.lower())
    tokens = lowered.split()
    if not tokens:
        return "unknown"
    en = sum(1 for t in tokens if t in COMMON_EN)
    de = sum(1 for t in tokens if t in COMMON_DE)
    fr = sum(1 for t in tokens if t in COMMON_FR)
    if en >= 2 and en >= de and en >= fr:
        return "en"
    if de >= 2 and de >= en and de >= fr:
        return "de"
    if fr >= 2 and fr >= en and fr >= de:
        return "fr"
    return "unknown"


def mine_publications(
    publications_path: Path,
    published_df: pd.DataFrame,
    out_candidates: Path,
    out_doc_pairs: Path,
    max_rows: int,
    max_per_id: int,
    english_only: bool,
) -> Tuple[int, int]:
    alias_map = build_alias_map(published_df)
    translit_map = {r["oare_id"]: r.get("transliteration", "") for _, r in published_df.iterrows()}

    id_patterns = [re.compile(pat, flags=re.IGNORECASE) for pat in ID_PATTERNS]
    per_id_count: Dict[str, int] = defaultdict(int)
    best_doc: Dict[str, Tuple[str, str]] = {}

    out_candidates.parent.mkdir(parents=True, exist_ok=True)
    with out_candidates.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["oare_id", "alias", "pdf_name", "page", "lang", "kind", "translation", "src_norm"])

        with publications_path.open(newline="", encoding="utf-8") as f_pub:
            reader = csv.DictReader(f_pub)
            for i, row in enumerate(reader, start=1):
                if max_rows > 0 and i > max_rows:
                    break

                has_akkadian = str(row.get("has_akkadian", "")).lower() in {"true", "1", "yes"}
                if not has_akkadian:
                    continue

                page_text = row.get("page_text", "") or ""
                if not page_text:
                    continue

                matches: List[str] = []
                for pat in id_patterns:
                    for m in pat.findall(page_text):
                        matches.append(m)

                if not matches:
                    continue

                for raw_id in matches:
                    norm_id = normalize_id(raw_id)
                    if norm_id not in alias_map:
                        continue

                    for oare_id, alias in alias_map[norm_id]:
                        if per_id_count[oare_id] >= max_per_id:
                            continue

                        candidates = extract_translation_candidates(page_text)
                        if not candidates:
                            continue

                        for text, kind in candidates:
                            lang = guess_lang(text)
                            if english_only and lang != "en":
                                continue

                            src_raw = translit_map.get(oare_id, "")
                            src_norm = normalize_transliteration(src_raw) if src_raw else ""
                            if not src_norm:
                                continue

                            writer.writerow([
                                oare_id,
                                alias,
                                row.get("pdf_name", ""),
                                row.get("page", ""),
                                lang,
                                kind,
                                text,
                                src_norm,
                            ])

                            per_id_count[oare_id] += 1

                            # Keep best (longest) translation as doc-level pair
                            prev = best_doc.get(oare_id)
                            if prev is None or len(text) > len(prev[0]):
                                best_doc[oare_id] = (text, src_norm)

                        if per_id_count[oare_id] >= max_per_id:
                            break

    # Write doc-level pairs
    out_doc_pairs.parent.mkdir(parents=True, exist_ok=True)
    with out_doc_pairs.open("w", newline="", encoding="utf-8") as f_doc:
        writer = csv.writer(f_doc)
        writer.writerow(["oare_id", "src", "tgt", "source"])
        for oare_id, (tgt_raw, src_norm) in best_doc.items():
            tgt_norm = normalize_translation(tgt_raw)
            if not tgt_norm:
                continue
            writer.writerow([oare_id, src_norm, tgt_norm, "publications_doc"])

    return len(best_doc), sum(per_id_count.values())


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V4 dataset from OG data")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="data/v4")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-rule-based", action="store_true")

    parser.add_argument("--min-src", type=int, default=5)
    parser.add_argument("--min-tgt", type=int, default=5)
    parser.add_argument("--min-ratio", type=float, default=0.2)
    parser.add_argument("--max-ratio", type=float, default=5.0)

    parser.add_argument("--dapt-min-len", type=int, default=10)

    parser.add_argument("--use-publications", action="store_true")
    parser.add_argument("--publications-max-rows", type=int, default=0, help="0 = all")
    parser.add_argument("--publications-max-per-id", type=int, default=3)
    parser.add_argument("--publications-english-only", action="store_true")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    sentences_path = data_dir / "Sentences_Oare_FirstWord_LinNum.csv"
    published_path = data_dir / "published_texts.csv"
    publications_path = data_dir / "publications.csv"

    train_df = pd.read_csv(train_path)
    sentences_df = pd.read_csv(sentences_path)
    published_df = pd.read_csv(published_path)

    # Document-level pairs (train.csv)
    doc_df = train_df[["oare_id", "transliteration", "translation"]].dropna().copy()
    doc_df["src"] = doc_df["transliteration"].apply(normalize_transliteration)
    doc_df["tgt"] = doc_df["translation"].apply(normalize_translation)
    doc_df["source"] = "train_doc"
    doc_df = doc_df[(doc_df["src"].str.len() > 0) & (doc_df["tgt"].str.len() > 0)].reset_index(drop=True)

    # Sentence-level pairs
    align_cfg = AlignConfig()
    sent_df = build_sentence_pairs(train_df, sentences_df, args.include_rule_based, align_cfg)
    if not sent_df.empty:
        sent_df = filter_pairs(sent_df, args.min_src, args.min_tgt, args.min_ratio, args.max_ratio)

    # Split by document id
    doc_train, doc_val, val_ids = split_by_oare_id(doc_df, args.val_frac, args.seed)
    if not sent_df.empty:
        sent_train = sent_df[~sent_df["oare_id"].isin(val_ids)].reset_index(drop=True)
        sent_val = sent_df[sent_df["oare_id"].isin(val_ids)].reset_index(drop=True)
    else:
        sent_train = sent_val = sent_df

    # Save
    doc_train.to_csv(out_dir / "v4_doc_train.csv", index=False)
    doc_val.to_csv(out_dir / "v4_doc_val.csv", index=False)

    if not sent_df.empty:
        sent_train.to_csv(out_dir / "v4_sentence_train.csv", index=False)
        sent_val.to_csv(out_dir / "v4_sentence_val.csv", index=False)

    # DAPT transliteration pool
    dapt_texts = published_df["transliteration"].dropna().astype(str).tolist()
    dapt_norm = [normalize_transliteration(t) for t in dapt_texts]
    dapt_norm = [t for t in dapt_norm if len(t) >= args.dapt_min_len]
    dapt_norm = sorted(set(dapt_norm))
    with (out_dir / "v4_dapt_translit.txt").open("w", encoding="utf-8") as f:
        for t in dapt_norm:
            f.write(t + "\n")

    # Publications mining (optional)
    pub_doc_pairs = 0
    pub_candidates = 0
    if args.use_publications:
        pub_candidates_path = out_dir / "v4_publications_candidates.csv"
        pub_doc_pairs_path = out_dir / "v4_publications_doc_pairs.csv"
        pub_doc_pairs, pub_candidates = mine_publications(
            publications_path,
            published_df,
            pub_candidates_path,
            pub_doc_pairs_path,
            args.publications_max_rows,
            args.publications_max_per_id,
            args.publications_english_only,
        )

    # Stats
    stats = {
        "doc_train": len(doc_train),
        "doc_val": len(doc_val),
        "sentence_train": len(sent_train) if not sent_df.empty else 0,
        "sentence_val": len(sent_val) if not sent_df.empty else 0,
        "dapt_translit": len(dapt_norm),
        "publications_doc_pairs": pub_doc_pairs,
        "publications_candidates": pub_candidates,
        "val_ids": len(val_ids),
    }
    with (out_dir / "v4_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("✅ V4 dataset built")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
