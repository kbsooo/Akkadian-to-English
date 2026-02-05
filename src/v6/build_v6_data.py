#!/usr/bin/env python3
"""
Build V6 Dataset - First-Word Anchor Extraction
================================================
Expand V5d dataset by joining Sentences_Oare + published_texts via UUID matching

Key Innovation:
- Use first_word_spelling as anchor to segment flat transliteration
- Expected yield: 3,173 (V5d) + 5,085 (new) = 8,258 samples (2.6x)

Usage:
    python src/v6/build_v6_data.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm.auto import tqdm


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class Config:
    # Input paths
    data_dir: Path = Path("data")
    v5d_dir: Path = Path("data/v5d")

    # Output path
    output_dir: Path = Path("data/v6")

    # Quality filters (same as V5d)
    min_src_chars: int = 10
    min_tgt_chars: int = 10
    min_ratio: float = 0.3
    max_ratio: float = 5.0
    min_src_words: int = 2
    min_tgt_words: int = 2

    # Extraction settings
    max_segment_chars: int = 512  # Prevent overly long segments
    first_word_match_threshold: float = 0.8  # Fuzzy match threshold

    # Split
    val_frac: float = 0.1
    seed: int = 42


CFG = Config()
CFG.output_dir.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Normalization (V5d-compatible)
# ==============================================================================

_VOWEL_MAP = {
    "à": "a", "á": "a", "â": "a", "ā": "a", "ä": "a",
    "À": "A", "Á": "A", "Â": "A", "Ā": "A", "Ä": "A",
    "è": "e", "é": "e", "ê": "e", "ē": "e", "ë": "e",
    "È": "E", "É": "E", "Ê": "E", "Ē": "E", "Ë": "E",
    "ì": "i", "í": "i", "î": "i", "ī": "i", "ï": "i",
    "Ì": "I", "Í": "I", "Î": "I", "Ī": "I", "Ï": "I",
    "ò": "o", "ó": "o", "ô": "o", "ō": "o", "ö": "o",
    "Ò": "O", "Ó": "O", "Ô": "O", "Ō": "O", "Ö": "O",
    "ù": "u", "ú": "u", "û": "u", "ū": "u", "ü": "u",
    "Ù": "U", "Ú": "U", "Û": "U", "Ū": "U", "Ü": "U",
}

_CONSONANT_MAP = {
    "š": "s", "Š": "S",
    "ṣ": "s", "Ṣ": "S",
    "ṭ": "t", "Ṭ": "T",
    "ḫ": "h", "Ḫ": "H",
}

_QUOTE_MAP = {
    "„": '"', """: '"', """: '"',
    "'": "'", "'": "'", "‚": "'",
    "ʾ": "'", "ʿ": "'",
}

_SUBSCRIPT_MAP = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "ₓ": "x",
}

_ALL_CHAR_MAP = {**_VOWEL_MAP, **_CONSONANT_MAP, **_QUOTE_MAP, **_SUBSCRIPT_MAP}
_TRANS_TABLE = {}
for k, v in _ALL_CHAR_MAP.items():
    if isinstance(k, str) and len(k) == 1:
        _TRANS_TABLE[ord(k)] = v


def normalize_transliteration(text: str) -> str:
    """Normalize Akkadian transliteration (V5d-compatible)."""
    if not text or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)

    # Protect literal gap tokens
    text = text.replace("<gap>", "__LIT_GAP__")
    text = text.replace("<big_gap>", "__LIT_BIG_GAP__")

    # Remove line numbers
    text = re.sub(r"\b\d+'{1,2}\b", " ", text)

    # Remove <content> blocks
    text = re.sub(r"<<([^>]+)>>", r"\1", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)

    # Large gaps
    text = re.sub(r"\[\s*…+\s*…*\s*\]", " __BIG_GAP__ ", text)
    text = re.sub(r"\[\s*\.\.\.+\s*\.\.\.+\s*\]", " __BIG_GAP__ ", text)
    text = text.replace("…", " __BIG_GAP__ ")
    text = re.sub(r"\.\.\.+", " __BIG_GAP__ ", text)

    # [x]
    text = re.sub(r"\[\s*x\s*\]", " __GAP__ ", text, flags=re.IGNORECASE)

    # [content] -> content
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)

    # Half brackets
    for char in "‹›⌈⌉⌊⌋˹˺":
        text = text.replace(char, "")

    # Character maps
    text = text.translate(_TRANS_TABLE)

    # Scribal notations
    text = re.sub(r"[!?/]", " ", text)
    text = re.sub(r"\s*:\s*", " ", text)

    # Standalone x
    text = re.sub(r"\bx\b", " __GAP__ ", text, flags=re.IGNORECASE)

    # Convert placeholders
    text = text.replace("__GAP__", "<gap>")
    text = text.replace("__BIG_GAP__", "<big_gap>")

    # Restore literal tokens
    text = text.replace("__LIT_GAP__", "<gap>")
    text = text.replace("__LIT_BIG_GAP__", "<big_gap>")

    # Cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_translation(text: str) -> str:
    """Normalize English translation."""
    if not text or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================================================================
# Quality Validation
# ==============================================================================

def validate_pair(src: str, tgt: str) -> bool:
    """Check if pair meets quality thresholds."""
    if not src or not tgt:
        return False

    # Length checks
    if len(src) < CFG.min_src_chars or len(tgt) < CFG.min_tgt_chars:
        return False
    if len(src) > CFG.max_segment_chars or len(tgt) > CFG.max_segment_chars:
        return False

    # Word count checks
    src_words = len(src.split())
    tgt_words = len(tgt.split())
    if src_words < CFG.min_src_words or tgt_words < CFG.min_tgt_words:
        return False

    # Ratio check
    ratio = tgt_words / max(src_words, 1)
    if ratio < CFG.min_ratio or ratio > CFG.max_ratio:
        return False

    return True


# ==============================================================================
# First-Word Anchor Extraction
# ==============================================================================

def fuzzy_match_score(s1: str, s2: str) -> float:
    """Simple character-level similarity."""
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    if not s1 or not s2:
        return 0.0

    # Exact match
    if s1 == s2:
        return 1.0

    # Character overlap (Jaccard)
    set1 = set(s1)
    set2 = set(s2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def find_first_word_position(transliteration: str, first_word: str, threshold: float = 0.8) -> int:
    """Find position of first_word in transliteration using fuzzy matching."""
    if not first_word or not transliteration:
        return -1

    # Normalize both
    trans_norm = normalize_transliteration(transliteration).lower()
    first_norm = normalize_transliteration(first_word).lower()

    # Try exact match first
    if first_norm in trans_norm:
        return trans_norm.find(first_norm)

    # Fuzzy search: scan through transliteration
    tokens = trans_norm.split()
    for i, token in enumerate(tokens):
        score = fuzzy_match_score(token, first_norm)
        if score >= threshold:
            # Return character position
            before = " ".join(tokens[:i])
            return len(before) + (1 if before else 0)

    return -1


def extract_segments_by_first_words(
    transliteration: str,
    sentences: List[dict],
) -> List[Tuple[str, str]]:
    """
    Extract (src, tgt) pairs using first_word_spelling as anchors.

    Args:
        transliteration: Flat transliteration text (no line breaks)
        sentences: List of sentence dicts with 'first_word_spelling', 'translation', 'sentence_obj_in_text'

    Returns:
        List of (src_segment, tgt_translation) pairs
    """
    # Sort by sentence_obj_in_text to maintain order
    sentences = sorted(sentences, key=lambda x: x.get("sentence_obj_in_text", 0))

    pairs = []
    trans_norm = normalize_transliteration(transliteration)

    # Find positions
    positions = []
    for sent in sentences:
        first_word = sent.get("first_word_spelling") or sent.get("first_word_transcription", "")
        if not first_word:
            continue

        pos = find_first_word_position(transliteration, first_word, CFG.first_word_match_threshold)
        if pos >= 0:
            positions.append({
                "pos": pos,
                "first_word": first_word,
                "translation": sent.get("translation", ""),
                "sentence_obj": sent.get("sentence_obj_in_text", 0),
            })

    # Sort by position
    positions = sorted(positions, key=lambda x: x["pos"])

    # Extract segments
    for i, item in enumerate(positions):
        start = item["pos"]
        end = positions[i + 1]["pos"] if i + 1 < len(positions) else len(trans_norm)

        src_raw = trans_norm[start:end].strip()
        tgt_raw = str(item["translation"]).strip() if item["translation"] else ""

        # Skip if translation is NaN or empty
        if not tgt_raw or tgt_raw == "nan":
            continue

        if validate_pair(src_raw, tgt_raw):
            pairs.append((src_raw, tgt_raw))

    return pairs


# ==============================================================================
# Main Pipeline
# ==============================================================================

def load_v5d_data() -> pd.DataFrame:
    """Load existing V5d data."""
    train_path = CFG.v5d_dir / "v5d_train.csv"
    val_path = CFG.v5d_dir / "v5d_val.csv"

    if not train_path.exists() or not val_path.exists():
        print(f"⚠️  V5d data not found at {CFG.v5d_dir}")
        return pd.DataFrame()

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    df = pd.concat([train, val], ignore_index=True)

    print(f"✅ Loaded V5d: {len(df):,} samples")
    return df


def extract_new_pairs() -> pd.DataFrame:
    """Extract new pairs from Sentences_Oare + published_texts."""
    print("\n📖 Loading data files...")

    # Load files
    sentences_path = CFG.data_dir / "Sentences_Oare_FirstWord_LinNum.csv"
    published_path = CFG.data_dir / "published_texts.csv"

    if not sentences_path.exists() or not published_path.exists():
        raise FileNotFoundError(f"Required files not found in {CFG.data_dir}")

    sentences_df = pd.read_csv(sentences_path)
    published_df = pd.read_csv(published_path)

    print(f"   Sentences_Oare: {len(sentences_df):,} rows")
    print(f"   published_texts: {len(published_df):,} rows")

    # Find overlapping UUIDs
    sentences_uuids = set(sentences_df["text_uuid"].dropna().unique())
    published_uuids = set(published_df["oare_id"].dropna().unique())
    overlapping_uuids = sentences_uuids & published_uuids

    print(f"\n🔗 UUID overlap: {len(overlapping_uuids):,} documents")

    # Extract pairs
    pairs = []
    skipped = 0

    for uuid in tqdm(overlapping_uuids, desc="📝 Extracting"):
        # Get sentences for this UUID
        sent_rows = sentences_df[sentences_df["text_uuid"] == uuid]
        sentences = sent_rows.to_dict("records")

        # Get transliteration
        pub_row = published_df[published_df["oare_id"] == uuid].iloc[0]
        transliteration = pub_row.get("transliteration", "")

        if not transliteration:
            skipped += 1
            continue

        # Extract segments
        segments = extract_segments_by_first_words(transliteration, sentences)

        for src, tgt in segments:
            pairs.append({
                "oare_id": uuid,
                "src_raw": src,
                "tgt_raw": tgt,
                "source": "annotated_from_published",
                "src": normalize_transliteration(src),
                "tgt": normalize_translation(tgt),
            })

    df = pd.DataFrame(pairs)

    print(f"\n✅ Extracted: {len(df):,} pairs")
    print(f"   Skipped: {skipped:,} documents (no transliteration)")

    return df


def merge_and_split(v5d_df: pd.DataFrame, new_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge V5d + new data, then split by oare_id."""
    print("\n🔀 Merging datasets...")

    # Ensure columns match
    required_cols = ["oare_id", "src_raw", "tgt_raw", "source", "src", "tgt"]

    if v5d_df.empty:
        combined = new_df
    else:
        # V5d might have extra columns, select only required
        v5d_subset = v5d_df[required_cols] if all(c in v5d_df.columns for c in required_cols) else v5d_df
        combined = pd.concat([v5d_subset, new_df], ignore_index=True)

    print(f"   Combined (before dedup): {len(combined):,} samples")

    # Remove duplicates (oare_id + src as key)
    combined['_key'] = combined['oare_id'] + '||' + combined['src']
    before = len(combined)
    combined = combined.drop_duplicates(subset=['_key'], keep='first').drop(columns=['_key'])
    after = len(combined)
    if before != after:
        print(f"   Removed {before - after:,} duplicates")

    print(f"   Combined (after dedup): {len(combined):,} samples")

    # Split by oare_id (document-level split)
    import random
    random.seed(CFG.seed)

    unique_ids = combined["oare_id"].unique().tolist()
    random.shuffle(unique_ids)

    n_val = max(1, int(len(unique_ids) * CFG.val_frac))
    val_ids = set(unique_ids[:n_val])

    train_df = combined[~combined["oare_id"].isin(val_ids)].reset_index(drop=True)
    val_df = combined[combined["oare_id"].isin(val_ids)].reset_index(drop=True)

    print(f"   Train: {len(train_df):,} ({len(train_df['oare_id'].unique())} docs)")
    print(f"   Val: {len(val_df):,} ({len(val_df['oare_id'].unique())} docs)")

    return train_df, val_df


def build_glossary(df: pd.DataFrame) -> dict:
    """Build simple glossary from training data."""
    from collections import defaultdict

    glossary = defaultdict(Counter)

    for _, row in df.iterrows():
        src_tokens = row["src"].split()
        tgt_tokens = row["tgt"].split()

        # Simple alignment: assume first src word maps to first tgt word
        if src_tokens and tgt_tokens:
            glossary[src_tokens[0]][tgt_tokens[0]] += 1

    # Keep top-1 translation per src word
    result = {}
    for src, tgt_counts in glossary.items():
        if tgt_counts:
            result[src] = [tgt_counts.most_common(1)[0][0]]

    return result


def main():
    print("=" * 60)
    print("🚀 Building V6 Dataset (First-Word Anchor)")
    print("=" * 60)

    # Load V5d base
    v5d_df = load_v5d_data()

    # Extract new pairs
    new_df = extract_new_pairs()

    # Merge and split
    train_df, val_df = merge_and_split(v5d_df, new_df)

    # Save
    train_path = CFG.output_dir / "v6_train.csv"
    val_path = CFG.output_dir / "v6_val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"\n💾 Saved:")
    print(f"   {train_path} ({len(train_df):,} rows)")
    print(f"   {val_path} ({len(val_df):,} rows)")

    # Build glossary from train
    print("\n🧠 Building glossary...")
    glossary = build_glossary(train_df)

    glossary_path = CFG.output_dir / "v6_glossary.json"
    with glossary_path.open("w", encoding="utf-8") as f:
        json.dump(glossary, f, indent=2, ensure_ascii=False)

    print(f"   {glossary_path} ({len(glossary):,} entries)")

    # Statistics
    print("\n" + "=" * 60)
    print("📊 V6 Dataset Statistics")
    print("=" * 60)
    print(f"Total samples: {len(train_df) + len(val_df):,}")
    print(f"  V5d contribution: {len(v5d_df):,}")
    print(f"  New contribution: {len(new_df):,}")
    print(f"\nSource breakdown:")
    for source in train_df["source"].value_counts().items():
        print(f"  {source[0]}: {source[1]:,}")
    print(f"\nAverage lengths:")
    print(f"  src: {train_df['src'].str.len().mean():.1f} chars")
    print(f"  tgt: {train_df['tgt'].str.len().mean():.1f} chars")
    print("=" * 60)
    print("✅ V6 dataset ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()
