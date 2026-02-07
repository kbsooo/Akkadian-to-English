# %% [markdown]
# # V7 Data Pipeline — From-Scratch Build (Diacritics-Preserving)
#
# **Critical change from V6:** This pipeline builds ALL training data from raw
# sources. V6 stripped diacritics (š→s, ṣ→s, ṭ→t) during extraction, making
# them irrecoverable. V7 preserves all diacritics throughout.
#
# ### Extraction Paths
# | Path | Source | Expected pairs |
# |------|--------|---------------|
# | A | train.csv document-level pairs | ~1,561 |
# | B | train.csv docs segmented via Sentences_Oare | ~1,213 |
# | C | New docs from Sentences_Oare + published_texts | ~7,263 |
# | **Total** | | **~10,037** |
#
# ### Format Note
# - train.csv & test.csv use `()` for determinatives: `ṣí-lá-(d)IM`
# - published_texts.csv uses `{}`: `ṣí-lá-{d}IM`
# - V7 normalization standardizes `{} → ()` to match test format

# %% [markdown]
# ## 0. Configuration

# %%
KAGGLE_USERNAME = "your-username"  # EDIT THIS
COMPETITION = "deep-past-initiative-machine-translation"
OUTPUT_DIR = "data_v7"

# Quality filter thresholds
MIN_CHARS = 10
MIN_WORDS = 2
MIN_RATIO = 0.15   # tgt_words / src_words
MAX_RATIO = 6.0
MAX_SEGMENT_CHARS = 2000  # skip extremely long segments

# Split
VAL_FRAC = 0.1
SEED = 42

print("V7 Data Pipeline — From-Scratch Build")
print(f"  Competition: {COMPETITION}")
print(f"  Output: {OUTPUT_DIR}/")

# %% [markdown]
# ## 1. Install Dependencies

# %%
import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "kagglehub", "pandas", "tqdm"
])
print("Dependencies ready: kagglehub, pandas, tqdm")

# %% [markdown]
# ## 2. Download Competition Data

# %%
import kagglehub
import os

comp_path = kagglehub.competition_download(COMPETITION)
print(f"Competition data: {comp_path}")
for f in sorted(os.listdir(comp_path)):
    fp = os.path.join(comp_path, f)
    if os.path.isfile(fp):
        mb = os.path.getsize(fp) / (1024 * 1024)
        print(f"  {f} ({mb:.1f} MB)")

# %% [markdown]
# ## 3. Load Raw Data Files

# %%
import pandas as pd

train_og = pd.read_csv(os.path.join(comp_path, "train.csv"))
sentences_df = pd.read_csv(os.path.join(comp_path, "Sentences_Oare_FirstWord_LinNum.csv"))
published_df = pd.read_csv(os.path.join(comp_path, "published_texts.csv"))
lexicon_df = pd.read_csv(os.path.join(comp_path, "OA_Lexicon_eBL.csv"))
dictionary_df = pd.read_csv(os.path.join(comp_path, "eBL_Dictionary.csv"))

print("Raw data loaded:")
print(f"  train.csv:          {len(train_og):,} rows")
print(f"  Sentences_Oare:     {len(sentences_df):,} rows")
print(f"  published_texts:    {len(published_df):,} rows")
print(f"  OA_Lexicon:         {len(lexicon_df):,} rows")
print(f"  eBL_Dictionary:     {len(dictionary_df):,} rows")

# Precompute ID sets
train_ids = set(train_og["oare_id"].dropna().unique())
sent_uuids = set(sentences_df["text_uuid"].dropna().unique())
pub_ids = set(published_df["oare_id"].dropna().unique())

print(f"\nID overlap:")
print(f"  train ∩ Sentences_Oare:     {len(train_ids & sent_uuids)}")
print(f"  train ∩ published_texts:    {len(train_ids & pub_ids)}")
print(f"  Sentences ∩ published:      {len(sent_uuids & pub_ids)}")
print(f"  New for extraction (S∩P-T): {len((sent_uuids & pub_ids) - train_ids)}")

# %% [markdown]
# ## 4. Define Normalization Functions
#
# Three normalization functions with distinct purposes:
# 1. `normalize_for_matching()` — Minimal, for first-word anchor position finding
# 2. `normalize_transliteration_v7()` — Full V7, for model input
# 3. `normalize_translation()` — For English translations

# %%
import re
import unicodedata


def normalize_for_matching(text):
    """
    Minimal normalization for first-word anchor matching.
    Only fixes known format mismatches between data sources.
    PRESERVES all diacritics, subscripts, and linguistic content.
    """
    if not text or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    # Standardize determinative format: {} → () to match train/test format
    text = text.replace("{", "(").replace("}", ")")
    # Remove square brackets (editorial marks that block exact matching)
    text = re.sub(r"[\[\]]", "", text)
    return text


# ---- V7 normalization (MUST be identical in train/infer scripts) ----

_V7_TRANS_TABLE = str.maketrans({
    # Ḫ/ḫ → H/h (competition instruction: test uses only H/h)
    "Ḫ": "H", "ḫ": "h",
    # Subscripts → digits
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9", "ₓ": "x",
    # Smart quotes → ASCII
    "„": '"', "\u201c": '"', "\u201d": '"',
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "ʾ": "'", "ʿ": "'",
})
# NOTE: š, Š, ṣ, Ṣ, ṭ, Ṭ, á, é, í, ó, ú, à, è, ì, ò, ù are PRESERVED


def normalize_transliteration_v7(text: str) -> str:
    """V7 normalization: preserves š, ṣ, ṭ and vowel accents. Only Ḫ→H."""
    if not text or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)

    # Standardize determinative braces: {} → () to match test format
    text = text.replace("{", "(").replace("}", ")")

    # Protect existing gap tokens
    text = text.replace("<gap>", "\x00GAP\x00")
    text = text.replace("<big_gap>", "\x00BIGGAP\x00")

    # Remove apostrophe line numbers (1', 1'')
    text = re.sub(r"\b\d+'{1,2}\b", " ", text)

    # Remove angle-bracket content markers (keep content)
    text = re.sub(r"<<([^>]+)>>", r"\1", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)

    # Large gaps: [...], [… …], …, ...
    text = re.sub(r"\[\s*…+\s*…*\s*\]", " \x00BIGGAP\x00 ", text)
    text = re.sub(r"\[\s*\.\.\.+\s*\]", " \x00BIGGAP\x00 ", text)
    text = text.replace("…", " \x00BIGGAP\x00 ")
    text = re.sub(r"\.\.\.+", " \x00BIGGAP\x00 ", text)

    # Single gap: [x]
    text = re.sub(r"\[\s*x\s*\]", " \x00GAP\x00 ", text, flags=re.IGNORECASE)

    # Strip square brackets, keep content: [content] → content
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)

    # Half brackets / editorial marks
    for c in "‹›⌈⌉⌊⌋˹˺":
        text = text.replace(c, "")

    # Character map (diacritics preserved except Ḫ→H)
    text = text.translate(_V7_TRANS_TABLE)

    # Scribal notations
    text = re.sub(r"[!?/]", " ", text)
    text = re.sub(r"\s*:\s*", " ", text)

    # Standalone x → gap (AFTER char map to avoid clobbering ₓ→x)
    text = re.sub(r"(?<![a-zA-Z\x00])\bx\b(?![a-zA-Z])", " \x00GAP\x00 ", text)

    # Restore gap tokens
    text = text.replace("\x00GAP\x00", "<gap>")
    text = text.replace("\x00BIGGAP\x00", "<big_gap>")

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_translation(text: str) -> str:
    """Normalize English translation: NFC + whitespace cleanup."""
    if not text or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Quick tests
assert "š" in normalize_transliteration_v7("ša-bu-um")
assert "ṣ" in normalize_transliteration_v7("ṣí-lá")
assert "ṭ" in normalize_transliteration_v7("ṭè-mu")
assert "á" in normalize_transliteration_v7("lá")
assert "H" in normalize_transliteration_v7("Ḫa-mu")
assert "(d)" in normalize_transliteration_v7("{d}IM")
assert "(d)" in normalize_transliteration_v7("(d)IM")
print("Normalization functions defined and tested:")
print("  normalize_for_matching    — minimal, for anchor matching")
print("  normalize_transliteration_v7 — full V7, for model input")
print("  normalize_translation     — for English translations")
print("  All diacritics assertions passed ✓")

# %% [markdown]
# ## 5. Define First-Word Anchor Extraction

# %%
def extract_sentence_pairs(transliteration, sentences_for_doc):
    """
    Extract (src_raw, tgt_raw) pairs from a transliteration using first-word
    anchoring from Sentences_Oare.

    Strategy:
    1. Apply normalize_for_matching() to transliteration ({} → (), remove [])
    2. For each sentence, find first_word_spelling position (forward-only search)
    3. Extract segments between consecutive anchor positions
    4. Return (segment, translation) pairs with diacritics preserved

    Args:
        transliteration: Raw transliteration from published_texts.transliteration
        sentences_for_doc: DataFrame rows for this document from Sentences_Oare

    Returns:
        List of (src_segment, translation) tuples
    """
    if not transliteration or str(transliteration) == "nan":
        return []

    search_text = normalize_for_matching(transliteration)
    if not search_text.strip():
        return []

    # Sort sentences by sentence_obj_in_text to maintain document order
    sent_list = []
    for _, row in sentences_for_doc.iterrows():
        fw = row.get("first_word_spelling", "")
        trans = row.get("translation", "")
        obj_idx = row.get("sentence_obj_in_text", 0)

        if (not fw or str(fw) == "nan" or
            not trans or str(trans) == "nan"):
            continue

        sent_list.append({
            "first_word": str(fw).strip(),
            "translation": str(trans).strip(),
            "obj_idx": int(obj_idx) if not pd.isna(obj_idx) else 0,
        })

    sent_list.sort(key=lambda x: x["obj_idx"])

    if not sent_list:
        return []

    # Find anchor positions with forward-only search
    anchors = []
    search_from = 0

    for sent in sent_list:
        search_fw = normalize_for_matching(sent["first_word"])
        if not search_fw:
            continue

        # Try exact substring match (forward from last position)
        pos = search_text.find(search_fw, search_from)
        if pos < 0:
            # Try case-insensitive
            pos_lower = search_text.lower().find(search_fw.lower(), search_from)
            if pos_lower >= 0:
                pos = pos_lower

        if pos >= 0:
            anchors.append((pos, sent["translation"]))
            search_from = pos + 1  # move forward to avoid re-matching

    if not anchors:
        return []

    # Extract segments between consecutive anchors
    pairs = []
    for i, (start, translation) in enumerate(anchors):
        end = anchors[i + 1][0] if i + 1 < len(anchors) else len(search_text)
        segment = search_text[start:end].strip()

        if segment and translation and len(segment) >= 5:
            pairs.append((segment, translation))

    return pairs


# Quick test with known data
test_trans = "KIŠIB ma-nu-ba-lúm-a-šur DUMU ṣí-lá-{d}IM šu-{d}EN.LÍL"
test_sents = pd.DataFrame([
    {"first_word_spelling": "KIŠIB", "translation": "Seal of...", "sentence_obj_in_text": 0},
    {"first_word_spelling": "šu-(d)EN.LÍL", "translation": "Šu-Illil...", "sentence_obj_in_text": 1},
])
result = extract_sentence_pairs(test_trans, test_sents)
assert len(result) == 2
assert "šur" in result[0][0]  # first segment has Akkadian content
assert "ṣí-lá" in result[0][0]  # diacritics preserved
print(f"Extraction test passed ✓")
print(f"  Segment 1: '{result[0][0][:60]}...' → '{result[0][1]}'")
print(f"  Segment 2: '{result[1][0][:60]}...' → '{result[1][1]}'")

# %% [markdown]
# ## 6. Path A — Document-Level Pairs from train.csv
#
# Direct extraction: each row is a (transliteration, translation) pair.
# Diacritics are **already preserved** in train.csv.

# %%
path_a_pairs = []

for _, row in train_og.iterrows():
    src_raw = str(row.get("transliteration", ""))
    tgt_raw = str(row.get("translation", ""))

    if src_raw in ("", "nan") or tgt_raw in ("", "nan"):
        continue

    path_a_pairs.append({
        "oare_id": row["oare_id"],
        "src_raw": src_raw,
        "tgt_raw": tgt_raw,
        "source": "train_document",
    })

path_a_df = pd.DataFrame(path_a_pairs)

# Verify diacritics preservation
has_diacritics = path_a_df["src_raw"].str.contains(r"[šṣṭáéíúàèìù]", regex=True, na=False)
print(f"Path A — Document-Level Pairs from train.csv:")
print(f"  Pairs extracted: {len(path_a_df):,}")
print(f"  With diacritics: {has_diacritics.sum():,} ({100*has_diacritics.mean():.1f}%)")
print(f"  Unique documents: {path_a_df['oare_id'].nunique()}")
print(f"  Sample src: {path_a_df['src_raw'].iloc[0][:100]}...")

# %% [markdown]
# ## 7. Path B — Segment Training Documents via Sentences_Oare
#
# For 253 training documents that have Sentences_Oare entries, we extract
# sentence-level pairs using first-word anchoring on the **original
# diacritics-preserved** transliteration from published_texts.

# %%
from tqdm import tqdm

# IDs: train docs that are also in Sentences_Oare AND published_texts
path_b_target_ids = train_ids & sent_uuids & pub_ids
print(f"Path B — Segment Training Documents:")
print(f"  Target documents: {len(path_b_target_ids)}")

path_b_pairs = []
path_b_stats = {"matched": 0, "unmatched": 0, "no_trans": 0}

for oid in tqdm(path_b_target_ids, desc="Path B"):
    # Get transliteration from published_texts (has diacritics + {} braces)
    pub_row = published_df[published_df["oare_id"] == oid]
    if pub_row.empty:
        path_b_stats["no_trans"] += 1
        continue

    transliteration = pub_row.iloc[0].get("transliteration", "")
    if not transliteration or str(transliteration) == "nan":
        path_b_stats["no_trans"] += 1
        continue

    # Get sentences from Sentences_Oare
    sent_rows = sentences_df[sentences_df["text_uuid"] == oid]
    if sent_rows.empty:
        continue

    # Extract pairs
    pairs = extract_sentence_pairs(transliteration, sent_rows)

    if pairs:
        path_b_stats["matched"] += 1
        for src, tgt in pairs:
            path_b_pairs.append({
                "oare_id": oid,
                "src_raw": src,
                "tgt_raw": tgt,
                "source": "train_segmented",
            })
    else:
        path_b_stats["unmatched"] += 1

path_b_df = pd.DataFrame(path_b_pairs)

has_diacritics_b = path_b_df["src_raw"].str.contains(r"[šṣṭáéíúàèìù]", regex=True, na=False) if len(path_b_df) > 0 else pd.Series(dtype=bool)
print(f"\n  Results:")
print(f"    Pairs extracted: {len(path_b_df):,}")
print(f"    Documents matched: {path_b_stats['matched']}")
print(f"    Documents unmatched: {path_b_stats['unmatched']}")
print(f"    With diacritics: {has_diacritics_b.sum() if len(has_diacritics_b) > 0 else 0} ({100*has_diacritics_b.mean():.1f}%)" if len(has_diacritics_b) > 0 else "    (no pairs)")
if len(path_b_df) > 0:
    print(f"    Sample: '{path_b_df['src_raw'].iloc[0][:80]}...'")

# %% [markdown]
# ## 8. Path C — Extract New Document Pairs
#
# 1,164 documents in Sentences_Oare ∩ published_texts that are **not** in
# train.csv. These are entirely new training data.

# %%
# IDs: Sentences_Oare ∩ published_texts - train.csv
path_c_target_ids = (sent_uuids & pub_ids) - train_ids
print(f"Path C — New Document Extraction:")
print(f"  Target documents: {len(path_c_target_ids)}")

path_c_pairs = []
path_c_stats = {"matched": 0, "unmatched": 0, "no_trans": 0}

for oid in tqdm(path_c_target_ids, desc="Path C"):
    pub_row = published_df[published_df["oare_id"] == oid]
    if pub_row.empty:
        path_c_stats["no_trans"] += 1
        continue

    transliteration = pub_row.iloc[0].get("transliteration", "")
    if not transliteration or str(transliteration) == "nan":
        path_c_stats["no_trans"] += 1
        continue

    sent_rows = sentences_df[sentences_df["text_uuid"] == oid]
    if sent_rows.empty:
        continue

    pairs = extract_sentence_pairs(transliteration, sent_rows)

    if pairs:
        path_c_stats["matched"] += 1
        for src, tgt in pairs:
            path_c_pairs.append({
                "oare_id": oid,
                "src_raw": src,
                "tgt_raw": tgt,
                "source": "new_segmented",
            })
    else:
        path_c_stats["unmatched"] += 1

path_c_df = pd.DataFrame(path_c_pairs)

has_diacritics_c = path_c_df["src_raw"].str.contains(r"[šṣṭáéíúàèìù]", regex=True, na=False) if len(path_c_df) > 0 else pd.Series(dtype=bool)
print(f"\n  Results:")
print(f"    Pairs extracted: {len(path_c_df):,}")
print(f"    Documents matched: {path_c_stats['matched']}")
print(f"    Documents unmatched: {path_c_stats['unmatched']}")
print(f"    No transliteration: {path_c_stats['no_trans']}")
print(f"    With diacritics: {has_diacritics_c.sum() if len(has_diacritics_c) > 0 else 0} ({100*has_diacritics_c.mean():.1f}%)" if len(has_diacritics_c) > 0 else "    (no pairs)")
if len(path_c_df) > 0:
    print(f"    Sample: '{path_c_df['src_raw'].iloc[0][:80]}...'")

# %% [markdown]
# ## 9. Combine All Paths

# %%
combined = pd.concat([path_a_df, path_b_df, path_c_df], ignore_index=True)

print(f"Combined dataset (before cleaning):")
print(f"  Path A (train_document):  {len(path_a_df):>6,}")
print(f"  Path B (train_segmented): {len(path_b_df):>6,}")
print(f"  Path C (new_segmented):   {len(path_c_df):>6,}")
print(f"  {'─' * 35}")
print(f"  Total:                    {len(combined):>6,}")
print(f"\nSource distribution:")
print(combined["source"].value_counts().to_string())

# %% [markdown]
# ## 10. Apply V7 Normalization (BEFORE cleaning)
#
# Apply V7 normalization to produce `src` and `tgt` columns.
# This is done **before** quality filtering so that length checks
# operate on the normalized text (which the model actually sees).

# %%
print("Applying V7 normalization...")

combined["src"] = combined["src_raw"].apply(normalize_transliteration_v7)
combined["tgt"] = combined["tgt_raw"].apply(normalize_translation)

# Show normalization effect
print(f"\nNormalization examples:")
for i in range(min(3, len(combined))):
    print(f"\n  [{combined['source'].iloc[i]}]")
    print(f"    src_raw: {combined['src_raw'].iloc[i][:80]}...")
    print(f"    src:     {combined['src'].iloc[i][:80]}...")

# Verify diacritics survive normalization
all_diacritics = combined["src"].str.contains(r"[šṣṭáéíúàèìù]", regex=True, na=False)
print(f"\nDiacritics in normalized src:")
print(f"  With diacritics: {all_diacritics.sum():,}/{len(combined):,} ({100*all_diacritics.mean():.1f}%)")

# Check specific diacritics
for char, name in [("š", "shin"), ("ṣ", "tsade"), ("ṭ", "teth"),
                   ("á", "a-acute"), ("ú", "u-acute"), ("ì", "i-grave")]:
    count = combined["src"].str.contains(char, na=False).sum()
    print(f"  {char} ({name}): {count:,} rows")

# %% [markdown]
# ## 11. Quality Filter and Deduplication

# %%
before = len(combined)
print(f"Quality filtering (starting with {before:,} pairs):")

# 1. Remove empty or NaN
empty_mask = combined["src"].isna() | combined["tgt"].isna() | (combined["src"] == "") | (combined["tgt"] == "")
combined = combined[~empty_mask]
print(f"  After removing empty: {len(combined):,} (removed {empty_mask.sum()})")

# 2. Remove exact duplicates on (src, tgt)
before_dedup = len(combined)
combined = combined.drop_duplicates(subset=["src", "tgt"], keep="first")
print(f"  After dedup (src, tgt): {len(combined):,} (removed {before_dedup - len(combined)})")

# 3. Remove too short (on normalized text)
short_mask = (combined["src"].str.len() < MIN_CHARS) | (combined["tgt"].str.len() < MIN_CHARS)
combined = combined[~short_mask]
print(f"  After min chars ({MIN_CHARS}): {len(combined):,} (removed {short_mask.sum()})")

# 4. Remove too few words
few_words_mask = (combined["src"].str.split().str.len() < MIN_WORDS) | (combined["tgt"].str.split().str.len() < MIN_WORDS)
combined = combined[~few_words_mask]
print(f"  After min words ({MIN_WORDS}): {len(combined):,} (removed {few_words_mask.sum()})")

# 5. Remove ratio outliers
src_words = combined["src"].str.split().str.len().clip(lower=1)
tgt_words = combined["tgt"].str.split().str.len().clip(lower=1)
ratio = tgt_words / src_words
ratio_mask = (ratio < MIN_RATIO) | (ratio > MAX_RATIO)
combined = combined[~ratio_mask]
print(f"  After ratio filter ({MIN_RATIO}-{MAX_RATIO}): {len(combined):,} (removed {ratio_mask.sum()})")

# 6. Remove extremely long segments
long_mask = (combined["src"].str.len() > MAX_SEGMENT_CHARS) | (combined["tgt"].str.len() > MAX_SEGMENT_CHARS)
combined = combined[~long_mask]
print(f"  After max chars ({MAX_SEGMENT_CHARS}): {len(combined):,} (removed {long_mask.sum()})")

combined = combined.reset_index(drop=True)
print(f"\nFinal cleaned dataset: {len(combined):,} pairs (removed {before - len(combined)} total)")

# %% [markdown]
# ## 12. Verify Diacritics Preservation
#
# This is the **most critical check**. V7's entire value proposition is that
# diacritics are preserved. If this cell fails, the pipeline is broken.

# %%
print("=" * 60)
print("DIACRITICS PRESERVATION VERIFICATION")
print("=" * 60)

diacritic_chars = {
    "š": "shin (voiceless postalveolar)",
    "ṣ": "tsade (emphatic s)",
    "ṭ": "teth (emphatic t)",
    "á": "a-acute",
    "é": "e-acute",
    "í": "i-acute",
    "ú": "u-acute",
    "à": "a-grave",
    "è": "e-grave",
    "ì": "i-grave",
    "ù": "u-grave",
}

all_pass = True
for char, name in diacritic_chars.items():
    in_src = combined["src"].str.contains(char, na=False).sum()
    in_raw = combined["src_raw"].str.contains(char, na=False).sum()
    status = "✓" if in_src > 0 else "✗ MISSING"
    if in_src == 0:
        all_pass = False
    print(f"  {char} ({name:30s}): src={in_src:>5,}  src_raw={in_raw:>5,}  {status}")

# Check that V7 normalization did NOT strip diacritics
stripped_chars = {"Ḫ": "H", "ḫ": "h"}
print(f"\n  Correctly converted:")
for old, new in stripped_chars.items():
    remaining = combined["src"].str.contains(old, na=False).sum()
    print(f"    {old}→{new}: {remaining} remaining (should be 0) {'✓' if remaining == 0 else '✗'}")
    if remaining > 0:
        all_pass = False

# Check {} → () conversion
curly_remaining = combined["src"].str.contains(r"\{", regex=True, na=False).sum()
paren_count = combined["src"].str.contains(r"\(", regex=True, na=False).sum()
print(f"\n  Determinative format:")
print(f"    Remaining {{}}:  {curly_remaining} (should be 0) {'✓' if curly_remaining == 0 else '✗'}")
print(f"    Using ():      {paren_count:,}")
if curly_remaining > 0:
    all_pass = False

print(f"\n{'=' * 60}")
print(f"VERIFICATION: {'PASSED ✓' if all_pass else 'FAILED ✗'}")
print(f"{'=' * 60}")

if not all_pass:
    print("WARNING: Diacritics verification failed! Check the normalization pipeline.")

# %% [markdown]
# ## 13. Build Glossary from OA_Lexicon + eBL_Dictionary
#
# Join: form → lemma (OA_Lexicon) → definition (eBL_Dictionary)
# Used for inference post-processing only, NOT for training.

# %%
import json

print("Building glossary from OA_Lexicon + eBL_Dictionary...")

# Build form → lemma mapping
form_to_lemma = {}
for _, row in lexicon_df.iterrows():
    form = row.get("form")
    lemma = row.get("lemma")
    if pd.notna(form) and pd.notna(lemma):
        form_str = str(form).strip()
        lemma_str = str(lemma).strip()
        if form_str and lemma_str:
            form_to_lemma[form_str] = lemma_str

# Build lemma → definition mapping
lemma_to_def = {}
for _, row in dictionary_df.iterrows():
    word = row.get("word")
    definition = row.get("definition")
    if pd.notna(word) and pd.notna(definition):
        word_str = str(word).strip()
        def_str = str(definition).strip()
        if word_str and def_str and len(def_str) > 1:
            lemma_to_def[word_str] = def_str

print(f"  Form → Lemma mappings: {len(form_to_lemma):,}")
print(f"  Lemma → Definition mappings: {len(lemma_to_def):,}")

# Join: form → lemma → definition
glossary = {}
for form, lemma in form_to_lemma.items():
    definition = lemma_to_def.get(lemma, "")
    if definition:
        # Take first sense, strip grammar notes
        first_sense = definition.split(";")[0].split(",")[0].strip()
        first_sense = re.sub(r"\(.*?\)", "", first_sense).strip()
        if first_sense and len(first_sense) > 1:
            glossary[form] = first_sense

print(f"  Glossary entries (form → first sense): {len(glossary):,}")

# Show samples
if glossary:
    items = list(glossary.items())[:8]
    print(f"\n  Samples:")
    for form, defn in items:
        print(f"    '{form}' → '{defn}'")

# %% [markdown]
# ## 14. Train/Validation Split (Document-Level)

# %%
import random

random.seed(SEED)

unique_ids = combined["oare_id"].dropna().unique().tolist()
random.shuffle(unique_ids)

n_val = max(1, int(len(unique_ids) * VAL_FRAC))
val_ids = set(unique_ids[:n_val])

train_df = combined[~combined["oare_id"].isin(val_ids)].reset_index(drop=True)
val_df = combined[combined["oare_id"].isin(val_ids)].reset_index(drop=True)

print(f"Train/Val split (document-level, {VAL_FRAC*100:.0f}% val):")
print(f"  Train: {len(train_df):>6,} pairs from {train_df['oare_id'].nunique():,} documents")
print(f"  Val:   {len(val_df):>6,} pairs from {val_df['oare_id'].nunique():,} documents")

print(f"\n  Train sources:")
print(train_df["source"].value_counts().to_string(header=False))
print(f"\n  Val sources:")
print(val_df["source"].value_counts().to_string(header=False))

# Verify no document leakage
leak = set(train_df["oare_id"].unique()) & set(val_df["oare_id"].unique())
print(f"\n  Document leakage check: {len(leak)} shared IDs {'(CLEAN ✓)' if len(leak) == 0 else '(LEAKED ✗)'}")

# %% [markdown]
# ## 15. Save Output Files

# %%
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = os.path.join(OUTPUT_DIR, "v7_train.csv")
val_path = os.path.join(OUTPUT_DIR, "v7_val.csv")
glossary_path = os.path.join(OUTPUT_DIR, "v7_glossary.json")

# Save with consistent column order
columns = ["oare_id", "src_raw", "tgt_raw", "source", "src", "tgt"]
train_df[columns].to_csv(train_path, index=False)
val_df[columns].to_csv(val_path, index=False)

with open(glossary_path, "w", encoding="utf-8") as f:
    json.dump(glossary, f, indent=2, ensure_ascii=False)

train_mb = os.path.getsize(train_path) / (1024 * 1024)
val_mb = os.path.getsize(val_path) / (1024 * 1024)
gloss_mb = os.path.getsize(glossary_path) / (1024 * 1024)

print(f"Files saved to '{OUTPUT_DIR}/':")
print(f"  v7_train.csv     ({len(train_df):>6,} pairs, {train_mb:.1f} MB)")
print(f"  v7_val.csv       ({len(val_df):>6,} pairs, {val_mb:.1f} MB)")
print(f"  v7_glossary.json ({len(glossary):>6,} entries, {gloss_mb:.1f} MB)")

# %% [markdown]
# ## 16. Final Statistics

# %%
print("=" * 70)
print("V7 DATA PIPELINE — FINAL STATISTICS")
print("=" * 70)

print(f"\n  Dataset sizes:")
print(f"    Train: {len(train_df):>6,} pairs from {train_df['oare_id'].nunique():>4,} documents")
print(f"    Val:   {len(val_df):>6,} pairs from {val_df['oare_id'].nunique():>4,} documents")
print(f"    Total: {len(combined):>6,} pairs from {combined['oare_id'].nunique():>4,} documents")

print(f"\n  Source breakdown (total):")
for src, cnt in combined["source"].value_counts().items():
    print(f"    {src:25s}: {cnt:>6,} ({100*cnt/len(combined):.1f}%)")

print(f"\n  Text statistics (train set):")
print(f"    Avg src length:   {train_df['src'].str.len().mean():>6.0f} chars")
print(f"    Avg tgt length:   {train_df['tgt'].str.len().mean():>6.0f} chars")
print(f"    Avg src words:    {train_df['src'].str.split().str.len().mean():>6.1f}")
print(f"    Avg tgt words:    {train_df['tgt'].str.split().str.len().mean():>6.1f}")
print(f"    Max src length:   {train_df['src'].str.len().max():>6,} chars")
print(f"    Max tgt length:   {train_df['tgt'].str.len().max():>6,} chars")

# Percentile analysis
p90_src = train_df["src"].str.len().quantile(0.9)
p95_src = train_df["src"].str.len().quantile(0.95)
p99_src = train_df["src"].str.len().quantile(0.99)
print(f"\n  Source length percentiles:")
print(f"    P90: {p90_src:.0f}  P95: {p95_src:.0f}  P99: {p99_src:.0f}")

print(f"\n  Glossary: {len(glossary):,} entries")
print(f"\n  Output: {OUTPUT_DIR}/")
print("=" * 70)

# %% [markdown]
# ## 17. Upload to Kaggle
#
# Upload the V7 dataset to Kaggle for use in Colab training.

# %%
# To upload to Kaggle as a dataset:
#
# Option 1: Using Kaggle CLI
# !pip install kaggle
# !kaggle datasets init -p {OUTPUT_DIR}
# Then edit dataset-metadata.json:
#   "id": "your-username/akkadian-v7-data"
#   "title": "Akkadian V7 Training Data"
# !kaggle datasets create -p {OUTPUT_DIR}
#
# Option 2: Manual upload via https://www.kaggle.com/datasets/new
#
# Option 3: Copy to Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# !cp -r {OUTPUT_DIR} '/content/drive/MyDrive/akkadian/data/'

print(f"Upload instructions: see cell above")
print(f"Dataset directory: {OUTPUT_DIR}/")
print(f"Files: v7_train.csv, v7_val.csv, v7_glossary.json")
