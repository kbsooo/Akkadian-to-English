# V7 Implementation Strategy

> This document is a **self-contained specification**. A coding agent reading only this file
> should be able to implement V7 end-to-end without additional context.

---

## 0. Project Context

**Competition:** [Deep Past Initiative — Translate Akkadian to English](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)

**Task:** Translate Old Assyrian cuneiform transliterations → English. Evaluation metric: **Geometric Mean of BLEU and chrF++** (corpus-level micro-average via SacreBLEU). Submission is a Kaggle Code Competition (GPU notebook ≤ 9h, no internet at inference).

**Current state (V6):** Public LB score **11.0** (leaderboard top-1 is 38.7). Model: `google/byt5-small`. Data: 7,391 train / 848 val pairs.

**V7 target:** 29–36 on public LB.

**Repository root:** `/sessions/blissful-tender-rubin/mnt/akkadian/`
- `data/` — all competition and supplementary data
- `src/` — versioned source code (v1–v6)
- `src/v7/` — **to be created** by this plan

---

## 1. Root Cause Analysis — Why V6 Scores 11.0

Six compounding issues were identified. Fixes are ranked by expected impact.

| # | Problem | Impact | Fix |
|---|---------|--------|-----|
| 1 | **Broken glossary** poisons training (first-word alignment maps `KU.BABBAR`→`The`) | HIGH | Remove from training; rebuild from eBL Dictionary for inference-only use |
| 2 | **Diacritics stripped** (`š→s`, `ṣ→s`) — loses phonemic distinctions | HIGH | Preserve consonant diacritics; A/B test on validation |
| 3 | **71% of available data unused** — only 2,308/7,953 published texts extracted | HIGH | Improve extraction from Sentences_Oare; recover missing train.csv docs |
| 4 | **No post-processing** — raw model output submitted as-is | HIGH | Add rule-based cleanup + Onomasticon name repair + MBR decoding |
| 5 | **max_length=256** truncates 12.6% of training src | MEDIUM | Increase to 384 |
| 6 | **78 duplicate pairs + 385 ratio outliers** in training data | LOW | Filter out |

---

## 2. Decisions Made

These are **final decisions**, not options. The reasoning is documented but the implementation should follow without re-debating.

### 2.1 Model: ByT5-small (keep)

ByT5-small (~300M params, verified: d_model=1472, 12 enc + 4 dec layers, gated-gelu FFN) is the correct choice at 8–12K training examples. Research shows ByT5 outperforms mT5 by +2–5 chrF++ in the 400–10K data regime because mT5-base has ~33% of parameters locked in its 250K-vocab embedding layers, while ByT5-small's 384-token byte vocabulary uses only 0.2% for embeddings — virtually all ~300M parameters go into transformer computation. ByT5's byte-level tokenization also handles Akkadian diacritics and special characters natively. If V7 data exceeds 15K pairs, reconsider ByT5-base (~580M).

### 2.2 Max sequence length: 384 (up from 256)

The hidden test is sentence-level (98.4% fits in 256 chars), but training data needs 384 to cover 93.6% of sources without truncation. This is the optimal balance between coverage and memory/speed.

### 2.3 Glossary: remove from training, rebuild for inference

The V6 glossary uses naive first-word alignment and produces wrong mappings. Remove `build_glossary_prompt()` from training entirely. Build a linguistically correct glossary by joining `OA_Lexicon_eBL.csv` (form→lemma) with `eBL_Dictionary.csv` (lemma→definition). Use this glossary only during inference post-processing.

### 2.4 Diacritics: selective preservation

Keep consonant diacritics (`š`, `ṣ`, `ṭ`) — these represent distinct Akkadian phonemes. Convert `Ḫ/ḫ → H/h` only (test data uses H/h per competition instructions). Vowel accents (`á`, `é`, `ú`): run A/B test on validation; preserve if score improves, strip if not.

### 2.5 Post-processing: 3-phase pipeline (new)

Phase 1 (rule-based) → Phase 2 (Onomasticon) → Phase 3 (MBR decoding). Details in Section 6.

---

## 3. File Inventory

### 3.1 Competition Data (read-only)

| File | Path | Rows | Columns | Purpose |
|------|------|------|---------|---------|
| train.csv | `data/train.csv` | 1,561 | oare_id, transliteration, translation | Original document-level pairs |
| test.csv | `data/test.csv` | 4 (public) | id, text_id, line_start, line_end, transliteration | Public test (hidden test is ~3x larger) |
| sample_submission.csv | `data/sample_submission.csv` | 4 | id, translation | Submission format |

### 3.2 Supplementary Data (read-only)

| File | Path | Rows | Key Columns | Purpose |
|------|------|------|-------------|---------|
| Sentences_Oare_FirstWord_LinNum.csv | `data/Sentences_Oare_...csv` | 9,782 | text_uuid, first_word_spelling, translation, sentence_obj_in_text, line_on_tablet | Sentence-level alignment anchors |
| published_texts.csv | `data/published_texts.csv` | 7,953 | oare_id, transliteration, label, genre | Full text metadata + transliterations |
| OA_Lexicon_eBL.csv | `data/OA_Lexicon_eBL.csv` | 39,332 | form, lemma, eBL_link | Transliteration form → lemma mapping |
| eBL_Dictionary.csv | `data/eBL_Dictionary.csv` | 19,215 | word, definition, derived_from | Akkadian lemma → English definition |
| publications.csv | `data/publications.csv` | 216,602 | (OCR text) | **Do not use** — too noisy, too large |

### 3.3 V7 Data Sources (input to V7 pipeline)

V7 builds ALL data from raw sources — **no V6 dependency**:

| Source | Path | Rows | Key Columns | Purpose |
|--------|------|------|-------------|---------|
| train.csv | `data/train.csv` | 1,561 | oare_id, transliteration, translation | Document-level pairs (Path A) |
| published_texts.csv | `data/published_texts.csv` | 7,953 | oare_id, transliteration, label, genre | Full text metadata (Path C) |
| Sentences_Oare_FirstWord_LinNum.csv | `data/Sentences_Oare_...csv` | 9,782 | text_uuid, first_word_spelling, translation, line_on_tablet | Sentence-level pairs (Path B+C) |
| OA_Lexicon_eBL.csv | `data/OA_Lexicon_eBL.csv` | 39,332 | form, lemma, eBL_link | Glossary building |
| eBL_Dictionary.csv | `data/eBL_Dictionary.csv` | 19,215 | word, definition, derived_from | Glossary building |

### 3.4 V6 Source Code (reference for V7)

| File | Path | Purpose |
|------|------|---------|
| build_v6_data.py | `src/v6/build_v6_data.py` | Data pipeline — adapt for V7 |
| akkadian_v6_train.py | `src/v6/akkadian_v6_train.py` | Training script — adapt for V7 |
| akkadian_v6_infer.py | `src/v6/akkadian_v6_infer.py` | Inference script — adapt for V7 |

### 3.5 External Data (must download)

| Resource | URL | File needed |
|----------|-----|-------------|
| Old Assyrian Grammars | `kaggle.com/datasets/deeppast/old-assyrian-grammars-and-other-resources` | `Onomasticon.csv` |

---

## 4. Data Pipeline — `src/v7/build_v7_data.py`

### 4.1 Overview (UPDATED)

```
Input:  Raw competition data (NO V6 dependency)
Output: data/v7/v7_train.csv, data/v7/v7_val.csv, data/v7/v7_glossary.json
Target: ~9,500–10,000 training pairs
```

### 4.2 Key Design Change: No V6 Dependency

V6's `src_raw` had already lost diacritics for 55% of extracted pairs (`annotated_from_published`). The function `normalize_transliteration()` was applied to transliterations BEFORE extraction, making diacritics irrecoverable. V7 rebuilds ALL data from raw sources:
- `train.csv` (document-level pairs, diacritics preserved)
- `published_texts.csv` (transliterations with diacritics)
- `Sentences_Oare_FirstWord_LinNum.csv` (sentence translations + first-word anchors)

### 4.3 Extraction Paths

| Path | Source | Pairs | Description |
|------|--------|-------|-------------|
| A | train.csv direct | ~1,561 | Document-level pairs |
| B | train.csv ∩ Sentences_Oare | ~1,213 | Sentence-level from training docs |
| C | published_texts ∩ Sentences_Oare − train | ~7,263 | New sentence-level pairs |

### 4.4 First-Word Anchor Matching

- Minimal normalization for matching: `{}→()`, remove `[]` (preserves all diacritics)
- Forward-only search: each subsequent first_word searched after previous position
- Match rate: 97.87% of first_words found in original transliterations
- No fuzzy matching needed (exact substring match with minimal normalization)

### 4.5 Determinative Format Standardization

- train.csv & test.csv use `()`: `ṣí-lá-(d)IM`, `a-lim(ki)`
- published_texts uses `{}`: `ṣí-lá-{d}IM`, `a-lim{ki}`
- V7 normalization converts `{} → ()` to match test format
- This conversion is in ALL three scripts (build, train, infer)

### 4.6 Step-by-step (UPDATED)

1. Load raw data (train.csv, Sentences_Oare, published_texts, OA_Lexicon, eBL_Dictionary)
2. Path A: Extract document-level pairs from train.csv (src_raw = original transliteration)
3. Path B+C: Join Sentences_Oare + published_texts, extract via first-word anchoring on ORIGINAL diacritics-preserved transliteration
4. Combine all paths
5. Apply V7 normalization (src_raw → src) — THIS is where {} → () happens
6. Quality filter: dedup, min chars/words, ratio filter
7. Verify diacritics preservation
8. Build glossary from OA_Lexicon + eBL_Dictionary
9. Train/val split (document-level, 10% val)
10. Save output

---

## 5. Training — `src/v7/akkadian_v7_train.py`

### 5.1 Configuration

```python
model_name = "google/byt5-small"           # ~300M params (d_model=1472, 12+4 layers)
max_source_length = 384                     # up from 256
max_target_length = 384                     # up from 256
epochs = 15
batch_size = 4
gradient_accumulation_steps = 4             # effective batch = 16
learning_rate = 1e-4
warmup_ratio = 0.1
weight_decay = 0.01
early_stopping_patience = 3
use_glossary = False                        # DISABLED — no glossary prompting
fp16 = False                                # ByT5 requires FP32
seed = 42
```

### 5.2 Key Differences from V6

1. **No glossary prompting** — remove `build_glossary_prompt()` entirely. Input is just normalized src text.
2. **max_length = 384** (was 256)
3. **Data uses V7 normalization** (preserves š, ṣ, ṭ, vowel accents)
4. **Metric: `eval_geo_mean`** for best model selection (same as V6 — keep this)
5. **Save tokenizer with model** (already done in V6 — keep this)

### 5.3 What to Copy from V6

Copy these unchanged from `src/v6/akkadian_v6_train.py`:
- Device setup (CUDA > MPS > CPU)
- `build_compute_metrics()` function
- `LogCallback`, `SampleOutputCallback` classes
- `Seq2SeqTrainer` configuration
- Tokenizer + model save logic

### 5.4 Training Output

```
outputs/v7/final/
  ├── config.json
  ├── model.safetensors
  ├── tokenizer.json
  ├── tokenizer_config.json
  ├── special_tokens_map.json
  └── v7_config.json
```

Upload this entire folder to Kaggle as a dataset named `akkadian-v7-model`.

---

## 6. Inference + Post-Processing — `src/v7/akkadian_v7_infer.py`

### 6.1 Pipeline Overview

```
test.csv
  → normalize_transliteration_v7()      # Same normalization as training
  → ByT5 translation (MBR decoding)     # Generate N candidates, pick consensus
  → rule_based_postprocess()            # Gap markers, numbers, whitespace
  → onomasticon_name_repair()           # Fix proper noun spellings
  → document_consistency_check()        # Uniform names within same text_id
  → submission.csv
```

### 6.2 MBR Decoding (replaces simple beam search)

Instead of `num_beams=5, num_return_sequences=1`, generate multiple candidates and pick the most agreed-upon translation.

```python
def mbr_decode(model, tokenizer, source_text, n=8, device='cuda'):
    inputs = tokenizer(source_text, return_tensors="pt", truncation=True,
                       max_length=384).to(device)

    # Generate N candidate translations
    outputs = model.generate(
        **inputs,
        num_beams=n,
        num_return_sequences=n,
        max_length=384,
        early_stopping=True,
    )
    candidates = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    # Score each candidate against all others using chrF++
    from sacrebleu.metrics import CHRF
    chrf = CHRF(word_order=2)

    best_score, best_idx = -1, 0
    for i, cand in enumerate(candidates):
        others = [c for j, c in enumerate(candidates) if j != i]
        avg = sum(chrf.corpus_score([cand], [[r]]).score for r in others) / len(others)
        if avg > best_score:
            best_score, best_idx = avg, i

    return candidates[best_idx]
```

Time cost: ~2x standard inference. For 1000 test sentences, adds ~30 min on T4 GPU.

### 6.3 Rule-Based Post-Processing

```python
import re

def rule_based_postprocess(translation: str) -> str:
    # 1. Gap marker normalization
    #    Models may output "gap", "[gap]", "(gap)" — normalize to <gap>/<big_gap>
    translation = re.sub(r'\b(?:gap)\b(?!\s*>)', '<gap>', translation, flags=re.IGNORECASE)
    translation = re.sub(r'\[(gap)\]', '<gap>', translation, flags=re.IGNORECASE)
    translation = re.sub(r'\b(?:big[_ ]gap|large[_ ]gap)\b', '<big_gap>',
                         translation, flags=re.IGNORECASE)
    # Consecutive gaps → big_gap
    translation = re.sub(r'(<gap>\s*){2,}', '<big_gap> ', translation)

    # 2. Number normalization
    #    Models may spell out fractions — convert back to decimal
    translation = re.sub(r'\bone[- ]third\b', '0.33333', translation, flags=re.IGNORECASE)
    translation = re.sub(r'\btwo[- ]thirds?\b', '0.66666', translation, flags=re.IGNORECASE)
    translation = re.sub(r'\bone[- ]half\b', '0.5', translation, flags=re.IGNORECASE)

    # 3. Whitespace / punctuation
    translation = re.sub(r'\s+([.,;:!?)])', r'\1', translation)
    translation = re.sub(r'([(\[])\s+', r'\1', translation)
    translation = re.sub(r'\s{2,}', ' ', translation)
    translation = re.sub(r'"{2,}', '"', translation)

    return translation.strip()
```

### 6.4 Onomasticon Name Repair

**Prerequisite:** Download `Onomasticon.csv` from the Kaggle supplementary dataset and include in the Kaggle inference dataset.

```python
import difflib

def load_onomasticon(path):
    """Load name dictionary. Format: one name per line, or CSV with 'name' column."""
    import pandas as pd
    df = pd.read_csv(path)
    # Adapt column name as needed
    names = df.iloc[:, 0].dropna().str.strip().unique().tolist()
    return {n.lower(): n for n in names}

def repair_names(translation: str, onomasticon: dict) -> str:
    words = translation.split()
    result = []
    for word in words:
        if not word or not word[0].isupper():
            result.append(word)
            continue

        # Strip trailing punctuation for matching
        stripped = word.rstrip('.,;:!?"\')]')
        suffix = word[len(stripped):]

        low = stripped.lower()
        if low in onomasticon:
            result.append(onomasticon[low] + suffix)
            continue

        # Fuzzy match (cutoff=0.85 to avoid false positives)
        matches = difflib.get_close_matches(low, onomasticon.keys(), n=1, cutoff=0.85)
        if matches:
            result.append(onomasticon[matches[0]] + suffix)
        else:
            result.append(word)

    return ' '.join(result)
```

### 6.5 Document Consistency

```python
from collections import defaultdict

def enforce_consistency(translations: list, text_ids: list) -> list:
    """Ensure same name is spelled identically across segments of the same document."""
    doc_groups = defaultdict(list)
    for i, tid in enumerate(text_ids):
        doc_groups[tid].append(i)

    for tid, indices in doc_groups.items():
        if len(indices) <= 1:
            continue

        # Collect all capitalized words across segments
        from collections import Counter
        name_counts = Counter()
        for idx in indices:
            for w in translations[idx].split():
                if w[0:1].isupper() and len(w) > 2:
                    name_counts[w] += 1

        # Group similar names, pick most frequent spelling
        canonical = {}
        used = set()
        for name in sorted(name_counts, key=name_counts.get, reverse=True):
            if name.lower() in used:
                continue
            for other in name_counts:
                if other.lower() != name.lower() and other.lower() not in used:
                    if difflib.SequenceMatcher(None, name.lower(), other.lower()).ratio() > 0.8:
                        canonical[other] = name
                        used.add(other.lower())
            used.add(name.lower())

        # Apply
        for idx in indices:
            for wrong, right in canonical.items():
                translations[idx] = translations[idx].replace(wrong, right)

    return translations
```

### 6.6 Full Inference Flow

```python
def main():
    # Load model + tokenizer (local_files_only=True for Kaggle offline)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model.to(device).eval()

    # Load test data
    test_df = pd.read_csv(COMP_DIR / "test.csv")

    # Step 1: Normalize transliterations (MUST match training normalization)
    normalized = [normalize_transliteration_v7(t) for t in test_df["transliteration"]]

    # Step 2: Translate with MBR decoding
    translations = []
    for src in tqdm(normalized):
        trans = mbr_decode(model, tokenizer, src, n=8, device=device)
        translations.append(trans)

    # Free model memory (in case LLM post-proc is added later)
    del model
    torch.cuda.empty_cache()

    # Step 3: Rule-based post-processing
    translations = [rule_based_postprocess(t) for t in translations]

    # Step 4: Onomasticon name repair
    onomasticon = load_onomasticon(ASSETS_DIR / "Onomasticon.csv")
    translations = [repair_names(t, onomasticon) for t in translations]

    # Step 5: Document consistency
    translations = enforce_consistency(translations, test_df["text_id"].tolist())

    # Save submission
    submission = pd.DataFrame({"id": test_df["id"], "translation": translations})
    submission.to_csv("submission.csv", index=False)
```

---

## 7. Normalization Spec — `normalize_transliteration_v7()`

This is the most critical function. It MUST be identical in build_v7_data.py, akkadian_v7_train.py, and akkadian_v7_infer.py.

```python
import re, unicodedata

_V7_TRANS_TABLE = str.maketrans({
    # Determinative format: {} → () (published_texts uses {}, test.csv uses ())
    "{": "(", "}": ")",
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
    if not text or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)

    # Protect existing gap tokens
    text = text.replace("<gap>", "\x00GAP\x00")
    text = text.replace("<big_gap>", "\x00BIGGAP\x00")

    # Remove apostrophe line numbers (1', 1'')
    text = re.sub(r"\b\d+'{1,2}\b", " ", text)

    # Remove angle-bracket content markers (keep content)
    text = re.sub(r"<<([^>]+)>>", r"\1", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)

    # Large gaps: [...], [… …], x x x, …
    text = re.sub(r"\[\s*…+\s*…*\s*\]", " \x00BIGGAP\x00 ", text)
    text = re.sub(r"\[\s*\.\.\.+\s*\]", " \x00BIGGAP\x00 ", text)
    text = text.replace("…", " \x00BIGGAP\x00 ")
    text = re.sub(r"\.\.\.+", " \x00BIGGAP\x00 ", text)

    # Single gap: [x]
    text = re.sub(r"\[\s*x\s*\]", " \x00GAP\x00 ", text, flags=re.IGNORECASE)

    # Strip brackets, keep content: [content] → content
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)

    # Half brackets
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
```

---

## 8. Diacritics A/B Test Protocol

Before committing to V7 normalization, run a quick comparison:

```
Variant A: V6 normalization (strip all diacritics: š→s, ṣ→s, ṭ→t, á→a, etc.)
Variant B: V7 normalization (preserve š, ṣ, ṭ; strip only Ḫ→H; preserve vowel accents)
```

Procedure:
1. Build V7 data with both normalizations → two train/val sets
2. Train ByT5-small for 5 epochs on each (same hyperparameters)
3. Evaluate on val set → compare geo_mean(BLEU, chrF++)
4. Pick the winner as the V7 normalization

If Variant B wins by ≥0.5 geo_mean → use V7 normalization.
If difference < 0.5 → use V7 normalization anyway (matches competition intent).
If Variant A wins by ≥1.0 → reconsider (possible overfitting to normalized test).

---

## 9. Environment Constraints

### 9.1 Training — Google Colab (A100 or T4)

- Data loading: `kagglehub.competition_download()` and `kagglehub.dataset_download()` for competition data and V6 base data
- Package management: `pip install kagglehub transformers datasets sacrebleu accelerate`
- Model download: `google/byt5-small` from HuggingFace (internet available)
- Model upload: Save to local `outputs_v7/final/`, then upload to Kaggle via web UI or `kaggle datasets create`
- FP32 only (ByT5 is unstable with FP16)

### 9.2 Inference — Kaggle (T4 × 2 GPUs)

- **Internet OFF** → `local_files_only=True` for all model/tokenizer loading
- **Multi-GPU**: Copy model to both GPUs with `copy.deepcopy()`, split test data in half, process in parallel via `ThreadPoolExecutor` (~2x speedup)
- **Older transformers**: Use `tokenizer=` (not `processing_class=`) in Trainer; avoid `generation_config` object; use direct kwargs to `model.generate()`
- **sacrebleu**: Available on Kaggle (used for MBR decoding chrF++ scoring)
- **Time budget**: MBR with 8 candidates on dual T4 ≈ 1.5h for 1000 samples (well within 9h)

### 9.3 Code Format

- All scripts written as `.py` files using **Jupytext percent format** (`# %%` cell markers)
- Convert to `.ipynb` with: `uv jupytext --to notebook <file>.py`
- **Immediate execution style**: No `main()` wrapper. Each cell runs top-to-bottom, prints results for debugging. Functions are defined inline where needed, not collected at top.

---

## 10. Files Created

```
src/v7/
  ├── build_v7_data.py          # Data pipeline (Jupytext, Colab)
  ├── akkadian_v7_train.py      # Training (Jupytext, Colab A100/T4)
  └── akkadian_v7_infer.py      # Inference + post-processing (Jupytext, Kaggle T4×2)

data/v7/                        # Output of build_v7_data.py (to be created)
  ├── v7_train.csv
  ├── v7_val.csv
  └── v7_glossary.json
```

Note: Normalization function is **inlined identically** in all three scripts (no shared module import needed). Glossary building and post-processing are inlined in their respective scripts.

---

## 11. Kaggle Submission Checklist

The Kaggle inference notebook must:

1. Load model + tokenizer from `/kaggle/input/akkadian-v7-model/` with `local_files_only=True`
2. Load test.csv from competition data directory
3. Load Onomasticon.csv from supplementary dataset
4. Apply `normalize_transliteration_v7()` to test transliterations
5. Run MBR decoding (n=8 candidates)
6. Apply rule-based post-processing
7. Apply Onomasticon name repair
8. Apply document consistency enforcement
9. Save `submission.csv` to `/kaggle/working/`
10. Verify: no NaN values, correct row count, id column matches test

Required Kaggle datasets to attach:
- `akkadian-v7-model` (model weights + tokenizer)
- `deeppast/old-assyrian-grammars-and-other-resources` (Onomasticon.csv)
- Competition data (auto-attached)

Time budget (T4 × 2 GPUs, 9h limit):
- Model load + dual-GPU setup: ~1 min
- Normalization: ~5s
- MBR inference (1000 sentences × 8 beams, dual-GPU): ~1.5h
- Post-processing: ~5 min
- **Total: ~2h — well within 9h limit**

---

## 12. Validation & Success Criteria

Before Kaggle submission, verify on V7 val set:

| Metric | V6 Baseline | V7 Target | Method |
|--------|-------------|-----------|--------|
| BLEU | ~8 | ≥25 | sacrebleu |
| chrF++ | ~15 | ≥30 | sacrebleu (word_order=2) |
| geo_mean | ~11 | ≥27 | √(BLEU × chrF++) |
| Empty outputs | unknown | 0 | count |

If val geo_mean < 20 after full training: investigate data quality or normalization mismatch.
If val geo_mean > 25: submit to Kaggle immediately, then iterate on post-processing.
