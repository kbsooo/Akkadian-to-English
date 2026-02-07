# V7 Planning — Complete Code & Data Audit

## 1. Version Evolution Summary

| Ver | Model | Params | Data Size | Key Idea | LB Score |
|-----|-------|--------|-----------|----------|----------|
| V1 | ByT5-base | 250M | 1,561 (doc) | First attempt, FP16 (buggy) | ~11? |
| V2 | ByT5-base | 250M | 1,561 + augmented | Unified ASCII norm, publication augment | ~11? |
| V3 | ByT5-large + LoRA | 1.2B | V2 sentence data | LoRA r=16, repetition penalty, merged model | ~11? |
| V4 | ByT5-base | 250M | 1,561 + pub | OCR noise augment, V4b-style preprocessing | — |
| V4B | ByT5-base | 250M | 1,561 only | Competition-aligned gap/bracket handling | — |
| V5 | ByT5-base | 250M | sent+pub (2-stage) | Publications DAPT → sentence-level FT | — |
| V5B | ByT5-small | ~300M | V5 + glossary | Glossary prompting (50% dropout), TM retrieval | — |
| V5C | ByT5-small | ~300M | V5 sent only | Output guardrails, early stopping, MPS support | — |
| V5D | ByT5-small | ~300M | 2,854+319 (v5d) | Simplified, EarlyStoppingCallback | — |
| **V6** | **ByT5-small** | **~300M** | **7,391+848** | **First-word anchor extraction, tokenizer saved** | **11.0** |

**Current best score: 11.0 (from V6)**
**Top 1 on leaderboard: 38.7** — gap of 27.7 points.

---

## 2. What Each Version Tried

### V1: Baseline
- ByT5-base with FP16 (UNSTABLE for ByT5)
- Raw document-level pairs from train.csv only
- LR 3e-4, 5 epochs, max_len=256
- **Problem:** FP16 causes NaN losses, train/test preprocessing mismatch

### V2: Normalization Fix + Augmentation
- Fixed: FP16 disabled, unified ASCII normalization
- Added sentence-level dataset from Sentences_Oare
- Publication augmentation with quality filters
- LR 1e-4, 10 epochs
- **Problem:** Augmentation quality inconsistent, sentence alignment fragile

### V3: Scale Up with LoRA
- Jumped to ByT5-large (1.2B) with LoRA (r=16, α=32)
- Added repetition penalty + no-repeat-ngram in decoding
- Merged LoRA into base model for Kaggle deployment
- **Problem:** Still using V2 sentence data, no data improvement

### V4/V4B: Preprocessing Refinement
- V4: OCR noise augmentation (diacritic dropping, subscript variation)
- V4B: Competition-aligned preprocessing (brackets → content, not gap)
- **Key insight:** `[content]` should keep content, not convert to `<gap>`
- Epochs reduced 10→8 (overfitting observed after epoch 7)

### V5: Two-Stage Training
- Stage A: DAPT on publications (2 epochs, LR 5e-5)
- Stage B: Sentence-level fine-tuning (8 epochs, LR 1e-4)
- First version to use geo_mean as eval metric
- **Problem:** Stage A has no validation, blind training

### V5B-V5C-V5D: Small Model + Glossary
- Switched to ByT5-small (~300M) for speed and overfitting control
- Introduced glossary prompting: `GLOSSARY: tok1=trans1; tok2=trans2 ||| source`
- 50% dropout during training, 0% during eval
- Early stopping (patience=3)
- V5C: Added output guardrails (no empty strings)
- V5D: EarlyStoppingCallback but tokenizer NOT saved

### V6: Data Expansion + Offline Fix
- **Key innovation:** First-word anchor extraction from Sentences_Oare + published_texts
- Expanded from 3,173 → 8,239 samples (2.6×)
- **Critical fix:** Tokenizer saved with model for Kaggle offline
- Glossary re-enabled, beam search 5 (from 4)
- **Score: 11.0** on public LB

---

## 3. Critical Problems Found

### 3.1 MODEL SIZE: ByT5-small May Be Adequate — Other Issues Are More Critical

**Previously estimated at 80M, but ByT5-small actually has ~300M parameters** (verified from model config: d_model=1472, 12 encoder layers, 4 decoder layers, gated-gelu FFN). This makes it comparable in total parameters to mT5-base (580M), though mT5-base has ~33% of parameters in its 250K-vocab embedding layer while ByT5-small uses only 384-token byte embeddings.

| Model | Params | Notes |
|-------|--------|-------|
| ByT5-small | ~300M | ← CURRENT. Byte-level, tiny embeddings, all params go to transformer layers |
| mT5-base | 580M | ~33% in embeddings (250K vocab). Effective transformer capacity similar to ByT5-small |
| NLLB-200-distilled | 600M | Multilingual, but no Akkadian in pretraining |
| ByT5-base | ~580M | Same architecture, 2x capacity. Consider if data > 15K pairs |
| mT5-large | 1.2B | Top competitors (with LoRA) |

**V7 decision (see V7_STRATEGY.md §2.1):** Keep ByT5-small. Research shows ByT5 outperforms mT5 by +2–5 chrF++ in the 400–10K data regime. The 11.0 score is primarily caused by issues #2–#6 below, not model capacity.

### 3.2 CRITICAL: Glossary is Fundamentally Broken

The `build_glossary()` function uses naive first-word alignment:
```python
glossary[src_tokens[0]][tgt_tokens[0]] += 1
```

This maps the first Akkadian word to the first English word. But Akkadian and English have **different word order**:
- `KU.BABBAR` (silver) → `The` (WRONG — "The" is just a common sentence opener)
- `DUMU` (son/daughter) → `The` (WRONG)
- `URUDU` (copper) → `The` (WRONG)

The glossary is **actively harmful** — it teaches the model wrong translations. Every time the glossary prompt is used (50% of training), it provides garbage signal.

### 3.3 CRITICAL: Diacritics Are Stripped (Losing Information)

The competition organizers explicitly said:
> "Converting diacritics into ASCII sequences before training was done in the WRONG DIRECTION. The evaluation data already contained diacritics."

Your normalization strips ALL diacritics:
- `š→s`, `ṣ→s` — But `š` (shin) and `ṣ` (tsade) are DIFFERENT SOUNDS
- `ṭ→t` — emphatic T is different from regular T
- `á→a`, `ú→u` — vowel quality markers are lost

**The top competitors preserve diacritics.** The hidden test data contains diacritics. If your model never sees them in training, it can't match them.

### 3.4 HIGH: Sentence Extraction Coverage Low

- published_texts.csv has 7,953 documents with **transliterations** (NOT translations — the `AICC_translation` column contains OARE API URLs, not actual translation text)
- Translations come from Sentences_Oare (9,782 sentence-level translations linked via text_uuid)
- Only 2,308 of these published_texts documents have been joined with Sentences_Oare via first-word anchor extraction
- V6 first-word anchor extraction has only 46.6% success rate — V7 improves to ~97.9% with minimal-normalization matching

### 3.5 HIGH: 222 Original Train Documents Missing

14.2% of the original train.csv documents were silently dropped during the V5d→V6 pipeline. These are **known-good, human-verified translations**.

### 3.6 MEDIUM: Normalization Destroys `<gap>` / `<big_gap>` Pattern

The regex `<([^>]+)>` for removing XML tags ALSO matches `<gap>` and `<big_gap>`:
```python
text = re.sub(r"<([^>]+)>", r"\1", text)  # This turns <gap> into just "gap"
```

The protect/restore mechanism tries to handle this but the ordering is fragile. If any intermediate step produces `<gap>` text, it gets eaten.

### 3.7 MEDIUM: No LLM Post-Processing

Discussion thread (17 upvotes) confirmed that LLM post-processing is a viable and popular approach among top competitors. You have zero post-processing.

---

## 4. V6 Data Pipeline Audit

### 4.1 Data Sources
```
V5d base data (loaded from data/v5d/):
  - v5d_train.csv: ~7,488 rows (already expanded)
  - v5d_val.csv: ~820 rows

New extraction (via first-word anchoring):
  - Sentences_Oare × published_texts UUID join
  - 4,551 new "annotated_from_published" pairs

After merge + dedup:
  - v6_train.csv: 7,391 rows
  - v6_val.csv: 848 rows
```

### 4.2 Quality by Source Type
```
Source                     Count    Avg Char Ratio    Std Dev
─────────────────────────────────────────────────────────────
rule_based                 2,555    1.11              0.49      BEST
annotated_from_published   4,551    1.03              0.69      VARIABLE
annotated                  285      0.72              0.56      WORST
```

### 4.3 Known Issues in V6 Data
- 78 exact duplicate (src, tgt) pairs
- 385 pairs with char_ratio < 0.3 or > 5.0 (misaligned)
- Determinatives ({d}, {ki}) preserved inconsistently (521 entries = 7%)
- No logging of extraction failures (silent drops)

### 4.4 What's Missing from V6

**External datasets NOT used:**
1. Onomasticon.csv (curated name list) — released by competition organizers
2. Old Assyrian Grammars dataset on Kaggle
3. Old Assyrian Kitepe Tablets PDFs
4. eBL Dictionary definitions (19,215 entries) — for glossary building
5. ~5,231 unextracted Sentences_Oare entries (V6 only extracted 46.6%)

---

## 5. Why Score is 11.0 (Root Cause Analysis)

Based on all evidence, the 11.0 score is caused by a **combination of compounding failures**:

| Root Cause | Impact | Fixable? |
|-----------|--------|----------|
| Broken glossary sending wrong signals | HIGH — actively hurts training | Yes: remove or fix |
| Diacritics stripped | HIGH — loses linguistic information | Yes: preserve diacritics |
| Low sentence extraction rate (46.6%) | HIGH — misses available data | Yes: improve matching |
| No post-processing | HIGH — raw MT output has errors | Yes: add post-proc pipeline |
| Max seq length 256 is too short | MEDIUM — truncates long examples | Yes: increase to 384 |
| 78 duplicate pairs + 385 outliers | LOW — noisy training data | Yes: filter out |

**If the top teams score 38.7 with proper setup, and your setup has 6 compounding issues, a score of 11.0 is actually explained.**

---

## 6. V7 Breakthrough Recommendations

### TIER 1: Must Do (11.0 → 30+)

**A. ~~Switch to mT5-base~~ → Keep ByT5-small (SUPERSEDED by V7_STRATEGY.md §2.1)**
- ByT5-small has ~300M params (not 80M as initially estimated), comparable transformer capacity to mT5-base
- Research (arxiv:2302.14220) shows ByT5 > mT5 by +2–5 chrF++ at 400–10K examples
- Byte-level tokenization handles Akkadian diacritics natively without vocabulary issues
- Reconsider ByT5-base (~580M) only if data exceeds 15K pairs

**B. STOP stripping diacritics — preserve them**
- The organizers explicitly said this was wrong
- Keep `š`, `ṣ`, `ṭ`, `ḫ` in transliteration
- Only normalize `Ḫ/ḫ → H/h` (test uses only H/h, per competition instructions)
- Keep vowel accents (`á`, `é`, `ú`, etc.) — they encode meaning

**C. Remove or completely rebuild the glossary**
- Option 1: Remove entirely (let the model learn alignments)
- Option 2: Build from eBL Dictionary + Onomasticon (not from training data)
- The current first-word alignment is garbage

**D. Increase max sequence length to 512**
- Many training examples exceed 256 chars
- Document-level context is critical for coherent translation

### TIER 2: Should Do (30 → 35+)

**E. Recover all missing training data**
- Include ALL 1,561 original train.csv documents
- Improve Sentences_Oare extraction rate (V7 achieves ~97.9% match with minimal normalization)
- Total target: ~10,000 pairs from improved extraction pipeline

**F. Use Onomasticon for name handling**
- Download from Kaggle supplementary dataset
- Use as lookup during inference
- Named entities are the #1 error source per competition organizers

**G. Better sentence segmentation**
- Current first-word anchor has 46.6% success rate
- Try: regex-based sentence splitting on common delimiters
- Try: sliding window approach
- Try: use full document-level pairs when sentence segmentation fails

**H. Clean existing data**
- Remove 78 exact duplicates
- Remove 385 ratio outliers
- Flag and review the 521 determinative entries

### TIER 3: Competitive Edge (35 → 37+)

**I. LLM Post-Processing**
- Use a small LLM (TinyLlama, Phi-2) to fix grammar/coherence
- Correct named entity spellings
- Ensure gap markers are properly formatted

**J. Ensemble Methods**
- Train 3-5 models with different seeds
- Use MBR decoding (Minimum Bayes Risk) for ensembling
- Geometric mean metric rewards consistency

**K. LoRA fine-tuning on mT5-large (1.2B)**
- Use LoRA (r=16-32) to fit within memory
- Pre-train on all available data, fine-tune on clean data
- You already have V3 LoRA code — adapt it

### TIER 4: Championship (37 → 39+)

**L. Two-stage curriculum**
- Stage 1: Train on all data (including noisy)
- Stage 2: Fine-tune on only clean, high-quality pairs
- Gradually increase difficulty

**M. Data augmentation with back-translation**
- Translate English → Akkadian using the model
- Use these synthetic pairs to augment training

**N. Domain adaptation from eBL Dictionary**
- Use 19,215 dictionary definitions for pretraining
- Build proper word-level translation table

---

## 7. Final V7 Architecture (see V7_STRATEGY.md for details)

```
Model: google/byt5-small (~300M params)
  - Byte-level tokenization: handles Akkadian diacritics natively
  - Tiny embedding layer (384 vocab) → all capacity goes to transformer

Training:
  - Data: V7 expanded (~9,500-10,000 pairs)
  - Epochs: 15 with early stopping (patience=3)
  - LR: 1e-4 with warmup_ratio=0.1
  - Batch: 4 × 4 accumulation = 16 effective
  - Max seq: 384 (covers 93.6% src, 99.7% hidden test)
  - NO glossary prompting
  - FP32 (ByT5 requires full precision)
  - Metric: geo_mean(BLEU, chrF++)

Inference:
  - MBR decoding: 8 candidates, chrF++ consensus
  - Post-processing: rule-based → Onomasticon → document consistency

Post-Processing:
  - Gap marker normalization
  - Onomasticon name repair (fuzzy match cutoff=0.85)
  - Document-level name consistency enforcement
```

---

## 8. Estimated Score Trajectory

| Change | Expected Score | Reasoning |
|--------|---------------|-----------|
| Current V6 | 11.0 | Baseline |
| + Remove broken glossary | 13-15 | Stop poisoning training signal |
| + Improved data extraction (~10K pairs) | 16-18 | +32% training data via better matching |
| + max_len=384 | 18-20 | Less truncation |
| + Preserve diacritics (š, ṣ, ṭ) | 20-23 | Match test format, preserve phonemes |
| + Rule-based post-processing | 23-24 | Gap/number/whitespace cleanup |
| + Onomasticon name repair | 26-29 | Fix #1 error source |
| + MBR decoding (n=8) | 29-31 | Stable consensus translation |
| + Glossary-assisted inference | 31-33 | OA_Lexicon+eBL Dict lookup |
| + LLM post-processing (optional) | 33-36 | Grammar/coherence polish |

---

## 9. Critical Files Reference

```
Code:
/src/v6/build_v6_data.py      → Data pipeline (needs rebuild for V7)
/src/v6/akkadian_v6_train.py   → Training script (needs major changes)
/src/v6/akkadian_v6_infer.py   → Inference script (needs adaptation)

Data:
/data/v6/v6_train.csv          → Current training data (7,391 rows)
/data/v6/v6_val.csv            → Current validation data (848 rows)
/data/v6/v6_glossary.json      → BROKEN glossary (DO NOT USE)
/data/train.csv                → Original data (1,561 docs)
/data/published_texts.csv      → 7,953 docs (transliterations only; AICC_translation is URL, not text)
/data/Sentences_Oare_FirstWord_LinNum.csv → 9,782 sentences
/data/OA_Lexicon_eBL.csv       → 39,332 word forms
/data/eBL_Dictionary.csv       → 19,215 definitions

External (Download from Kaggle):
- Onomasticon.csv (name list)
- Old Assyrian Grammars dataset
```
