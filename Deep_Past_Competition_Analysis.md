# Deep Past Initiative — Competition Deep Analysis & Breakthrough Strategy

## 1. Competition Overview

**Task:** Translate Old Assyrian cuneiform transliterations → English
**Metric:** Geometric Mean of BLEU and chrF++ (corpus-level micro-average via SacreBLEU)
**Deadline:** March 23, 2026
**Prize Pool:** $50,000 (1st: $15K, 2nd: $10K, 3rd: $8K, 4th: $7K, 5th+6th: $5K each)
**Constraints:** Code competition, CPU ≤9h or GPU ≤9h, no internet at inference

### Current Leaderboard (Top 10)

| Rank | Team | Score |
|------|------|-------|
| 1 | Darragh | 38.7 |
| 2 | Jack | 38.2 |
| 3 | Shallow Future | 38.1 |
| 4 | yanqiangmiffy | 37.9 |
| 5 | xbar19 | 37.0 |
| 6 | Raja Biswas | 36.9 |
| 7 | Yurnero | 36.7 |
| 8-9 | Hrithik Reddy / Daniel Gärber | 36.5 |
| 10 | look for job | 36.5 |

**Your current best:** 11.0 — significant room for improvement.
**Public LB uses ~34% of test; private uses ~66%** — robustness matters.

---

## 2. Data Landscape

### 2.1 Primary Files

| File | Rows | Key Content |
|------|------|-------------|
| `train.csv` | 1,561 | oare_id, transliteration, translation (document-level) |
| `test.csv` | 4 | id, text_id, line_start, line_end, transliteration |
| `sample_submission.csv` | 4 | id, translation (example output format) |
| **`v6/v6_train.csv`** | **7,391** | **src_raw, src (preprocessed), tgt — USE THIS** |
| **`v6/v6_val.csv`** | **848** | Same columns as v6_train |
| `v6/v6_glossary.json` | 1,060 | Word-level Akkadian→English mappings |

### 2.2 Supplementary Files

| File | Rows | Purpose |
|------|------|---------|
| `OA_Lexicon_eBL.csv` | 39,332 | Akkadian word forms + normalized lemmas |
| `eBL_Dictionary.csv` | 19,215 | Akkadian definitions in English |
| `Sentences_Oare_FirstWord_LinNum.csv` | 9,782 | Sentence-level alignment aid |
| `published_texts.csv` | 7,953 | Full text metadata, CDLI IDs, genre, transliterations |
| `publications.csv` | 216,602 | OCR from academic publications (554 MB) |
| `resources.csv` | ~100 | Links to external resources |

### 2.3 Data Version Evolution

| Version | Training Pairs | Notes |
|---------|---------------|-------|
| Original (`train.csv`) | 1,561 | Document-level only |
| v5 | 4,259 | Added sentence-level pairs |
| v5d | 8,308 | 7,488 train + 820 val |
| **v6 (LATEST)** | **8,239** | **7,391 train + 848 val, quality-filtered** |

**v6 composition by source:**
- `annotated_from_published`: 4,551 (61.6%) — highest quality, from publications
- `rule_based`: 2,555 (34.6%) — template-generated
- `annotated`: 285 (3.9%) — manual annotations

---

## 3. Test Set Analysis — Critical Insights

### 3.1 Structure

The public test contains **4 segments** from a **single document** (text_id: `332fda50`):

| ID | Lines | Tokens | Subject |
|----|-------|--------|---------|
| 0 | 1-7 | ~23 | Colony address, messenger instructions |
| 1 | 7-14 | ~21 | City letter about meteoric iron (KÙ.AN) trade |
| 2 | 14-24 | ~34 | Instructions for reporting iron sales |
| 3 | 25-30 | ~17 | Copy distribution to colonies/stations |

**Overlapping boundaries** (segment 0 ends at line 7, segment 1 starts at line 7) indicate continuous text.

### 3.2 Source Document Found

The test document corresponds to **Cuneiform Tablet Kt 92/k 221 (AKT 5 1)** — an administrative regulation about meteoric iron trading in the Kanesh colony (CDLI: P390599, oare_id: `3e87aad8-...`). This text exists in the training data as a complete document.

### 3.3 Vocabulary Coverage

- **59 unique test tokens** total
- 32/59 (54.2%) appear in training data
- 8/59 (13.6%) have direct glossary entries
- **27/59 (45.8%) are OOV** — this is the primary challenge
- Key OOV terms: `mup-pu-um`, `kà-ni-ia-ma`, `a-aí-im` (morphological variants)

### 3.4 Hidden Test Set

The private test uses **~66% of test data** (not visible). Based on discussion threads:
- Hidden test follows the same formatting (`<gap>`, `<big_gap>`)
- Subscript digits are converted to normal digits
- Determinatives may appear as `{d}`, `{ki}`, or `(TÚG)` — normalize both
- The hidden test is **much larger** than 4 samples

---

## 4. The Two Critical Bottlenecks (from Competition Organizers)

The DeepPast team explicitly stated these are the **dominant performance limiters** — more than model architecture:

### 4.1 Named Entities (Onomasticon)

Personal names, geographic names, and divine names cause the most translation errors:
- They are transliterated inconsistently across editions
- Semantically opaque to the model
- Small orthographic deviations collapse BLEU scores

**Solution:** Use the **Onomasticon** dataset (curated name list) for:
- Lookup/constraint layer during decoding
- Bias decoding toward known name spellings
- Post-generation repair of mangled names

### 4.2 Transliteration Format Normalization

Different corpora encode the same text using different conventions. The evaluation data expects **specific forms**:
- Keep diacritics (`š`, `ṭ`, `ṣ`) — don't strip them
- Convert ASCII substitutes INTO diacritics (e.g., `sz` → `š`)
- Normalize toward the format in training/evaluation sets
- `Ḫ`/`ḫ` → `H`/`h` (test uses only H/h)

### 4.3 Gap Markers

- Single `x` or `[x]` → `<gap>`
- Multiple `x x x` or `[… …]` or `…` → `<big_gap>`
- Parallelize gap markers between source and target
- Edge cases like `<gap>-A-Šur` should be preserved

---

## 5. Preprocessing Pipeline (MUST DO)

Based on competition instructions and v6 analysis, the **exact preprocessing** for transliteration:

```
1. Remove: ! ? / : . (scribal notations)
2. Remove: < > ˹ ˺ [ ] (brackets, keep enclosed text)
3. Replace: [x] → <gap>
4. Replace: x x x, [… …], …  → <big_gap>
5. Normalize: Ḫ/ḫ → H/h
6. Normalize: subscripts (₄→4, ₅→5, etc.)
7. Normalize: {ki} → determinative handling
8. Keep: diacritics (š, ṭ, ṣ, á, é, etc.)
9. Keep: hyphens (syllable boundaries)
10. Keep: ALL CAPS (Sumerian logograms)
```

For translations:
- Remove scribal notations
- Keep `<gap>` and `<big_gap>` aligned with source
- Preserve proper noun capitalization
- Handle floating point numbers (e.g., 0.33333 = ⅓)

---

## 6. Breakthrough Strategies (Ranked by Impact)

### TIER 1 — Guaranteed Score Boost (11.0 → 25-30+)

**A. Use v6 Data Instead of Original train.csv**
- 5.4× more training pairs (1,561 → 8,239)
- Already preprocessed with proper normalization
- Includes sentence-level pairs (better for MT)

**B. Proper Preprocessing of Test Input**
- Match test format to v6 training format exactly
- Normalize diacritics, subscripts, gap markers
- Without this, the model sees completely different distributions

**C. Use a Pretrained Multilingual Model**
- Fine-tune mT5-base, mBART, or NLLB on v6 data
- These have seen related Semitic languages during pretraining
- Much better than training from scratch on 8K examples

### TIER 2 — Competitive Score (30 → 35+)

**D. Named Entity Handling with Onomasticon**
- Download the Onomasticon from the supplementary Kaggle dataset
- Build a constraint/lookup layer for known names
- Post-process to fix name spellings

**E. Lexicon-Augmented Decoding**
- Use v6_glossary.json (1,060 entries) during beam search
- Bias generation toward glossary translations for known tokens
- Especially critical for Sumerian logograms (KÙ.BABBAR → "silver")

**F. Additional Training Data**
- External dataset: "Old Assyrian Grammars and Other Resources" on Kaggle
- Mine published_texts.csv for additional transliteration-translation pairs
- Use Sentences_Oare_FirstWord_LinNum.csv for sentence alignment
- Back-translation augmentation

### TIER 3 — Top 10 Contention (35 → 37+)

**G. LLM Post-Processing**
- Use a local LLM (fits in 9h GPU time) to post-process MT output
- Fix grammatical coherence, name consistency, formatting
- Discussion confirms this is a viable and popular approach

**H. Ensemble Methods**
- Train multiple models (different seeds, architectures, data subsets)
- Ensemble at inference via MBR (Minimum Bayes Risk) decoding
- Geometric mean metric rewards balanced, consistent output

**I. Domain-Specific Fine-Tuning**
- The corpus is overwhelmingly commercial/legal (debts, trade, letters)
- Fine-tune on domain-specific vocabulary
- Leverage genre metadata from published_texts.csv (letter: 28%, debt note: 7%, legal: 3%)

### TIER 4 — Winning Edge (37 → 39+)

**J. Pointer-Generator Networks**
- Copy mechanism for names, numbers, and logograms
- Names should be copied directly, not "translated"
- Numbers/measurements should be preserved exactly

**K. Multi-Granularity Training**
- Train on both document-level and sentence-level pairs
- Use document context during sentence-level translation
- Sliding window approach for long documents

**L. Curriculum Learning**
- Start with simple, clean examples (short, no gaps)
- Gradually introduce complex ones (long, many gaps, unclear)
- Rule-based examples first, then annotated_from_published

---

## 7. Recommended Architecture

```
Input: Akkadian Transliteration (preprocessed)
  ↓
Tokenizer: SentencePiece BPE (8K-16K vocab, preserving diacritics)
  ↓
Encoder: mT5-base or NLLB-200 encoder (pretrained)
  ↓
Decoder: Auto-regressive with:
  - Lexicon-constrained beam search (width 5-10)
  - Pointer-generator for name/number copying
  - Length normalization
  ↓
Post-Processing:
  - Onomasticon name repair
  - Gap marker alignment check
  - Number format verification
  ↓
Output: English Translation
```

**Hardware fit:** mT5-base (~580M params) or NLLB-200-distilled (~600M) fits within 9h GPU notebook.

---

## 8. Evaluation Metric Deep Dive

**Score = √(BLEU × chrF++)**

This geometric mean penalizes being bad at either metric:
- BLEU = 50, chrF++ = 50 → Score = 50
- BLEU = 80, chrF++ = 20 → Score = 40 (worse despite higher max!)
- BLEU = 0, chrF++ = 100 → Score = 0

**Implications:**
- Must be decent at BOTH word-level accuracy (BLEU) AND character-level accuracy (chrF++)
- Named entity spelling matters enormously for chrF++
- Word order matters for BLEU
- Avoid hallucination — wrong content hurts both metrics
- Gap markers must be correct format (`<gap>`, `<big_gap>`)

---

## 9. Quick Wins Checklist

- [ ] Switch from train.csv to v6_train.csv + v6_val.csv
- [ ] Apply proper preprocessing to match v6 format
- [ ] Fine-tune mT5-base or NLLB on v6 data
- [ ] Integrate v6_glossary.json for known token translation
- [ ] Download and use Onomasticon for name handling
- [ ] Add `<gap>`/`<big_gap>` handling in pre/post-processing
- [ ] Normalize `Ḫ`/`ḫ` → `H`/`h` in test input
- [ ] Handle floating point numbers (0.33333 → ⅓)
- [ ] Use beam search with length normalization
- [ ] Post-process with name repair

---

## 10. Key External Resources

1. **Kaggle Dataset:** [Old Assyrian Grammars and Resources](https://www.kaggle.com/datasets/deeppast/old-assyrian-grammars-and-other-resources/data) — Onomasticon.csv, grammars, lexicons
2. **Kaggle Dataset:** [Old Assyrian Kitepe Tablets (PDF)](https://www.kaggle.com/datasets/deeppast/old-assyrian-kitepe-tablets-in-pdf/data) — Text editions
3. **eBL Dictionary:** https://www.ebl.lmu.de/dictionary — Online Akkadian dictionary
4. **SacreBLEU:** Library for metric calculation
5. **Competition Discord:** Linked from discussion page

---

## 11. Critical File Paths

```
TRAINING (USE THESE):
/data/v6/v6_train.csv        → 7,391 preprocessed training pairs
/data/v6/v6_val.csv          → 848 preprocessed validation pairs
/data/v6/v6_glossary.json    → 1,060 word-level translations

EVALUATION:
/data/test.csv               → 4 test segments (public)
/data/sample_submission.csv  → Expected output format

SUPPLEMENTARY:
/data/OA_Lexicon_eBL.csv     → 39,332 word forms
/data/eBL_Dictionary.csv     → 19,215 definitions
/data/published_texts.csv    → 7,953 text metadata
/data/Sentences_Oare_FirstWord_LinNum.csv → 9,782 sentence pairs
```
