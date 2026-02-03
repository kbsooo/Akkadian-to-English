# Deep Research EDA Report: Akkadian Translation Competition

**Date**: 2026-02-02
**Objective**: Comprehensive understanding of all competition data for optimal translation strategy

---

## Executive Summary

This analysis reveals **critical structural differences** between train and test data that will significantly impact model performance:

1. **Test data is LINE-SEGMENTED** (4 segments from 1 document), while **train data contains FULL DOCUMENTS** (1561 complete texts)
2. **Test uses a HELD-OUT document** (text_id: `332fda50`) not found in any provided dataset
3. **Encoding differences** exist between test and train (e.g., `„` vs `₄`)
4. **Length distribution shift**: Test avg ~21 tokens vs Train avg ~57 tokens

---

## 1. Test Data Analysis (MOST CRITICAL)

### 1.1 Structure Overview

| Field | Description |
|-------|-------------|
| `id` | Sequential integer (0-3) |
| `text_id` | Document identifier: `332fda50` (8 chars, truncated UUID?) |
| `line_start` | Starting line number in original tablet |
| `line_end` | Ending line number in original tablet |
| `transliteration` | Akkadian text in transliteration format |

### 1.2 Test Data Contents

| id | line_start | line_end | span | tokens | chars |
|----|------------|----------|------|--------|-------|
| 0 | 1 | 7 | 6 lines | 16 | 133 |
| 1 | 7 | 14 | 7 lines | 19 | 146 |
| 2 | 14 | 24 | 10 lines | 34 | 267 |
| 3 | 25 | 30 | 5 lines | 16 | 129 |

**Key Observations**:
- All 4 rows belong to the **SAME document** (text_id: 332fda50)
- This is a **LINE-SEGMENTED** format - each row covers specific line ranges
- Line ranges are **consecutive but overlap at boundaries** (7, 14, 25)
- The document spans lines 1-30 on the original tablet

### 1.3 Test Transliteration Patterns

**Sample from Row 0**:
```
um-ma kà-ru-um kà-ni-ia-ma a-na aa-qí-il... da-tim aí-ip-ri-ni kà-ar kà-ar-ma ú wa-bar-ra-tim qí-bi„-ma mup-pu-um aa a-lim(ki) i-li-kam
```

**Special Characters in Test**:
- `„` (U+201E) - Used instead of subscript numbers (appears 6 times)
- `...` (U+2026) - Ellipsis marking damaged/missing text (1 occurrence)
- `+` - Unknown purpose, appears in `me-+e-er` (1 occurrence)
- `(ki)` - Determinative indicating place name

**Test-Only Vocabulary** (27 tokens not in train):
- `aa`, `aa-qí-il...`, `aé-bi„-lá`, `aé-bi„-lá-nim`, `aí-ip-ri-ni`
- `aí-mì-im`, `au-mì`, `au-um-au`, `da-aùr`, `ia-ra-tí-au`
- `ia-tí`, `ia-tù`, `kà-ni-ia`, `kà-ni-ia-ma`, `me-+e-er`
- `mup-pu-um`, `mup-pì-im`, `mup-pì-ni`, `na-aí-ma`, `na-áa-ú`
- `ni-bi„-it`, `qí-bi„-ma`, `ta-áa-me-a-ni`, `u„-mì-im`

**Vocabulary Overlap**: Only 54.2% of test tokens appear in train data

### 1.4 Critical Finding: Test text_id is ORPHAN

The test `text_id` value `332fda50` does **NOT** match:
- Any `oare_id` in train.csv
- Any `oare_id` in published_texts.csv
- Any `text_uuid` in Sentences_Oare.csv
- Any pattern in the entire dataset

**Conclusion**: Test data comes from a **HELD-OUT SOURCE** not available in training data.

---

## 2. Train Data Analysis

### 2.1 Structure Overview

| Field | Description |
|-------|-------------|
| `oare_id` | UUID format (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx) |
| `transliteration` | Akkadian text in transliteration format |
| `translation` | English translation |

### 2.2 Statistics

| Metric | Value |
|--------|-------|
| Total rows | 1,561 |
| Unique documents | 1,561 |
| Min token count | 3 |
| Max token count | 187 |
| Mean token count | 57.5 |
| Median token count | 49 |

### 2.3 Length Distribution

```
Percentile | Token Count
-----------|------------
5%         | 12
25%        | 28
50%        | 49
75%        | 84
95%        | 140
```

### 2.4 Special Characters in Train

| Pattern | Count | Purpose |
|---------|-------|---------|
| `...` (ellipsis) | 1,687 | Damaged/missing text |
| ` x ` (isolated x) | 1,807 | Unreadable signs |
| `[...]` | 210 | Reconstructed text |
| `₄, ₆, ₂` etc. | 2,380+ | Subscript numerals for sign variants |

---

## 3. Train vs Test Distribution Comparison

### 3.1 Length Comparison

| Metric | Train | Test |
|--------|-------|------|
| Mean tokens | 57.5 | 21.2 |
| Median tokens | 49 | 17.5 |
| Min tokens | 3 | 16 |
| Max tokens | 187 | 34 |

**Critical Issue**: Test segments are **~37% the length** of average train documents.

### 3.2 Encoding Differences

| Character | Train | Test | Unicode |
|-----------|-------|------|---------|
| `₄` (subscript 4) | Common | Absent | U+2084 |
| `„` (double low quote) | Absent | Present (6x) | U+201E |
| `+` | Absent | Present (1x) | Standard |

The `„` character in test appears to be an **alternative encoding** for subscript numerals:
- `qí-bi„-ma` (test) corresponds to `qí-bi₄-ma` (train)

### 3.3 Vocabulary Overlap Analysis

| Metric | Value |
|--------|-------|
| Test vocabulary size | 59 unique tokens |
| Train vocabulary size | 11,761 unique tokens |
| Overlap | 32 tokens |
| Test-only tokens | 27 tokens |
| Overlap percentage | 54.2% |

---

## 4. Supplementary Data Analysis

### 4.1 Sentences_Oare_FirstWord_LinNum.csv

**Key for Line-Based Training Data Creation**

| Metric | Value |
|--------|-------|
| Total sentences | 9,782 |
| Unique texts | 1,700 |
| Sentences per text | 1-201 (mean: 5.8) |
| Line number range | 1-113 |

**Columns Available**:
- `text_uuid`: Full UUID linking to texts
- `sentence_uuid`: Unique sentence ID
- `translation`: English translation
- `line_number`: Line number on tablet
- `first_word_spelling`: Transliteration of first word
- `side`, `column`: Position on tablet

**Connection to Train**: 253 of 1,561 train texts have sentence-level data

**Strategic Value**: Can reconstruct **line-segmented training pairs** similar to test format!

### 4.2 published_texts.csv

| Metric | Value |
|--------|-------|
| Total texts | 7,953 |
| With transliteration | 7,953 (100%) |
| With AICC_translation | 7,702 (96.8%) |
| With <big_gap> markers | 3,298 (41.5%) |

**Key Columns**:
- `oare_id`: UUID linking to train data
- `transliteration`: Full text (different gap notation)
- `transliteration_orig`: Original notation
- `genre_label`: Text type classification

**Genre Distribution**:
- Unknown: 4,046 (50.9%)
- Letter: 2,261 (28.4%)
- Debt note: 527 (6.6%)
- Note: 218 (2.7%)
- Agreement/Contract: 128 (1.6%)

### 4.3 OA_Lexicon_eBL.csv

| Metric | Value |
|--------|-------|
| Total entries | 39,332 |
| Words | 25,574 |
| Personal names (PN) | 13,424 |
| Geographic names (GN) | 334 |
| Unique forms | 35,048 |
| Unique lexemes | 6,353 |

**Mapping Available**: `form` (spelling) -> `norm` (normalized) -> `lexeme` (dictionary entry)

### 4.4 eBL_Dictionary.csv

| Metric | Value |
|--------|-------|
| Total entries | 19,215 |
| With definitions | 19,215 |
| With derivation info | Variable |

### 4.5 publications.csv

| Metric | Value |
|--------|-------|
| Total pages | 216,602 |
| Unique PDFs | 952 |
| Pages with Akkadian | 31,286 (14.4%) |

**Content**: OCR text from academic publications - could provide additional context.

---

## 5. Data Connections Map

```
train.csv (oare_id)
    |
    |-- 100% match --> published_texts.csv (oare_id)
    |                        |
    |-- 16.2% match ------> Sentences_Oare.csv (text_uuid)
    |                              |
    |                              |-- line_number data
    |                              |-- translation data
    |
    +-- NO MATCH --> test.csv (text_id: 332fda50)
                         |
                         |-- ORPHAN - Held-out document
```

---

## 6. Critical Insights

### Insight 1: Test is a LINE-SEGMENTED, HELD-OUT Document

The test data represents a **completely new document** segmented by line ranges. This means:
- We cannot lookup this document's translation anywhere
- Models must generalize to unseen vocabulary
- The translation must be inferred purely from learned patterns

### Insight 2: Length Distribution Shift

Training on full documents (avg 57 tokens) and testing on segments (avg 21 tokens) creates a **distribution mismatch**. Solutions:
- Segment train data by lines using Sentences_Oare
- Use sliding window to create shorter training examples
- Fine-tune on similar-length examples

### Insight 3: Encoding Normalization Required

The `„` character must be normalized to match train encoding (`₄`) before inference, or both formats must be supported.

### Insight 4: Sentences_Oare is KEY for Matching Test Format

Only Sentences_Oare provides:
- Line-level granularity
- Individual sentence translations
- Ability to reconstruct segment-based training pairs

### Insight 5: Low Vocabulary Overlap

With only 54.2% vocabulary overlap, the model must:
- Handle OOV tokens gracefully
- Potentially use character-level features
- Leverage lexicon for unknown word forms

---

## 7. Recommended Data Strategy

### 7.1 Preprocessing Pipeline

```python
# 1. Normalize encoding
def normalize_encoding(text):
    # Replace test-specific chars with train equivalents
    text = text.replace('„', '₄')  # Special quote -> subscript 4
    return text

# 2. Handle gap markers
def normalize_gaps(text):
    text = text.replace('<big_gap>', '...')
    text = text.replace('{large break}', '...')
    return text
```

### 7.2 Training Data Augmentation

**Option A: Create Line-Segmented Training Data**
```python
# Use Sentences_Oare to create segments similar to test
for text_uuid in sentences['text_uuid'].unique():
    text_sentences = sentences[sentences['text_uuid'] == text_uuid]
    # Group sentences by line ranges (e.g., lines 1-7, 7-14, etc.)
    # Create training pairs: (segmented_transliteration, combined_translations)
```

**Option B: Sliding Window on Train Data**
```python
# Create shorter segments from full documents
for doc in train:
    tokens = doc['transliteration'].split()
    for i in range(0, len(tokens), window_size):
        segment = tokens[i:i+window_size]
        # Need corresponding translation segment (harder without alignment)
```

### 7.3 Recommended Approach Priority

1. **HIGH**: Use Sentences_Oare to create line-segmented training pairs
2. **MEDIUM**: Filter train to shorter documents (15-35 tokens)
3. **LOW**: Use published_texts for vocabulary augmentation
4. **SUPPORT**: Use lexicon for OOV token handling

---

## 8. File Reference Summary

| File | Rows | Key Columns | Use Case |
|------|------|-------------|----------|
| train.csv | 1,561 | oare_id, transliteration, translation | Primary training |
| test.csv | 4 | text_id, line_start, line_end, transliteration | Inference target |
| sample_submission.csv | 4 | id, translation | Submission format |
| published_texts.csv | 7,953 | oare_id, transliteration | Vocabulary expansion |
| Sentences_Oare.csv | 9,782 | text_uuid, line_number, translation | Line-segmented data |
| OA_Lexicon_eBL.csv | 39,332 | form, norm, lexeme | Token normalization |
| eBL_Dictionary.csv | 19,215 | word, definition | Word definitions |
| publications.csv | 216,602 | page_text, has_akkadian | Academic context |

---

## 9. Expected Translations (from sample_submission.csv)

The sample submission shows the expected translation style:

**Row 0 (lines 1-7)**:
```
Thus Kanesh, say to the -payers, our messenger, every single colony, and the trading stations: A letter of the City has arrived.
```

**Row 1 (lines 7-14)**:
```
In the letter of the City (it is written): From this day on, whoever buys meteoric iron, (the City of) Assur is not part of the profit made, the tithe on it Kanesh will collect.
```

**Row 2 (lines 14-24)**:
```
As soon as you have heard our letter, who(ever) over there has either sold it to a palace, or has offered it to palace officials, or still carries it with him without having yet sold it - all meteoric iron he carries, write the exact amount of every (piece of) meteoric iron, his name and the name of his father in a tablet and send it here with our messenger.
```

**Row 3 (lines 25-30)**:
```
Send a copy of (this) letter of ours to every single colony and to all the trading stations. Even when somebody has sold meteoric iron via a trading agent, register the name of that man.
```

**Note**: These translations reveal the document is an **official decree about meteoric iron trade regulation**.

---

## 10. Conclusion

The Akkadian translation competition presents a **domain shift challenge** where:
- Test data is structurally different (line segments vs full documents)
- Test data is from a held-out source (no lookup possible)
- Vocabulary overlap is limited (54.2%)

**Success will require**:
1. Creating line-segmented training data from Sentences_Oare
2. Proper encoding normalization
3. Robust handling of OOV tokens
4. Models that generalize well to shorter text segments

The key insight is that **Sentences_Oare provides the bridge** between full-document training data and line-segmented test data.

---

*Report generated by Claude Code Deep Research EDA*
