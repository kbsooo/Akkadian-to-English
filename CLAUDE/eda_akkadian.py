#!/usr/bin/env python3
"""
Deep Past Challenge - Akkadian to English Translation
Exploratory Data Analysis (EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
DATA_DIR = "/sessions/loving-brave-allen/mnt/akkadian/data"
OUTPUT_DIR = "/sessions/loving-brave-allen/mnt/akkadian/CLAUDE"

print("=" * 60)
print("Deep Past Challenge - EDA")
print("Akkadian to English Machine Translation")
print("=" * 60)

# =============================================================================
# 1. Load and Explore Main Files
# =============================================================================

print("\n" + "=" * 60)
print("1. DATA FILES OVERVIEW")
print("=" * 60)

# List all files with sizes
files = os.listdir(DATA_DIR)
file_info = []
for f in files:
    path = os.path.join(DATA_DIR, f)
    size = os.path.getsize(path)
    file_info.append({'file': f, 'size_mb': size / (1024 * 1024)})

file_df = pd.DataFrame(file_info).sort_values('size_mb', ascending=False)
print("\nFile Sizes:")
for _, row in file_df.iterrows():
    print(f"  {row['file']:<40} {row['size_mb']:>10.2f} MB")

# =============================================================================
# 2. Train Data Analysis
# =============================================================================

print("\n" + "=" * 60)
print("2. TRAIN DATA ANALYSIS")
print("=" * 60)

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
print(f"\nShape: {train.shape}")
print(f"Columns: {list(train.columns)}")
print(f"\nData Types:\n{train.dtypes}")
print(f"\nMissing Values:\n{train.isnull().sum()}")
print(f"\nSample Data (first 3 rows):")
print(train.head(3).to_string())

# Text length analysis
train['trans_len'] = train['transliteration'].str.len()
train['transl_len'] = train['translation'].str.len()
train['trans_words'] = train['transliteration'].str.split().str.len()
train['transl_words'] = train['translation'].str.split().str.len()

print(f"\n--- Transliteration (Akkadian) Statistics ---")
print(f"  Character length: min={train['trans_len'].min()}, max={train['trans_len'].max()}, mean={train['trans_len'].mean():.1f}, median={train['trans_len'].median():.1f}")
print(f"  Word count: min={train['trans_words'].min()}, max={train['trans_words'].max()}, mean={train['trans_words'].mean():.1f}, median={train['trans_words'].median():.1f}")

print(f"\n--- Translation (English) Statistics ---")
print(f"  Character length: min={train['transl_len'].min()}, max={train['transl_len'].max()}, mean={train['transl_len'].mean():.1f}, median={train['transl_len'].median():.1f}")
print(f"  Word count: min={train['transl_words'].min()}, max={train['transl_words'].max()}, mean={train['transl_words'].mean():.1f}, median={train['transl_words'].median():.1f}")

# Unique OARE IDs
print(f"\nUnique oare_id count: {train['oare_id'].nunique()}")

# =============================================================================
# 3. Test Data Analysis
# =============================================================================

print("\n" + "=" * 60)
print("3. TEST DATA ANALYSIS")
print("=" * 60)

test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(f"\nShape: {test.shape}")
print(f"Columns: {list(test.columns)}")
print(f"\nSample Data:")
print(test.head().to_string())

test['trans_len'] = test['transliteration'].str.len()
test['trans_words'] = test['transliteration'].str.split().str.len()

print(f"\n--- Test Transliteration Statistics ---")
print(f"  Character length: min={test['trans_len'].min()}, max={test['trans_len'].max()}, mean={test['trans_len'].mean():.1f}")
print(f"  Word count: min={test['trans_words'].min()}, max={test['trans_words'].max()}, mean={test['trans_words'].mean():.1f}")

print(f"\nUnique text_id count: {test['text_id'].nunique()}")
print(f"Test sentences per document: {len(test) / test['text_id'].nunique():.1f}")

# =============================================================================
# 4. Sample Submission Analysis
# =============================================================================

print("\n" + "=" * 60)
print("4. SAMPLE SUBMISSION")
print("=" * 60)

sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
print(f"\nShape: {sample_sub.shape}")
print(f"Columns: {list(sample_sub.columns)}")
print(f"\nSample:")
print(sample_sub.head().to_string())

# =============================================================================
# 5. Supplemental Data Analysis
# =============================================================================

print("\n" + "=" * 60)
print("5. SUPPLEMENTAL DATA ANALYSIS")
print("=" * 60)

# Published texts
print("\n--- published_texts.csv ---")
pub_texts = pd.read_csv(os.path.join(DATA_DIR, "published_texts.csv"))
print(f"Shape: {pub_texts.shape}")
print(f"Columns: {list(pub_texts.columns)}")
print(f"Sample (2 rows):")
print(pub_texts.head(2).to_string())

# Lexicon
print("\n--- OA_Lexicon_eBL.csv ---")
lexicon = pd.read_csv(os.path.join(DATA_DIR, "OA_Lexicon_eBL.csv"))
print(f"Shape: {lexicon.shape}")
print(f"Columns: {list(lexicon.columns)}")
print(f"Word types distribution:")
print(lexicon['type'].value_counts().head(10))

# Dictionary
print("\n--- eBL_Dictionary.csv ---")
dictionary = pd.read_csv(os.path.join(DATA_DIR, "eBL_Dictionary.csv"))
print(f"Shape: {dictionary.shape}")
print(f"Columns: {list(dictionary.columns)}")

# Sentences helper
print("\n--- Sentences_Oare_FirstWord_LinNum.csv ---")
sentences = pd.read_csv(os.path.join(DATA_DIR, "Sentences_Oare_FirstWord_LinNum.csv"))
print(f"Shape: {sentences.shape}")
print(f"Columns: {list(sentences.columns)}")

# Bibliography
print("\n--- bibliography.csv ---")
bib = pd.read_csv(os.path.join(DATA_DIR, "bibliography.csv"))
print(f"Shape: {bib.shape}")
print(f"Columns: {list(bib.columns)}")

# Resources
print("\n--- resources.csv ---")
resources = pd.read_csv(os.path.join(DATA_DIR, "resources.csv"))
print(f"Shape: {resources.shape}")
print(f"Columns: {list(resources.columns)}")

# Publications (large file - just check structure)
print("\n--- publications.csv ---")
pubs_sample = pd.read_csv(os.path.join(DATA_DIR, "publications.csv"), nrows=5)
print(f"Columns: {list(pubs_sample.columns)}")
print("(Large file - 580MB, contains OCR from 880 scholarly publications)")

# =============================================================================
# 6. Akkadian Text Patterns Analysis
# =============================================================================

print("\n" + "=" * 60)
print("6. AKKADIAN TEXT PATTERNS")
print("=" * 60)

# Common patterns in transliteration
all_trans = ' '.join(train['transliteration'].dropna())

# Find special markers
gap_count = all_trans.count('<gap>')
big_gap_count = all_trans.count('<big_gap>')
curly_bracket = len(re.findall(r'\{[^}]+\}', all_trans))

print(f"\nSpecial markers in training data:")
print(f"  <gap> occurrences: {gap_count}")
print(f"  <big_gap> occurrences: {big_gap_count}")
print(f"  Curly bracket expressions: {curly_bracket}")

# Word frequency
words = all_trans.lower().split()
word_freq = Counter(words)
print(f"\nTotal unique words in transliteration: {len(word_freq)}")
print(f"Top 20 most common words:")
for word, count in word_freq.most_common(20):
    print(f"  {word}: {count}")

# Common Akkadian patterns
print("\nChecking for common Akkadian elements:")
determinatives = re.findall(r'\{[^}]+\}', all_trans)
det_freq = Counter(determinatives)
print(f"Top 10 determinatives (curly brackets):")
for det, count in det_freq.most_common(10):
    print(f"  {det}: {count}")

# =============================================================================
# 7. Translation (English) Patterns
# =============================================================================

print("\n" + "=" * 60)
print("7. ENGLISH TRANSLATION PATTERNS")
print("=" * 60)

all_transl = ' '.join(train['translation'].dropna())
eng_words = all_transl.lower().split()
eng_freq = Counter(eng_words)

print(f"\nTotal unique words in translations: {len(eng_freq)}")
print(f"Top 20 most common words:")
for word, count in eng_freq.most_common(20):
    print(f"  {word}: {count}")

# Check for special characters in translations
special_chars = set(re.findall(r'[^\w\s.,;:!?\'\"-]', all_transl))
print(f"\nSpecial characters in translations: {special_chars}")

# Patterns like [...] indicating gaps
brackets = len(re.findall(r'\[[^\]]+\]', all_transl))
ellipsis = all_transl.count('...')
print(f"\nBracket expressions [...]: {brackets}")
print(f"Ellipsis (...): {ellipsis}")

# =============================================================================
# 8. Create Visualizations
# =============================================================================

print("\n" + "=" * 60)
print("8. CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Deep Past Challenge - Akkadian to English Translation EDA', fontsize=14, fontweight='bold')

# Plot 1: Transliteration length distribution
axes[0, 0].hist(train['trans_len'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Character Length')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Akkadian Transliteration Length Distribution')
axes[0, 0].axvline(train['trans_len'].median(), color='red', linestyle='--', label=f'Median: {train["trans_len"].median():.0f}')
axes[0, 0].legend()

# Plot 2: Translation length distribution
axes[0, 1].hist(train['transl_len'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_xlabel('Character Length')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('English Translation Length Distribution')
axes[0, 1].axvline(train['transl_len'].median(), color='red', linestyle='--', label=f'Median: {train["transl_len"].median():.0f}')
axes[0, 1].legend()

# Plot 3: Transliteration vs Translation length
axes[0, 2].scatter(train['trans_len'], train['transl_len'], alpha=0.5, s=10)
axes[0, 2].set_xlabel('Akkadian Length (chars)')
axes[0, 2].set_ylabel('English Length (chars)')
axes[0, 2].set_title('Source vs Target Length Correlation')
# Add correlation coefficient
corr = train['trans_len'].corr(train['transl_len'])
axes[0, 2].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[0, 2].transAxes, fontsize=10, verticalalignment='top')

# Plot 4: Word count distribution
axes[1, 0].hist(train['trans_words'], bins=50, edgecolor='black', alpha=0.7, label='Akkadian')
axes[1, 0].hist(train['transl_words'], bins=50, edgecolor='black', alpha=0.5, label='English')
axes[1, 0].set_xlabel('Word Count')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Word Count Distribution')
axes[1, 0].legend()

# Plot 5: Top 10 Akkadian words
top_words = word_freq.most_common(15)
words_list = [w[0] for w in top_words]
counts_list = [w[1] for w in top_words]
axes[1, 1].barh(range(len(words_list)), counts_list, color='steelblue')
axes[1, 1].set_yticks(range(len(words_list)))
axes[1, 1].set_yticklabels(words_list)
axes[1, 1].invert_yaxis()
axes[1, 1].set_xlabel('Frequency')
axes[1, 1].set_title('Top 15 Akkadian Words')

# Plot 6: Lexicon word types
type_counts = lexicon['type'].value_counts().head(8)
axes[1, 2].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
axes[1, 2].set_title('Lexicon Word Types Distribution')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_visualizations.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {os.path.join(OUTPUT_DIR, 'eda_visualizations.png')}")

# =============================================================================
# 9. Key Findings Summary
# =============================================================================

print("\n" + "=" * 60)
print("9. KEY FINDINGS SUMMARY")
print("=" * 60)

print("""
1. COMPETITION TASK:
   - Translate Old Assyrian (Akkadian) transliterations to English
   - Training: ~1500 document-level translations
   - Test: ~4000 sentence-level translations from ~400 documents
   - Evaluation: Geometric mean of BLEU and chrF++ scores

2. DATA CHARACTERISTICS:
   - Train has document-level alignment (oare_id, transliteration, translation)
   - Test has sentence-level alignment (id, text_id, line_start, line_end, transliteration)
   - Key challenge: Train is document-level but test is sentence-level

3. TEXT PATTERNS:
   - Special markers: <gap>, <big_gap> for broken/missing text
   - Determinatives in curly brackets {ki}, {d}, etc. (semantic classifiers)
   - Line numbers in transliteration (1, 5, 10, etc.)
   - Translations may contain [...] for gaps and proper nouns

4. SUPPLEMENTAL RESOURCES:
   - published_texts.csv: 8,000 transliterations without translations
   - OA_Lexicon_eBL.csv: Word dictionary with ~35K entries
   - publications.csv: 880 scholarly PDFs (OCR text) - potential extra training data
   - Sentences_Oare_FirstWord_LinNum.csv: Helps with sentence alignment

5. CHALLENGES:
   - Low-resource language (limited training data)
   - Morphologically complex language
   - Document-to-sentence alignment mismatch
   - Ancient text gaps and damage
   - Proper noun handling

6. POTENTIAL APPROACHES:
   - Use pre-trained multilingual models (mBART, NLLB, etc.)
   - Data augmentation from publications.csv
   - Sentence-level alignment using Sentences_Oare_FirstWord_LinNum.csv
   - Leverage lexicon for vocabulary enhancement
   - Handle special tokens (<gap>, determinatives) appropriately
""")

print("\n" + "=" * 60)
print("EDA COMPLETE!")
print("=" * 60)
