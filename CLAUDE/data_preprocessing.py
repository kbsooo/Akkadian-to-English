#!/usr/bin/env python3
"""
Akkadian Data Preprocessing Pipeline
Deep Past Initiative - Kaggle Competition

This module implements the data preprocessing strategies outlined in DATA_STRATEGY.md
"""

import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict, Optional
from collections import Counter
import json


# =============================================================================
# 1. CONSTANTS AND MAPPINGS
# =============================================================================

# Sumerogram to semantic token mapping
SUMEROGRAM_MAP = {
    'KÙ.BABBAR': '[SILVER]',
    'KÙ.GI': '[GOLD]',
    'URUDU': '[COPPER]',
    'AN.NA': '[TIN]',
    'TÚG': '[TEXTILE]',
    'ANŠE': '[DONKEY]',
    'GÍN': '[SHEKEL]',
    'MA.NA': '[MINA]',
    'GÚ': '[TALENT]',
    'ITU.KAM': '[MONTH]',
    'ITU': '[MONTH]',
    'DUMU': '[SON]',
    'DAM': '[WIFE]',
    'IGI': '[WITNESS]',
    'KIŠIB': '[SEAL]',
    'É': '[HOUSE]',
    'É.GAL': '[PALACE]',
    'ŠU.NÍGIN': '[TOTAL]',
    'SÍG.ḪI.A': '[WOOL]',
    'TÚG.ḪI.A': '[TEXTILES]',
    'ANŠE.ḪI.A': '[DONKEYS]',
}

# Fraction normalization
FRACTION_MAP = {
    '0.33333': '⅓',
    '0.5': '½',
    '0.66666': '⅔',
    '0.83333': '⅚',
    '0.16666': '⅙',
    '0.25': '¼',
    '0.75': '¾',
}

# Sentence boundary markers (Akkadian patterns)
SENTENCE_BOUNDARIES = [
    r'um-ma\s+[\w\-]+\-ma',      # Quote introduction: "From X-ma"
    r'a-na\s+[\w\-]+\s+qí-bi',   # "To X say"
    r'qí-bi₄?-ma',               # "say:"
    r'\bIGI\b',                   # Witness list
    r'\bKIŠIB\b',                 # Seal list
    r'li-mu-um',                  # Eponymy dating
    r'ITU\.KAM',                  # Month marker
    r'ḫa-muš-tim',                # Week marker
]


# =============================================================================
# 2. TEXT NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_determinatives(text: str) -> str:
    """Standardize determinative markers."""
    text = re.sub(r'\(d\)', '{d}', text)      # Divine names
    text = re.sub(r'\(ki\)', '{ki}', text)    # Place names
    text = re.sub(r'\(f\)', '{f}', text)      # Female names
    text = re.sub(r'\(m\)', '{m}', text)      # Male names
    return text


def normalize_sumerograms(text: str, use_tokens: bool = True) -> str:
    """Replace Sumerograms with semantic tokens or keep as-is."""
    if not use_tokens:
        return text

    for sumero, token in SUMEROGRAM_MAP.items():
        text = re.sub(re.escape(sumero), token, text)
    return text


def normalize_fractions(text: str) -> str:
    """Convert decimal fractions to Unicode fraction characters."""
    for decimal, fraction in FRACTION_MAP.items():
        text = text.replace(decimal, fraction)
    return text


def handle_unclear_markers(text: str) -> str:
    """Standardize unclear/gap markers."""
    text = re.sub(r'\bx\b', '[?]', text)
    text = re.sub(r'<gap>', '[GAP]', text)
    text = re.sub(r'<big_gap>', '[BIG_GAP]', text)
    text = re.sub(r'\.\.\.', '[...]', text)
    return text


def clean_ocr_artifacts(text: str) -> str:
    """Remove or fix OCR artifacts from test data."""
    # Common OCR errors in Akkadian transliterations
    text = text.replace('„', '"')
    text = text.replace('…', '...')
    text = text.replace('+', '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_subscripts(text: str) -> str:
    """Standardize subscript numbers."""
    # Keep subscripts as-is for ByT5 (character-level handles them)
    # But for tokenizer-based models, might need conversion
    return text


def full_preprocessing(text: str,
                       use_sumerogram_tokens: bool = True,
                       normalize_nums: bool = True) -> str:
    """Apply full preprocessing pipeline."""
    text = clean_ocr_artifacts(text)
    text = normalize_determinatives(text)
    if use_sumerogram_tokens:
        text = normalize_sumerograms(text, use_tokens=True)
    if normalize_nums:
        text = normalize_fractions(text)
    text = handle_unclear_markers(text)
    text = normalize_subscripts(text)
    return text


# =============================================================================
# 3. SENTENCE SEGMENTATION
# =============================================================================

def detect_sentence_boundaries(transliteration: str) -> List[int]:
    """Detect potential sentence boundaries in transliteration."""
    boundaries = [0]  # Start of document

    for pattern in SENTENCE_BOUNDARIES:
        for match in re.finditer(pattern, transliteration):
            boundaries.append(match.start())

    boundaries.append(len(transliteration))
    return sorted(set(boundaries))


def segment_by_markers(transliteration: str, translation: str) -> List[Tuple[str, str]]:
    """
    Segment document into sentences using boundary markers.
    Returns list of (transliteration_segment, translation_segment) pairs.
    """
    # Find boundaries in transliteration
    trans_boundaries = detect_sentence_boundaries(transliteration)

    # Create transliteration segments
    trans_segments = []
    for i in range(len(trans_boundaries) - 1):
        start, end = trans_boundaries[i], trans_boundaries[i + 1]
        segment = transliteration[start:end].strip()
        if segment:
            trans_segments.append(segment)

    # Try to align with translation using key phrases
    # Translation boundary markers
    eng_patterns = [
        r'From\s+\w+\s+to',
        r'To\s+\w+:',
        r'Witnessed by',
        r'Witnesses?:',
        r'Seal of',
        r'Month:',
        r'Year:',
    ]

    # Find boundaries in translation
    eng_boundaries = [0]
    for pattern in eng_patterns:
        for match in re.finditer(pattern, translation, re.IGNORECASE):
            eng_boundaries.append(match.start())
    eng_boundaries.append(len(translation))
    eng_boundaries = sorted(set(eng_boundaries))

    # Create translation segments
    eng_segments = []
    for i in range(len(eng_boundaries) - 1):
        start, end = eng_boundaries[i], eng_boundaries[i + 1]
        segment = translation[start:end].strip()
        if segment:
            eng_segments.append(segment)

    # Align segments (simple 1:1 if counts match, otherwise use DTW)
    if len(trans_segments) == len(eng_segments):
        return list(zip(trans_segments, eng_segments))
    else:
        # Fallback: return whole document as single pair
        return [(transliteration, translation)]


def segment_with_sentences_file(doc_id: str,
                                 transliteration: str,
                                 sentences_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Extract sentence pairs using Sentences_Oare_FirstWord_LinNum.csv annotations.
    """
    doc_sentences = sentences_df[sentences_df['text_uuid'] == doc_id].copy()

    if len(doc_sentences) == 0:
        return []

    # Sort by line number
    doc_sentences = doc_sentences.sort_values('line_number')

    pairs = []
    for _, sent in doc_sentences.iterrows():
        translation = sent['translation']
        first_word = sent.get('first_word_transcription', '')

        # Try to extract transliteration segment based on first word
        if first_word and pd.notna(first_word):
            # Find the first word in the transliteration
            pattern = re.escape(first_word)
            matches = list(re.finditer(pattern, transliteration, re.IGNORECASE))

            if matches:
                # Use heuristics to extract the sentence
                start = matches[0].start()
                # End at next sentence or after ~100 chars
                end = min(start + 150, len(transliteration))

                # Try to find natural end (next boundary marker)
                for bp in SENTENCE_BOUNDARIES:
                    boundary_match = re.search(bp, transliteration[start+10:end])
                    if boundary_match:
                        end = start + 10 + boundary_match.start()
                        break

                trans_segment = transliteration[start:end].strip()
                if trans_segment and pd.notna(translation):
                    pairs.append((trans_segment, translation))
        else:
            # If no first word, skip
            continue

    return pairs


# =============================================================================
# 4. LEXICON INTEGRATION
# =============================================================================

class LexiconNormalizer:
    """Normalize Akkadian words using OA_Lexicon_eBL."""

    def __init__(self, lexicon_path: str):
        self.lexicon = pd.read_csv(lexicon_path)
        self.form_to_norm = dict(zip(self.lexicon['form'], self.lexicon['norm']))
        self.form_to_lexeme = dict(zip(self.lexicon['form'], self.lexicon['lexeme']))
        self.lexeme_variants = self._build_variant_map()

    def _build_variant_map(self) -> Dict[str, List[str]]:
        """Build mapping from lexeme to all its spelling variants."""
        variants = {}
        for lexeme in self.lexicon['lexeme'].unique():
            forms = self.lexicon[self.lexicon['lexeme'] == lexeme]['form'].tolist()
            variants[lexeme] = forms
        return variants

    def normalize_word(self, word: str) -> str:
        """Normalize a single word to its standard form."""
        return self.form_to_norm.get(word, word)

    def normalize_text(self, text: str) -> str:
        """Normalize all words in a text."""
        words = text.split()
        normalized = [self.normalize_word(w) for w in words]
        return ' '.join(normalized)

    def get_variants(self, word: str, max_variants: int = 5) -> List[str]:
        """Get spelling variants for a word (for augmentation)."""
        if word in self.form_to_lexeme:
            lexeme = self.form_to_lexeme[word]
            variants = self.lexeme_variants.get(lexeme, [])
            return [v for v in variants if v != word][:max_variants]
        return []

    def augment_with_variants(self, text: str) -> List[str]:
        """Generate augmented versions by substituting word variants."""
        words = text.split()
        augmented = [text]  # Include original

        for i, word in enumerate(words):
            variants = self.get_variants(word, max_variants=2)
            for var in variants:
                new_words = words.copy()
                new_words[i] = var
                augmented.append(' '.join(new_words))

        return augmented


# =============================================================================
# 5. DATA AUGMENTATION
# =============================================================================

def augment_with_dropout(text: str, dropout_prob: float = 0.1) -> str:
    """Randomly drop words to create augmented samples."""
    words = text.split()
    kept = [w for w in words if np.random.random() > dropout_prob]
    return ' '.join(kept) if kept else text


def augment_with_shuffle(text: str, window_size: int = 3) -> str:
    """Shuffle words within local windows."""
    words = text.split()
    if len(words) <= window_size:
        return text

    # Shuffle within windows
    for i in range(0, len(words) - window_size, window_size):
        window = words[i:i+window_size]
        np.random.shuffle(window)
        words[i:i+window_size] = window

    return ' '.join(words)


def create_pseudo_labels(model, published_texts_df: pd.DataFrame,
                        confidence_threshold: float = 0.7) -> List[Dict]:
    """
    Generate pseudo-labeled data from published_texts.

    Args:
        model: Trained translation model with .translate() and .get_confidence() methods
        published_texts_df: DataFrame with 'transliteration' column
        confidence_threshold: Minimum confidence for including pseudo-label

    Returns:
        List of dicts with 'transliteration', 'translation', 'confidence'
    """
    pseudo_pairs = []

    for _, row in published_texts_df.iterrows():
        text = row['transliteration']

        # Skip texts with gaps
        if '<gap>' in text or '<big_gap>' in text:
            continue

        # Get translation and confidence
        translation = model.translate(text)
        confidence = model.get_confidence(translation)

        if confidence >= confidence_threshold:
            pseudo_pairs.append({
                'transliteration': text,
                'translation': translation,
                'confidence': confidence,
                'source': 'pseudo'
            })

    return pseudo_pairs


# =============================================================================
# 6. QUALITY VALIDATION
# =============================================================================

def extract_numbers(text: str) -> List[str]:
    """Extract all numbers from text."""
    return re.findall(r'\d+(?:\.\d+)?', text)


def check_number_consistency(trans: str, eng: str) -> bool:
    """Check if numbers are preserved in translation."""
    trans_nums = set(extract_numbers(trans))
    eng_nums = set(extract_numbers(eng))

    # Allow for minor differences (fractions converted, etc.)
    if not trans_nums and not eng_nums:
        return True

    overlap = len(trans_nums & eng_nums) / max(len(trans_nums), 1)
    return overlap > 0.5


def check_length_ratio(trans: str, eng: str,
                       min_ratio: float = 0.3,
                       max_ratio: float = 3.0) -> bool:
    """Check if translation length is reasonable."""
    if len(eng) == 0:
        return False
    ratio = len(trans) / len(eng)
    return min_ratio <= ratio <= max_ratio


def check_sumerogram_translation(trans: str, eng: str) -> bool:
    """Check if Sumerograms are translated."""
    # Key Sumerograms and their expected translations
    sumero_eng = {
        'KÙ.BABBAR': 'silver',
        'DUMU': 'son',
        'KIŠIB': 'seal',
        'IGI': 'witness',
    }

    for sumero, eng_word in sumero_eng.items():
        if sumero in trans:
            if eng_word.lower() not in eng.lower():
                return False
    return True


def validate_pair(trans: str, eng: str) -> Tuple[bool, List[str]]:
    """
    Validate a translation pair.
    Returns (is_valid, list_of_issues)
    """
    issues = []

    if not check_length_ratio(trans, eng):
        issues.append('length_ratio')

    if not check_number_consistency(trans, eng):
        issues.append('number_mismatch')

    if not check_sumerogram_translation(trans, eng):
        issues.append('sumerogram_untranslated')

    # Check for empty or too short
    if len(trans.strip()) < 10:
        issues.append('trans_too_short')
    if len(eng.strip()) < 5:
        issues.append('eng_too_short')

    return len(issues) == 0, issues


# =============================================================================
# 7. MAIN PREPROCESSING PIPELINE
# =============================================================================

class AkkadianPreprocessor:
    """Main preprocessing pipeline for Akkadian translation data."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.train_df = None
        self.sentences_df = None
        self.lexicon = None

    def load_data(self):
        """Load all required datasets."""
        self.train_df = pd.read_csv(f'{self.data_dir}/train.csv')
        self.sentences_df = pd.read_csv(f'{self.data_dir}/Sentences_Oare_FirstWord_LinNum.csv')
        self.lexicon = LexiconNormalizer(f'{self.data_dir}/OA_Lexicon_eBL.csv')
        self.published_df = pd.read_csv(f'{self.data_dir}/published_texts.csv')

        print(f"Loaded {len(self.train_df)} training documents")
        print(f"Loaded {len(self.sentences_df)} sentence annotations")
        print(f"Loaded {len(self.published_df)} published texts")

    def create_sentence_pairs(self) -> pd.DataFrame:
        """Create sentence-level training pairs from document-level data."""
        all_pairs = []

        # Get IDs that have sentence annotations
        annotated_ids = set(self.sentences_df['text_uuid'])

        for _, doc in self.train_df.iterrows():
            doc_id = doc['oare_id']
            trans = doc['transliteration']
            eng = doc['translation']

            if doc_id in annotated_ids:
                # Use sentence annotations
                pairs = segment_with_sentences_file(
                    doc_id, trans, self.sentences_df
                )
            else:
                # Use rule-based segmentation
                pairs = segment_by_markers(trans, eng)

            for t, e in pairs:
                all_pairs.append({
                    'oare_id': doc_id,
                    'transliteration': t,
                    'translation': e,
                    'source': 'annotated' if doc_id in annotated_ids else 'rule_based'
                })

        df = pd.DataFrame(all_pairs)
        print(f"Created {len(df)} sentence pairs")
        return df

    def preprocess_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to all pairs."""
        df = df.copy()

        # Preprocess transliterations
        df['trans_processed'] = df['transliteration'].apply(full_preprocessing)

        # Validate pairs
        validations = df.apply(
            lambda row: validate_pair(row['transliteration'], row['translation']),
            axis=1
        )
        df['is_valid'] = [v[0] for v in validations]
        df['issues'] = [v[1] for v in validations]

        valid_count = df['is_valid'].sum()
        print(f"Valid pairs: {valid_count}/{len(df)} ({valid_count/len(df)*100:.1f}%)")

        return df

    def augment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply augmentation strategies."""
        augmented = []

        for _, row in df.iterrows():
            # Original pair
            augmented.append({
                'transliteration': row['transliteration'],
                'translation': row['translation'],
                'augmentation': 'none'
            })

            # Lexicon variants
            variants = self.lexicon.augment_with_variants(row['transliteration'])
            for var in variants[1:3]:  # Max 2 variants
                augmented.append({
                    'transliteration': var,
                    'translation': row['translation'],
                    'augmentation': 'lexicon_variant'
                })

        return pd.DataFrame(augmented)

    def run_pipeline(self, output_dir: str):
        """Run the complete preprocessing pipeline."""
        print("="*50)
        print("AKKADIAN DATA PREPROCESSING PIPELINE")
        print("="*50)

        # Step 1: Load data
        print("\n[1/4] Loading data...")
        self.load_data()

        # Step 2: Create sentence pairs
        print("\n[2/4] Creating sentence pairs...")
        sentence_df = self.create_sentence_pairs()

        # Step 3: Preprocess
        print("\n[3/4] Preprocessing...")
        processed_df = self.preprocess_all(sentence_df)

        # Step 4: Augment
        print("\n[4/4] Augmenting...")
        final_df = self.augment_data(processed_df[processed_df['is_valid']])

        # Save outputs
        processed_df.to_csv(f'{output_dir}/processed_sentences.csv', index=False)
        final_df.to_csv(f'{output_dir}/augmented_train.csv', index=False)

        print(f"\n{'='*50}")
        print(f"PIPELINE COMPLETE")
        print(f"Processed sentences: {len(processed_df)}")
        print(f"Augmented training samples: {len(final_df)}")
        print(f"Output saved to: {output_dir}")
        print(f"{'='*50}")

        return final_df


# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    import os

    # Paths
    DATA_DIR = '../data'
    OUTPUT_DIR = '.'

    # Run pipeline
    preprocessor = AkkadianPreprocessor(DATA_DIR)
    result_df = preprocessor.run_pipeline(OUTPUT_DIR)

    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"\nAugmentation distribution:")
    print(result_df['augmentation'].value_counts())

    print(f"\nTransliteration length stats:")
    print(result_df['transliteration'].str.len().describe())
