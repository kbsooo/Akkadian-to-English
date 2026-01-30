"""
Akkadian V2: Unified Normalization Module

This module provides normalization functions that work identically
for both training and inference, resolving the Train/Test style mismatch.

Key insight: Train data uses `š`, `ṣ`, `ḫ`, `ṭ` while Test data uses
grave accents and different OCR artifacts. We normalize BOTH to ASCII.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

# ==============================================================================
# Character Mapping Tables
# ==============================================================================

# Vowels with diacritics → base vowels (using Unicode escapes for safety)
_VOWEL_MAP = {
    # a variants
    '\u00e0': 'a', '\u00e1': 'a', '\u00e2': 'a', '\u0101': 'a', '\u00e4': 'a',
    '\u00c0': 'A', '\u00c1': 'A', '\u00c2': 'A', '\u0100': 'A', '\u00c4': 'A',
    # e variants
    '\u00e8': 'e', '\u00e9': 'e', '\u00ea': 'e', '\u0113': 'e', '\u00eb': 'e',
    '\u00c8': 'E', '\u00c9': 'E', '\u00ca': 'E', '\u0112': 'E', '\u00cb': 'E',
    # i variants  
    '\u00ec': 'i', '\u00ed': 'i', '\u00ee': 'i', '\u012b': 'i', '\u00ef': 'i',
    '\u00cc': 'I', '\u00cd': 'I', '\u00ce': 'I', '\u012a': 'I', '\u00cf': 'I',
    # o variants
    '\u00f2': 'o', '\u00f3': 'o', '\u00f4': 'o', '\u014d': 'o', '\u00f6': 'o',
    '\u00d2': 'O', '\u00d3': 'O', '\u00d4': 'O', '\u014c': 'O', '\u00d6': 'O',
    # u variants
    '\u00f9': 'u', '\u00fa': 'u', '\u00fb': 'u', '\u016b': 'u', '\u00fc': 'u',
    '\u00d9': 'U', '\u00da': 'U', '\u00db': 'U', '\u016a': 'U', '\u00dc': 'U',
}

# Special Akkadian consonants → ASCII
_CONSONANT_MAP = {
    '\u0161': 's', '\u0160': 'S',  # š, Š (shin)
    '\u1e63': 's', '\u1e62': 'S',  # ṣ, Ṣ (tsade)
    '\u1e6d': 't', '\u1e6c': 'T',  # ṭ, Ṭ (emphatic t)
    '\u1e2b': 'h', '\u1e2a': 'H',  # ḫ, Ḫ (het)
}

# OCR artifacts and typography (using Unicode escapes)
_OCR_MAP = {
    '\u201e': '"',   # „ German low quote
    '\u201c': '"',   # " left double quote
    '\u201d': '"',   # " right double quote
    '\u2018': "'",   # ' left single quote
    '\u2019': "'",   # ' right single quote
    '\u201a': "'",   # ‚ single low quote
    '\u02be': "'",   # ʾ aleph (modifier letter right half ring)
    '\u02bf': "'",   # ʿ ayin (modifier letter left half ring)
    '\u2308': '[',   # ⌈ left ceiling (half bracket)
    '\u2309': ']',   # ⌉ right ceiling
    '\u230a': '[',   # ⌊ left floor
    '\u230b': ']',   # ⌋ right floor
}

# Subscripts → numbers
_SUBSCRIPT_MAP = str.maketrans({
    '\u2080': '0', '\u2081': '1', '\u2082': '2', '\u2083': '3', '\u2084': '4',
    '\u2085': '5', '\u2086': '6', '\u2087': '7', '\u2088': '8', '\u2089': '9',
    '\u2093': 'x',
})

# Combined translation table
_FULL_MAP = str.maketrans({**_VOWEL_MAP, **_CONSONANT_MAP, **_OCR_MAP})


# ==============================================================================
# Normalization Functions
# ==============================================================================

def normalize_transliteration(text: Optional[str]) -> str:
    """
    Normalize Akkadian transliteration to ASCII-compatible format.
    
    This function ensures Train and Test data use the same character set.
    
    Transformations:
    1. Unicode NFC normalization
    2. Diacritics removal (à → a, š → s, etc.)
    3. OCR artifact cleanup (curly quotes → straight quotes)
    4. Subscript normalization (₄ → 4)
    5. Gap/damage markers ([...] → <gap>)
    6. Unknown sign markers (x → <unk>)
    7. Editorial mark removal (!?/)
    8. Whitespace normalization
    
    Args:
        text: Raw transliteration string
        
    Returns:
        Normalized string ready for tokenization
    """
    if text is None or (isinstance(text, float) and text != text):  # NaN check
        return ""
    
    text = str(text)
    
    # 1. Unicode normalization
    text = unicodedata.normalize("NFC", text)
    
    # 2. Apply character mappings (diacritics, consonants, OCR)
    text = text.translate(_FULL_MAP)
    
    # 3. Subscript normalization
    text = text.translate(_SUBSCRIPT_MAP)
    
    # 4. Handle ellipsis and big gaps
    text = text.replace('\u2026', ' <gap> ')  # …
    text = re.sub(r'\.\.\.+', ' <gap> ', text)
    
    # 5. Handle bracketed content (damaged/reconstructed)
    text = re.sub(r'\[([^\]]*)\]', ' <gap> ', text)
    
    # 6. Handle unknown signs
    text = re.sub(r'\bx\b', ' <unk> ', text, flags=re.IGNORECASE)
    
    # 7. Remove editorial marks
    text = re.sub(r'[!?/]', ' ', text)
    
    # 8. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def normalize_translation(text: Optional[str]) -> str:
    """
    Normalize English translation text.
    
    Args:
        text: Raw English translation
        
    Returns:
        Normalized translation
    """
    if text is None or (isinstance(text, float) and text != text):
        return ""
    
    text = str(text)
    
    # Unicode normalization
    text = unicodedata.normalize("NFC", text)
    
    # Normalize quotes (using Unicode escapes)
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ==============================================================================
# Validation & Testing
# ==============================================================================

def validate_normalization():
    """Run validation tests for normalization functions."""
    test_cases = [
        # (input, expected_output)
        ('\u0161u-ma', 'su-ma'),  # šu-ma
        ('\u1e63\u00ed-l\u00e1', 'si-la'),  # ṣí-lá
        ('\u1e2ba-mu-u\u0161', 'ha-mu-us'),  # ḫa-mu-uš
        ('q\u00ed-bi\u201e-ma', 'qi-bi"-ma'),  # qí-bi„-ma
        ('k\u00e0-ru-um', 'ka-ru-um'),  # kà-ru-um
        ('a-na aa-q\u00ed-il', 'a-na aa-qi-il'),  # a-na aa-qí-il
        ('[...]', '<gap>'),
        ('u\u2084-me-\u0161u', 'u4-me-su'),  # u₄-me-šu
        ('KI\u0160IB', 'KISIB'),  # KIŠIB
        ('...text...', '<gap> text <gap>'),
    ]
    
    print("Running normalization tests...")
    all_passed = True
    
    for input_text, expected in test_cases:
        result = normalize_transliteration(input_text)
        if result != expected:
            print(f"  ❌ FAIL: '{input_text}' → '{result}' (expected '{expected}')")
            all_passed = False
        else:
            print(f"  ✅ PASS: '{input_text}' → '{result}'")
    
    if all_passed:
        print("\n✅ All normalization tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return all_passed


if __name__ == "__main__":
    validate_normalization()
