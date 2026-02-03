"""V5 normalization utilities (consistent with V4b rules + fixes)."""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

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


def normalize_transliteration(text: Optional[str]) -> str:
    """
    V5 normalization (competition-aligned):
    - [x] -> <gap>
    - … and [… …] -> <big_gap>
    - [content] -> content
    - <content> -> content
    - remove half brackets, scribal notations (!?/), word divider (:)
    - map diacritics + subscripts
    - protect literal <gap>/<big_gap>
    - remove apostrophe-only line numbers (1', 1'')
    """
    if text is None or (isinstance(text, float) and text != text):
        return ""

    text = str(text)
    text = unicodedata.normalize("NFC", text)

    # Protect literal gap tokens before removing <content>
    text = text.replace("<gap>", "__LIT_GAP__")
    text = text.replace("<big_gap>", "__LIT_BIG_GAP__")

    # Remove apostrophe line numbers only (keep numeric quantities)
    text = re.sub(r"\b\d+'{1,2}\b", " ", text)

    # Remove <content> blocks first (scribal insertions)
    text = re.sub(r"<<([^>]+)>>", r"\1", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)

    # Large gaps
    text = re.sub(r"\[\s*\u2026+\s*\u2026*\s*\]", " __BIG_GAP__ ", text)
    text = re.sub(r"\[\s*\.\.\.+\s*\.\.\.+\s*\]", " __BIG_GAP__ ", text)

    # Ellipsis
    text = text.replace("\u2026", " __BIG_GAP__ ")
    text = re.sub(r"\.\.\.+", " __BIG_GAP__ ", text)

    # [x]
    text = re.sub(r"\[\s*x\s*\]", " __GAP__ ", text, flags=re.IGNORECASE)

    # [content] -> content
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)

    # Half brackets and variants
    text = text.replace("\u2039", "").replace("\u203A", "")  # ‹ ›
    text = text.replace("\u2308", "").replace("\u2309", "")  # ⌈ ⌉
    text = text.replace("\u230A", "").replace("\u230B", "")  # ⌊ ⌋
    text = text.replace("\u02F9", "").replace("\u02FA", "")  # ˹ ˺

    # Character maps
    text = text.translate(_FULL_MAP)
    text = text.translate(_SUBSCRIPT_MAP)

    # Scribal notations / word divider
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


def normalize_translation(text: Optional[str]) -> str:
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text
