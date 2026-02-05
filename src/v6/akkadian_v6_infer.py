#!/usr/bin/env python3
"""
Akkadian V6 Inference Script (Kaggle Offline)
==============================================
CRITICAL CHANGE: Tokenizer loaded FROM MODEL PATH with local_files_only=True
- Model: ByT5-small (fine-tuned V6)
- Tokenizer: Saved WITH model (NOT from HuggingFace)
- Environment: Kaggle, Internet OFF
- Features: Glossary + TM Retrieval (optional)
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class Config:
    kaggle_input: Path = Path("/kaggle/input")
    kaggle_working: Path = Path("/kaggle/working")

    # Model path on Kaggle
    model_name: str = "akkadian-v6-model"  # Kaggle dataset name
    model_path: Path = field(init=False)

    # Inference settings
    max_source_length: int = 256
    max_target_length: int = 256
    batch_size: int = 4
    num_beams: int = 5  # Slightly higher for better quality

    # Retrieval + glossary (optional)
    use_glossary: bool = False  # Simplified - can enable if needed
    tm_k: int = 5
    glossary_max_items: int = 8
    max_prompt_chars: int = 512

    def __post_init__(self):
        # Model path format: /kaggle/input/{dataset-name}/pytorch/default/1
        # Or simpler: /kaggle/input/{dataset-name}/final
        self.model_path = self.kaggle_input / self.model_name


CFG = Config()


# ==============================================================================
# Normalization (MUST match train exactly!)
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

# Merge all character maps
_ALL_CHAR_MAP = {**_VOWEL_MAP, **_CONSONANT_MAP, **_QUOTE_MAP, **_SUBSCRIPT_MAP}

# Build translation table safely
_TRANS_TABLE = {}
for k, v in _ALL_CHAR_MAP.items():
    if isinstance(k, str) and len(k) == 1:
        _TRANS_TABLE[ord(k)] = v


def normalize_transliteration(text) -> str:
    """Normalize Akkadian transliteration - MUST match train exactly!"""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)

    # Protect literal gap tokens
    text = text.replace("<gap>", "__LIT_GAP__")
    text = text.replace("<big_gap>", "__LIT_BIG_GAP__")

    # Remove apostrophe line numbers only (1', 1'')
    text = re.sub(r"\b\d+'{1,2}\b", " ", text)

    # Remove <content> blocks first
    text = re.sub(r"<<([^>]+)>>", r"\1", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)

    # Large gaps
    text = re.sub(r"\[\s*…+\s*…*\s*\]", " __BIG_GAP__ ", text)
    text = re.sub(r"\[\s*\.\.\.+\s*\.\.\.+\s*\]", " __BIG_GAP__ ", text)

    # Ellipsis
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


# ==============================================================================
# Glossary & Retrieval (optional)
# ==============================================================================

SRC_SPLIT_RE = re.compile(r"[\s\-]+")
TGT_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+")


def tokenize_src(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in SRC_SPLIT_RE.split(str(text)) if t]


def tokenize_tgt(text: str) -> list[str]:
    if not text:
        return []
    return TGT_TOKEN_RE.findall(str(text))


def char_ngrams(text: str, n: int = 3) -> list[str]:
    text = f" {text} "
    if len(text) < n:
        return [text]
    return [text[i : i + n] for i in range(len(text) - n + 1)]


class JaccardRetriever:
    """Simple retrieval fallback (no sklearn needed)."""

    def __init__(self, texts: list[str], n: int = 3, max_candidates: int = 500):
        self.texts = texts
        self.n = n
        self.max_candidates = max_candidates
        self.grams = [set(char_ngrams(t, n)) for t in texts]
        self.inverted = {}
        for idx, grams in enumerate(self.grams):
            for g in grams:
                self.inverted.setdefault(g, []).append(idx)

    def retrieve(self, query: str, k: int = 5) -> list[int]:
        grams = set(char_ngrams(query, self.n))
        freq = Counter()
        for g in grams:
            for idx in self.inverted.get(g, []):
                freq[idx] += 1

        if freq:
            candidates = [idx for idx, _ in freq.most_common(self.max_candidates)]
        else:
            candidates = list(range(min(len(self.texts), self.max_candidates)))

        scores = []
        for idx in candidates:
            inter = len(grams & self.grams[idx])
            union = len(grams) + len(self.grams[idx]) - inter
            score = inter / union if union else 0.0
            scores.append((score, idx))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [idx for _, idx in scores[:k]]


def build_prompt_with_retrieval(
    src: str,
    tm_pairs: list[dict],
    retriever: JaccardRetriever | None,
    glossary: dict[str, list[str]] | None,
    max_items: int,
    max_prompt_chars: int,
    tm_k: int,
) -> str:
    """Build prompt with glossary hints from retrieval + global glossary."""
    if not tm_pairs or retriever is None:
        # Fallback: just use global glossary
        if not glossary:
            return src
        items = []
        used = set()
        for tok in tokenize_src(src):
            if tok in used:
                continue
            tgts = glossary.get(tok)
            if tgts:
                items.append(f"{tok}={tgts[0]}")
                used.add(tok)
            if len(items) >= max_items:
                break
        if not items:
            return src
        return "GLOSSARY: " + "; ".join(items) + " ||| " + src

    # Retrieval-based
    idxs = retriever.retrieve(src, k=tm_k)
    neighbors = [tm_pairs[i] for i in idxs]

    query_tokens = tokenize_src(src)
    local_counts: dict[str, Counter] = {t: Counter() for t in query_tokens}

    for nb in neighbors:
        nb_src_tokens = set(tokenize_src(nb.get("src", "")))
        nb_tgt_tokens = tokenize_tgt(nb.get("tgt", ""))
        if not nb_src_tokens or not nb_tgt_tokens:
            continue
        for tok in query_tokens:
            if tok in nb_src_tokens:
                local_counts[tok].update(nb_tgt_tokens)

    items = []
    used = set()
    for tok in query_tokens:
        if tok in used:
            continue
        tgt = None
        if local_counts.get(tok):
            tgt = local_counts[tok].most_common(1)[0][0]
        elif glossary and tok in glossary:
            tgt = glossary[tok][0]
        if tgt:
            items.append(f"{tok}={tgt}")
            used.add(tok)
        if len(items) >= max_items:
            break

    if not items:
        return src

    prompt = "GLOSSARY: " + "; ".join(items) + " ||| " + src
    if len(prompt) > max_prompt_chars:
        return src
    return prompt


# ==============================================================================
# Helper Functions
# ==============================================================================

def is_kaggle() -> bool:
    return CFG.kaggle_input.exists()


def find_competition_data() -> Path:
    if not is_kaggle():
        local = Path("data")
        if local.exists():
            return local
        raise FileNotFoundError("Local data not found")

    for d in CFG.kaggle_input.iterdir():
        if "deep-past" in d.name.lower() or "akkadian" in d.name.lower():
            if (d / "test.csv").exists():
                return d
    raise FileNotFoundError("Competition data not found")


def find_assets_dir() -> Path | None:
    """Find directory with glossary and TM pairs."""
    if not is_kaggle():
        local = Path("data/v5d")
        if local.exists():
            return local
        return None

    for d in CFG.kaggle_input.iterdir():
        if (d / "v5d_glossary.json").exists() or (d / "v5d_tm_pairs.jsonl").exists():
            return d
    return None


def find_model_dir() -> Path:
    """Find V6 model directory (with tokenizer included)."""
    if not is_kaggle():
        local = Path("outputs/v6/final")
        if local.exists():
            return local
        raise FileNotFoundError("Local model not found")

    # Try common Kaggle paths
    candidates = [
        CFG.model_path / "final",  # /kaggle/input/akkadian-v6-model/final
        CFG.model_path / "pytorch" / "default" / "1",  # Standard Kaggle format
        CFG.model_path,  # Direct path
    ]

    for path in candidates:
        if path.exists() and (path / "config.json").exists():
            return path

    # Fallback: search
    for d in CFG.kaggle_input.iterdir():
        if "v6" in d.name.lower() or "akkadian-v6" in d.name.lower():
            if (d / "config.json").exists():
                return d
            for sub in d.glob("**/config.json"):
                return sub.parent

    raise FileNotFoundError(f"V6 model not found in {CFG.kaggle_input}")


def load_tm_pairs(path: Path) -> list[dict]:
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def load_glossary(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: list(v) for k, v in data.items()}


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 60)
    print("🚀 Akkadian V6 Inference (Kaggle Offline)")
    print("=" * 60)

    COMP_DIR = find_competition_data()
    MODEL_DIR = find_model_dir()
    ASSETS_DIR = find_assets_dir() if CFG.use_glossary else None

    print(f"📁 Competition data: {COMP_DIR}")
    print(f"🤖 Model: {MODEL_DIR}")
    if CFG.use_glossary:
        print(f"🧠 Assets: {ASSETS_DIR if ASSETS_DIR else 'not found'}")
    else:
        print(f"🧠 Glossary: Disabled")
    print(f"🎮 CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # ⭐ CRITICAL: Load tokenizer FROM MODEL PATH with local_files_only
    print(f"\n🔤 Loading tokenizer from: {MODEL_DIR}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True  # ⭐ ESSENTIAL for Kaggle offline
        )
        print(f"   ✅ Tokenizer loaded (offline mode)")
        print(f"   Tokenizer vocab: {len(tokenizer)}")
    except Exception as e:
        print(f"   ❌ ERROR loading tokenizer: {e}")
        print(f"   Make sure tokenizer files exist in {MODEL_DIR}:")
        print(f"   - tokenizer.json")
        print(f"   - tokenizer_config.json")
        print(f"   - special_tokens_map.json")
        raise

    # ⭐ CRITICAL: Load model with local_files_only
    print(f"\n🤖 Loading model from: {MODEL_DIR}")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True  # ⭐ ESSENTIAL for Kaggle offline
        )
        print(f"   ✅ Model loaded (offline mode)")
        print(f"   Model vocab: {model.config.vocab_size}")
        print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ❌ ERROR loading model: {e}")
        raise

    # Vocab check
    if len(tokenizer) != model.config.vocab_size:
        print(f"   ⚠️ WARNING: Vocab mismatch! Tokenizer={len(tokenizer)}, Model={model.config.vocab_size}")
    else:
        print("   ✅ Vocab match")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"   ✅ Model on {device}")

    # Load TM + Glossary (optional)
    TM_PAIRS = []
    GLOSSARY = None
    RETRIEVER = None

    if CFG.use_glossary and ASSETS_DIR:
        tm_path = ASSETS_DIR / "v5d_tm_pairs.jsonl"
        glossary_path = ASSETS_DIR / "v5d_glossary.json"

        if tm_path.exists():
            TM_PAIRS = load_tm_pairs(tm_path)
            print(f"\n🧠 TM pairs: {len(TM_PAIRS):,}")
        else:
            print("\n🧠 TM pairs: not found")

        if glossary_path.exists():
            GLOSSARY = load_glossary(glossary_path)
            print(f"🧠 Glossary size: {len(GLOSSARY):,}")
        else:
            print("🧠 Glossary: not found")

        if TM_PAIRS:
            RETRIEVER = JaccardRetriever([p.get("src", "") for p in TM_PAIRS])

    # Sanity check
    print("\n🔍 Sanity check...")
    test_input = "um-ma ka-ru-um"
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50, num_beams=4)
    test_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Input: '{test_input}'")
    print(f"   Output: '{test_output}'")

    if not test_output or test_output.strip() == "":
        print("   ⚠️ WARNING: Empty output! Using fallback mode.")
        FALLBACK_MODE = True
    else:
        print("   ✅ Model produces non-empty output")
        FALLBACK_MODE = False

    # Inference function
    @torch.no_grad()
    def generate_batch(texts: list[str], debug: bool = False) -> list[str]:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CFG.max_source_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if debug:
            print(f"   [DEBUG] Input shape: {inputs['input_ids'].shape}")

        outputs = model.generate(
            **inputs,
            max_length=CFG.max_target_length,
            num_beams=CFG.num_beams,
            early_stopping=True,
        )

        if debug:
            print(f"   [DEBUG] Output shape: {outputs.shape}")

        results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Fallback for empty outputs
        final_results = []
        for r in results:
            if not r or r.strip() == "":
                final_results.append("[translation unavailable]")
            else:
                final_results.append(r)

        return final_results

    def translate_all(texts: list[str]) -> list[str]:
        translations = []
        pbar = tqdm(range(0, len(texts), CFG.batch_size), desc="🔮 Translating", unit="batch")
        for i in pbar:
            batch = texts[i : i + CFG.batch_size]
            translations.extend(generate_batch(batch))
        return translations

    # Load test data
    print("\n📖 Loading test data...")
    test_df = pd.read_csv(COMP_DIR / "test.csv")
    print(f"   Test samples: {len(test_df):,}")

    # Normalize
    print("\n🔧 Normalizing...")
    normalized = [normalize_transliteration(t) for t in tqdm(test_df["transliteration"], desc="Normalizing")]

    # Build prompts (with or without glossary)
    if CFG.use_glossary and (TM_PAIRS or GLOSSARY):
        print("\n🧠 Building glossary prompts...")
        prompts = []
        for src in tqdm(normalized, desc="Glossary"):
            prompt = build_prompt_with_retrieval(
                src,
                tm_pairs=TM_PAIRS,
                retriever=RETRIEVER,
                glossary=GLOSSARY,
                max_items=CFG.glossary_max_items,
                max_prompt_chars=CFG.max_prompt_chars,
                tm_k=CFG.tm_k,
            )
            prompts.append(prompt)
    else:
        print("\n📝 Using plain normalized inputs (no glossary)...")
        prompts = normalized

    print(f"\n📝 Sample prompts:")
    for i in range(min(2, len(prompts))):
        print(f"   [{i}] {prompts[i][:120]}...")

    # Run inference
    print("\n🚀 Running inference...")
    print("\n[DEBUG] First sample test...")
    _test = generate_batch([prompts[0]], debug=True)
    print(f"[DEBUG] Translation: '{_test[0][:100]}...'")

    translations = translate_all(prompts)

    empty_count = sum(1 for t in translations if t == "[translation unavailable]")
    if empty_count > 0:
        print(f"\n⚠️ WARNING: {empty_count} fallback translations!")

    print(f"\n📝 Sample outputs:")
    for i in range(min(3, len(translations))):
        print(f"   [{i}] {translations[i][:150]}...")

    # Create submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "translation": translations,
    })

    assert len(submission) == len(test_df), "Length mismatch!"
    assert submission["translation"].notna().all(), "NaN values!"

    output_path = CFG.kaggle_working / "submission.csv" if is_kaggle() else Path("submission.csv")
    submission.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("✅ V6 Inference Complete!")
    print("=" * 60)
    print(f"📁 Saved: {output_path}")
    print(f"   Rows: {len(submission)}")
    print()
    print(submission.head())
    print("=" * 60)


if __name__ == "__main__":
    main()
