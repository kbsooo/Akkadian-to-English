#%% [markdown]
# # Akkadian V3 Inference: Merged Model
#
# **Key Features:**
# - Pre-merged model (LoRA already integrated into base model)
# - NO PEFT required → Works on Kaggle Internet OFF
# - **V2-identical normalization** for consistent results
#
# **Environment**: Kaggle T4/P100 GPU (Internet OFF supported)
#
# **Usage:**
# ```bash
# uv run jupytext --to notebook src/v3/akkadian_v3_infer.py
# ```

#%% [markdown]
# ## 1. Imports & Configuration

#%%
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, ByT5Tokenizer

#%%
@dataclass
class Config:
    """Inference configuration."""
    kaggle_input: Path = Path("/kaggle/input")
    kaggle_working: Path = Path("/kaggle/working")
    
    # Model path (merged model - no PEFT needed)
    model_path: str = "/kaggle/input/akkadian-v3/pytorch/default/4"
    
    # Inference params (MUST match V2 for fair comparison)
    max_source_length: int = 512   # Same as V2
    max_target_length: int = 512   # Same as V2
    batch_size: int = 4
    num_beams: int = 4
    fp16: bool = False  # ByT5 is unstable with FP16


CFG = Config()

#%% [markdown]
# ## 2. Helper Functions

#%%
def is_kaggle():
    return CFG.kaggle_input.exists()


def find_competition_data():
    """Find competition data directory."""
    if not is_kaggle():
        local = Path("data")
        if local.exists():
            return local
        raise FileNotFoundError("Local data directory not found")
    
    for d in CFG.kaggle_input.iterdir():
        if "deep-past" in d.name.lower() or "akkadian" in d.name.lower():
            if (d / "test.csv").exists():
                return d
    raise FileNotFoundError("Competition data not found")

#%% [markdown]
# ## 3. Normalization (IDENTICAL to V2 normalize.py)

#%%
# ==============================================================================
# Character Mapping Tables (copied from V2 normalize.py)
# ==============================================================================

# Vowels with diacritics → base vowels
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

# OCR artifacts and typography
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


def normalize_transliteration(text) -> str:
    """
    Normalize Akkadian transliteration to ASCII-compatible format.
    
    IDENTICAL to V2 src/v2/normalize.py:normalize_transliteration
    
    Transformations:
    1. Unicode NFC normalization
    2. Diacritics removal (à → a, š → s, etc.)
    3. OCR artifact cleanup (curly quotes → straight quotes)
    4. Subscript normalization (₄ → 4)
    5. Gap/damage markers ([...] → <gap>)
    6. Unknown sign markers (x → <unk>)
    7. Editorial mark removal (!?/)
    8. Whitespace normalization
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

#%% [markdown]
# ## 4. Setup

#%%
print("=" * 60)
print("🚀 Akkadian V3 Inference: Merged Model (PEFT-free)")
print("   Normalization: V2-identical")
print("=" * 60)

COMP_DIR = find_competition_data()
MODEL_DIR = Path(CFG.model_path)

print(f"📁 Competition data: {COMP_DIR}")
print(f"🤖 Merged model: {MODEL_DIR}")
print(f"🎮 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

#%% [markdown]
# ## 5. Load Model

#%%
print(f"\n🤖 Loading merged model from: {MODEL_DIR}")
print("   This may take a few minutes...")

# Use ByT5Tokenizer with extra_ids to match V2 exactly
# ByT5 vocab: 256 (bytes) + 3 (special) + 125 (extra_ids) = 384
tokenizer = ByT5Tokenizer(extra_ids=125)
print(f"   Tokenizer vocab size: {len(tokenizer)}")

model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))
print(f"   Model vocab size: {model.config.vocab_size}")
print(f"   Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# Verify vocab sizes match
if len(tokenizer) != model.config.vocab_size:
    print(f"   ⚠️ Vocab mismatch! Tokenizer: {len(tokenizer)}, Model: {model.config.vocab_size}")
    # Do NOT resize - use model's vocab size
else:
    print(f"   ✅ Vocab sizes match: {len(tokenizer)}")

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"   ✅ Model on {device}")

#%% [markdown]
# ## 6. Inference Functions

#%%
@torch.no_grad()
def generate_batch(texts, debug=False):
    """Generate translations for a batch of texts."""
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
        repetition_penalty=1.2,   # Prevent repetition
        no_repeat_ngram_size=3,   # No 3-gram repeats
        early_stopping=True,
    )
    
    if debug:
        print(f"   [DEBUG] Output shape: {outputs.shape}")
        print(f"   [DEBUG] Output tokens (first 20): {outputs[0][:20].tolist()}")
    
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    if debug:
        print(f"   [DEBUG] Decoded (first): '{decoded[0][:100]}'")
    
    return decoded


def translate_all(texts, batch_size=None):
    """Translate all texts with progress bar."""
    if batch_size is None:
        batch_size = CFG.batch_size
    
    translations = []
    pbar = tqdm(range(0, len(texts), batch_size), desc="🔮 Translating", unit="batch", ncols=80)
    
    for i in pbar:
        batch = texts[i:i + batch_size]
        results = generate_batch(batch)
        translations.extend(results)
        pbar.set_postfix(done=f"{min(i + batch_size, len(texts))}/{len(texts)}")
    
    return translations

#%% [markdown]
# ## 7. Run Inference

#%%
print("\n📖 Loading test data...")
test_df = pd.read_csv(COMP_DIR / "test.csv")
print(f"   Test samples: {len(test_df):,}")

# Normalize using V2-identical function
print("\n🔧 Normalizing (V2-identical)...")
normalized = [normalize_transliteration(t) for t in tqdm(test_df["transliteration"], desc="Normalizing")]

print(f"\n📝 Sample normalized:")
for i in range(min(2, len(normalized))):
    print(f"   [{i}] {normalized[i][:100]}...")

#%%
print("\n🚀 Running inference...")

# Debug first sample
print("\n[DEBUG] Testing first sample...")
test_result = generate_batch([normalized[0]], debug=True)
print(f"[DEBUG] First translation: '{test_result[0][:100]}...'")

translations = translate_all(normalized)

print(f"\n📝 Sample outputs:")
for i in range(min(3, len(translations))):
    print(f"   [{i}] {translations[i][:150]}...")

#%% [markdown]
# ## 8. Create Submission

#%%
submission = pd.DataFrame({
    "id": test_df["id"],
    "translation": translations,
})

# Save
output_path = CFG.kaggle_working / "submission.csv" if is_kaggle() else Path("submission.csv")
submission.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("✅ Inference Complete!")
print("=" * 60)
print(f"📁 Saved: {output_path}")
print(f"   Rows: {len(submission)}")
print()
print(submission.head())
print("=" * 60)
