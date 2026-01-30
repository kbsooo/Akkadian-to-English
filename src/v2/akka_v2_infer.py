#%% [markdown]
# # Akkadian V2 Inference
#
# **Key Changes from V1:**
# - Unified ASCII normalization (same as training)
# - All diacritics converted to ASCII (š→s, à→a, etc.)
# - Test data now uses same character set as training
#
# **Environment**: Kaggle T4 GPU x2
#
# **Usage:**
# ```bash
# uv run jupytext --to notebook src/v2/akka_v2_infer.py
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
    
    # Inference params
    max_source_length: int = 512
    max_target_length: int = 512
    batch_size: int = 4
    num_beams: int = 4
    fp16: bool = True


CFG = Config()

#%% [markdown]
# ## 2. Normalization (MUST match training)

#%%
# Vowels with diacritics → base vowels
_VOWEL_MAP = {
    '\u00e0': 'a', '\u00e1': 'a', '\u00e2': 'a', '\u0101': 'a', '\u00e4': 'a',
    '\u00c0': 'A', '\u00c1': 'A', '\u00c2': 'A', '\u0100': 'A', '\u00c4': 'A',
    '\u00e8': 'e', '\u00e9': 'e', '\u00ea': 'e', '\u0113': 'e', '\u00eb': 'e',
    '\u00c8': 'E', '\u00c9': 'E', '\u00ca': 'E', '\u0112': 'E', '\u00cb': 'E',
    '\u00ec': 'i', '\u00ed': 'i', '\u00ee': 'i', '\u012b': 'i', '\u00ef': 'i',
    '\u00cc': 'I', '\u00cd': 'I', '\u00ce': 'I', '\u012a': 'I', '\u00cf': 'I',
    '\u00f2': 'o', '\u00f3': 'o', '\u00f4': 'o', '\u014d': 'o', '\u00f6': 'o',
    '\u00d2': 'O', '\u00d3': 'O', '\u00d4': 'O', '\u014c': 'O', '\u00d6': 'O',
    '\u00f9': 'u', '\u00fa': 'u', '\u00fb': 'u', '\u016b': 'u', '\u00fc': 'u',
    '\u00d9': 'U', '\u00da': 'U', '\u00db': 'U', '\u016a': 'U', '\u00dc': 'U',
}

# Akkadian consonants → ASCII
_CONSONANT_MAP = {
    '\u0161': 's', '\u0160': 'S',  # š, Š
    '\u1e63': 's', '\u1e62': 'S',  # ṣ, Ṣ
    '\u1e6d': 't', '\u1e6c': 'T',  # ṭ, Ṭ
    '\u1e2b': 'h', '\u1e2a': 'H',  # ḫ, Ḫ
}

# OCR artifacts
_OCR_MAP = {
    '\u201e': '"', '\u201c': '"', '\u201d': '"',
    '\u2018': "'", '\u2019': "'", '\u201a': "'",
    '\u02be': "'", '\u02bf': "'",
    '\u2308': '[', '\u2309': ']', '\u230a': '[', '\u230b': ']',
}

# Subscripts
_SUBSCRIPT_MAP = str.maketrans({
    '\u2080': '0', '\u2081': '1', '\u2082': '2', '\u2083': '3', '\u2084': '4',
    '\u2085': '5', '\u2086': '6', '\u2087': '7', '\u2088': '8', '\u2089': '9',
    '\u2093': 'x',
})

_FULL_MAP = str.maketrans({**_VOWEL_MAP, **_CONSONANT_MAP, **_OCR_MAP})


def normalize_transliteration(text):
    """Normalize to ASCII - MUST match training preprocessing."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_FULL_MAP)
    text = text.translate(_SUBSCRIPT_MAP)
    text = text.replace('\u2026', ' <gap> ')
    text = re.sub(r'\.\.\.+', ' <gap> ', text)
    text = re.sub(r'\[([^\]]*)\]', ' <gap> ', text)
    text = re.sub(r'\bx\b', ' <unk> ', text, flags=re.IGNORECASE)
    text = re.sub(r'[!?/]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#%% [markdown]
# ## 3. Environment Detection

#%%
def is_kaggle():
    return Path("/kaggle/input").exists()


def find_competition_data():
    if not is_kaggle():
        return Path("data")
    for d in CFG.kaggle_input.iterdir():
        if (d / "test.csv").exists():
            return d
    raise FileNotFoundError("Competition data not found")


def find_model_dir():
    if not is_kaggle():
        local = Path("outputs/akkadian_v2/final")
        if local.exists():
            return local
        raise FileNotFoundError("Local model not found")
    
    # Kaggle: find model dataset
    for d in CFG.kaggle_input.iterdir():
        if (d / "config.json").exists():
            return d
        for sub in d.glob("**/config.json"):
            return sub.parent
    raise FileNotFoundError("Model not found in /kaggle/input")


COMP_DIR = find_competition_data()
MODEL_DIR = find_model_dir()

print(f"📁 Competition data: {COMP_DIR}")
print(f"🤖 Model: {MODEL_DIR}")
print(f"🎮 CUDA: {torch.cuda.is_available()}")

#%% [markdown]
# ## 4. Load Model

#%%
print(f"🤖 Loading model from {MODEL_DIR}")

# ByT5 tokenizer with extra_ids to match training
tokenizer = ByT5Tokenizer(extra_ids=125)
print(f"   Tokenizer len: {len(tokenizer)}")  # Should be 384

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
print(f"   Model vocab: {model.config.vocab_size}")

# Ensure vocab sizes match
if len(tokenizer) != model.config.vocab_size:
    model.resize_token_embeddings(len(tokenizer))
    print(f"   Resized to: {model.config.vocab_size}")

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if CFG.fp16 and device.type == "cuda":
    model = model.half()
    print("   ✅ Using FP16")

model.eval()
print(f"   ✅ Model on {device}")

#%% [markdown]
# ## 5. Inference Functions

#%%
@torch.no_grad()
def generate_batch(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=CFG.max_source_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_length=CFG.max_target_length,
        num_beams=CFG.num_beams,
        early_stopping=True,
    )
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def translate_all(texts, batch_size=None):
    if batch_size is None:
        batch_size = CFG.batch_size
    
    translations = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i:i + batch_size]
        translations.extend(generate_batch(batch))
    
    return translations

#%% [markdown]
# ## 6. Load Test Data & Run Inference

#%%
print("📖 Loading test data...")
test_df = pd.read_csv(COMP_DIR / "test.csv")
print(f"   Test samples: {len(test_df)}")

# Normalize (CRITICAL: same as training)
print("🔧 Normalizing (ASCII conversion)...")
normalized = [normalize_transliteration(t) for t in test_df["transliteration"]]

print("\n📝 Sample normalized:")
for i in range(min(2, len(normalized))):
    print(f"   [{i}] {normalized[i][:100]}...")

#%%
print("\n🚀 Running inference...")
translations = translate_all(normalized)

print("\n📝 Sample outputs:")
for i in range(min(2, len(translations))):
    print(f"   [{i}] {translations[i][:150]}...")

#%% [markdown]
# ## 7. Create Submission

#%%
submission = pd.DataFrame({
    "id": test_df["id"],
    "translation": translations,
})

assert len(submission) == len(test_df)
assert submission["translation"].notna().all()

output_path = CFG.kaggle_working / "submission.csv" if is_kaggle() else Path("submission.csv")
submission.to_csv(output_path, index=False)

print(f"\n✅ Saved: {output_path}")
print(submission.head())

#%% [markdown]
# ## Done!
#
# Submit `submission.csv` to the competition.
