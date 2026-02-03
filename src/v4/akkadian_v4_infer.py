#%% [markdown]
# # Akkadian V4 Inference
#
# **Key Features:**
# - V4 model (Full FT + OCR-augmented training)
# - V2-identical normalization (NO augmentation at inference)
# - ByT5Tokenizer for consistency
# - NO repetition_penalty (avoid BLEU degradation)
#
# **Environment**: Kaggle T4/P100 GPU
#
# **Usage:**
# ```bash
# uv run jupytext --to notebook src/v4/akkadian_v4_infer.py
# ```

#%% [markdown]
# ## 1. Imports & Configuration

#%%
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
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
    
    # Model selection: "base" or "large"
    model_size: str = "base"  # Change to "large" for v4-large
    
    # Model path (set in __post_init__)
    model_path: Path = field(init=False)
    
    # Inference params (same as V2)
    max_source_length: int = 512
    max_target_length: int = 512
    batch_size: int = 4
    num_beams: int = 4
    
    def __post_init__(self):
        if self.model_size == "base":
            self.model_path = self.kaggle_input / "akkadian-v4-base/pytorch/default/1"
        else:
            self.model_path = self.kaggle_input / "akkadian-v4-large/pytorch/default/1"


# ============================================
# ⚠️ CHANGE THIS FOR v4-large
# ============================================
CFG = Config(model_size="base")  # "base" or "large"

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


def find_model():
    """Find model directory."""
    if not is_kaggle():
        local = Path(f"outputs/akkadian_v4_{CFG.model_size}/model")
        if local.exists():
            return local
        raise FileNotFoundError("Local model not found")
    
    # Check configured path
    if CFG.model_path.exists():
        return CFG.model_path
    
    # Search for v4 model
    for d in CFG.kaggle_input.iterdir():
        if "v4" in d.name.lower() and CFG.model_size in d.name.lower():
            if (d / "config.json").exists():
                return d
            for sub in d.glob("**/config.json"):
                return sub.parent
    
    raise FileNotFoundError(f"V4-{CFG.model_size} model not found")

#%% [markdown]
# ## 3. Normalization (V2-identical)

#%%
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

_CONSONANT_MAP = {
    '\u0161': 's', '\u0160': 'S',
    '\u1e63': 's', '\u1e62': 'S',
    '\u1e6d': 't', '\u1e6c': 'T',
    '\u1e2b': 'h', '\u1e2a': 'H',
}

_OCR_MAP = {
    '\u201e': '"', '\u201c': '"', '\u201d': '"',
    '\u2018': "'", '\u2019': "'", '\u201a': "'",
    '\u02be': "'", '\u02bf': "'",
    '\u2308': '[', '\u2309': ']', '\u230a': '[', '\u230b': ']',
}

_SUBSCRIPT_MAP = str.maketrans({
    '\u2080': '0', '\u2081': '1', '\u2082': '2', '\u2083': '3', '\u2084': '4',
    '\u2085': '5', '\u2086': '6', '\u2087': '7', '\u2088': '8', '\u2089': '9',
    '\u2093': 'x',
})

_FULL_MAP = str.maketrans({**_VOWEL_MAP, **_CONSONANT_MAP, **_OCR_MAP})


def normalize_transliteration(text) -> str:
    """Normalize Akkadian transliteration to ASCII (V2-identical)."""
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
# ## 4. Setup

#%%
print("=" * 60)
print(f"🚀 Akkadian V4 Inference: {CFG.model_size.upper()}")
print("=" * 60)

COMP_DIR = find_competition_data()
MODEL_DIR = find_model()

print(f"📁 Competition data: {COMP_DIR}")
print(f"🤖 Model: {MODEL_DIR}")
print(f"🎮 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

#%% [markdown]
# ## 5. Load Model

#%%
print(f"\n🤖 Loading model from: {MODEL_DIR}")
print("   This may take a few minutes...")

# Use ByT5Tokenizer for consistency (same as training)
# ByT5 vocab: 256 (bytes) + 3 (special) + 125 (extra_ids) = 384
tokenizer = ByT5Tokenizer(extra_ids=125)
print(f"   Tokenizer vocab size: {len(tokenizer)}")

model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))
print(f"   Model vocab size: {model.config.vocab_size}")
print(f"   Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# Verify vocab match
assert len(tokenizer) == model.config.vocab_size, \
    f"Vocab mismatch! Tokenizer: {len(tokenizer)}, Model: {model.config.vocab_size}"
print("   ✅ Vocab sizes match")

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
    
    # Repetition penalty to prevent "the merchant ... the merchant ..." loops
    outputs = model.generate(
        **inputs,
        max_length=CFG.max_target_length,
        num_beams=CFG.num_beams,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
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

print("\n🔧 Normalizing (V2-identical)...")
normalized = [normalize_transliteration(t) for t in tqdm(test_df["transliteration"], desc="Normalizing")]

print(f"\n📝 Sample normalized:")
for i in range(min(2, len(normalized))):
    print(f"   [{i}] {normalized[i][:100]}...")

#%%
print("\n🚀 Running inference...")

print("\n[DEBUG] Testing first sample...")
test_result = generate_batch([normalized[0]], debug=True)
print(f"[DEBUG] First translation: '{test_result[0][:100]}...'")

translations = translate_all(normalized)

# Validate outputs
empty_count = sum(1 for t in translations if not t or not t.strip())
if empty_count > 0:
    print(f"\n⚠️ WARNING: {empty_count} empty translations!")

print(f"\n📝 Sample outputs:")
for i in range(min(3, len(translations))):
    print(f"   [{i}] {translations[i][:150]}...")

#%% [markdown]
# ## 8. Create Submission

#%%
# Check for empty translations
empty_count = sum(1 for t in translations if not t or not t.strip())
if empty_count > 0:
    print(f"\n⚠️ WARNING: {empty_count}/{len(translations)} empty translations!")
    print("   This may indicate model issues. Please verify.")
    # Keep empty as is for now - replacing hurts BLEU more

submission = pd.DataFrame({
    "id": test_df["id"],
    "translation": translations,
})

# Validate
assert len(submission) == len(test_df), "Submission length mismatch!"
assert submission["translation"].notna().all(), "Submission has NaN values!"

output_path = CFG.kaggle_working / "submission.csv" if is_kaggle() else Path("submission.csv")
submission.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print(f"✅ V4-{CFG.model_size.upper()} Inference Complete!")
print("=" * 60)
print(f"📁 Saved: {output_path}")
print(f"   Rows: {len(submission)}")
print()
print(submission.head())
print("=" * 60)
