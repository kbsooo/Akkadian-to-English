#%% [markdown]
# # Akkadian V3 Inference: ByT5-Large + LoRA
#
# **Key Features:**
# - Load LoRA adapter weights and merge with base model
# - Same normalization as training for consistent results
# - tqdm progress bar for translation progress
#
# **Environment**: Kaggle T4/P100 GPU
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
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, ByT5Tokenizer

#%%
@dataclass
class Config:
    """Inference configuration."""
    kaggle_input: Path = Path("/kaggle/input")
    kaggle_working: Path = Path("/kaggle/working")
    
    # Model
    base_model_name: str = "google/byt5-large"
    
    # Inference params
    max_source_length: int = 256
    max_target_length: int = 256
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
    """Find competition test data."""
    if not is_kaggle():
        return Path("data")
    for d in CFG.kaggle_input.iterdir():
        if (d / "test.csv").exists():
            return d
    raise FileNotFoundError("Competition data not found")


def find_lora_adapter():
    """Find LoRA adapter in Kaggle input."""
    if not is_kaggle():
        local = Path("outputs/akkadian_v3/lora_adapter")
        if local.exists():
            return local
        raise FileNotFoundError("Local LoRA adapter not found")
    
    # Kaggle: find adapter (look for adapter_config.json)
    for d in CFG.kaggle_input.iterdir():
        if (d / "adapter_config.json").exists():
            return d
        for sub in d.glob("**/adapter_config.json"):
            return sub.parent
    raise FileNotFoundError("LoRA adapter not found in /kaggle/input")


def find_base_model():
    """Find base model (byt5-large) in Kaggle input for offline use."""
    if not is_kaggle():
        # Local: use HuggingFace model name
        return "google/byt5-large"
    
    # Kaggle: search for local byt5-large model
    # Common paths from Kaggle Models
    possible_paths = [
        CFG.kaggle_input / "byt5-large" / "pytorch" / "default" / "1",
        CFG.kaggle_input / "byt5-large",
        CFG.kaggle_input / "google-byt5-large",
    ]
    
    for p in possible_paths:
        if p.exists() and (p / "config.json").exists():
            return str(p)
    
    # Search all input directories for config.json (byt5-large marker)
    for d in CFG.kaggle_input.iterdir():
        if "byt5" in d.name.lower():
            if (d / "config.json").exists():
                return str(d)
            for sub in d.glob("**/config.json"):
                return str(sub.parent)
    
    # Fallback to online (will fail if internet is off)
    print("⚠️ Local byt5-large not found, trying online...")
    return "google/byt5-large"


print("=" * 60)
print("🚀 Akkadian V3 Inference: ByT5-Large + LoRA")
print("=" * 60)

COMP_DIR = find_competition_data()
ADAPTER_DIR = find_lora_adapter()
BASE_MODEL_PATH = find_base_model()

print(f"📁 Competition data: {COMP_DIR}")
print(f"🔧 LoRA adapter: {ADAPTER_DIR}")
print(f"🤖 Base model: {BASE_MODEL_PATH}")
print(f"🎮 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

#%% [markdown]
# ## 4. Load Model with LoRA

#%%
print(f"\n🤖 Loading base model: {BASE_MODEL_PATH}")
print("   This may take a few minutes...")

# Load tokenizer (ByT5 uses byte-level tokenization, no vocab file needed)
tokenizer = ByT5Tokenizer()
print(f"   Tokenizer vocab size: {len(tokenizer)}")

# Load base model (offline-compatible)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_PATH)
print(f"   Base model loaded")

#%%
# Load LoRA adapter
print(f"\n🔧 Loading LoRA adapter from: {ADAPTER_DIR}")
model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
print("   LoRA adapter loaded")

# Merge and unload for faster inference
print("   Merging adapter weights...")
model = model.merge_and_unload()
print("   ✅ Merged successfully")

#%%
# Ensure vocab sizes match
if len(tokenizer) != model.config.vocab_size:
    model.resize_token_embeddings(len(tokenizer))
    print(f"   Resized embeddings to: {model.config.vocab_size}")

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
    """Generate translations for a batch of texts."""
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
    """Translate all texts with progress bar."""
    if batch_size is None:
        batch_size = CFG.batch_size
    
    translations = []
    
    # tqdm progress bar
    pbar = tqdm(
        range(0, len(texts), batch_size),
        desc="🔮 Translating",
        unit="batch",
        ncols=80
    )
    
    for i in pbar:
        batch = texts[i:i + batch_size]
        translations.extend(generate_batch(batch))
        pbar.set_postfix({"done": f"{len(translations)}/{len(texts)}"})
    
    return translations

#%% [markdown]
# ## 6. Load Test Data & Run Inference

#%%
print("\n📖 Loading test data...")
test_df = pd.read_csv(COMP_DIR / "test.csv")
print(f"   Test samples: {len(test_df):,}")

# Normalize (CRITICAL: same as training)
print("\n🔧 Normalizing (ASCII conversion)...")
normalized = [normalize_transliteration(t) for t in tqdm(test_df["transliteration"], desc="Normalizing")]

print(f"\n📝 Sample normalized:")
for i in range(min(2, len(normalized))):
    print(f"   [{i}] {normalized[i][:100]}...")

#%%
print("\n🚀 Running inference...")
translations = translate_all(normalized)

print(f"\n📝 Sample outputs:")
for i in range(min(3, len(translations))):
    print(f"   [{i}] {translations[i][:150]}...")

#%% [markdown]
# ## 7. Create Submission

#%%
submission = pd.DataFrame({
    "id": test_df["id"],
    "translation": translations,
})

# Validation
assert len(submission) == len(test_df), "Row count mismatch!"
assert submission["translation"].notna().all(), "Found NaN translations!"

# Save
output_path = CFG.kaggle_working / "submission.csv" if is_kaggle() else Path("submission.csv")
submission.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("✅ Inference Complete!")
print("=" * 60)
print(f"📁 Saved: {output_path}")
print(f"   Rows: {len(submission):,}")
print()
print(submission.head())
print("=" * 60)

#%% [markdown]
# ## Done!
#
# Submit `submission.csv` to the competition.
