#%% [markdown]
# # Akkadian V3 Inference: Merged Model
#
# **Key Features:**
# - Pre-merged model (LoRA already integrated into base model)
# - NO PEFT required → Works on Kaggle Internet OFF
# - Same normalization as training
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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

#%%
@dataclass
class Config:
    """Inference configuration."""
    kaggle_input: Path = Path("/kaggle/input")
    kaggle_working: Path = Path("/kaggle/working")
    
    # Model path (merged model - no PEFT needed)
    model_path: str = "/kaggle/input/akkadian-v3/pytorch/default/4"
    
    # Inference params
    max_source_length: int = 256
    max_target_length: int = 256
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
# ## 3. Normalization (must match training)

#%%
# ASCII transliteration mapping
_NORMALIZE_MAP = {
    # Shin/Sibilants
    'š': 's', 'Š': 'S',
    'ṣ': 's', 'Ṣ': 'S',
    'ś': 's', 'Ś': 'S',
    
    # Emphatics
    'ṭ': 't', 'Ṭ': 'T',
    'ḫ': 'h', 'Ḫ': 'H',
    'ḥ': 'h', 'Ḥ': 'H',
    
    # Ayin, Aleph
    'ʾ': "'", 'ʿ': "'", ''': "'", ''': "'",
    'ˀ': "'", 'ˁ': "'",
    'ʔ': "'", 'ʕ': "'",
    
    # Nasals
    'ṃ': 'm', 'Ṃ': 'M',
    'ṅ': 'n', 'Ṅ': 'N',
    'ñ': 'n', 'Ñ': 'N',
    
    # Long vowels (macron)
    'ā': 'a', 'Ā': 'A',
    'ē': 'e', 'Ē': 'E',
    'ī': 'i', 'Ī': 'I',
    'ō': 'o', 'Ō': 'O',
    'ū': 'u', 'Ū': 'U',
    
    # Breve
    'ă': 'a', 'ĕ': 'e', 'ĭ': 'i', 'ŏ': 'o', 'ŭ': 'u',
    
    # Subscript numbers → normal
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
    
    # Superscript
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    
    # Special
    '×': 'x', '·': '.', '°': '',
}


def normalize_transliteration(text: str) -> str:
    """Convert Akkadian transliteration to ASCII (must match training)."""
    if not isinstance(text, str):
        return ""
    
    # NFD decomposition
    text = unicodedata.normalize('NFD', text)
    
    # Apply mapping
    result = []
    for char in text:
        if char in _NORMALIZE_MAP:
            result.append(_NORMALIZE_MAP[char])
        elif unicodedata.category(char) == 'Mn':  # Skip combining marks
            continue
        else:
            result.append(char)
    
    text = ''.join(result)
    
    # Whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

#%% [markdown]
# ## 4. Setup

#%%
print("=" * 60)
print("🚀 Akkadian V3 Inference: Merged Model (PEFT-free)")
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

# Load tokenizer and model directly (no PEFT needed!)
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
print(f"   Tokenizer vocab size: {len(tokenizer)}")

model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))
print(f"   Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

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

# Normalize
print("\n🔧 Normalizing (ASCII conversion)...")
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
