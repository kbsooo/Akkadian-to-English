#%% [markdown]
# # Akkadian → English Translation: Inference (V1)
#
# **Environment**: Kaggle T4 GPU x2
#
# **Model**: ByT5-base (loaded from Kaggle Models)
#
# **Workflow**:
# 1. Load trained model from Kaggle Models/Dataset
# 2. Load test data from competition
# 3. Run inference with batching
# 4. Create submission.csv
#
# **Usage (convert to notebook)**:
# ```bash
# uv run jupytext --to notebook src/akka_v1_infer.py
# ```

#%% [markdown]
# ## 1. Imports & Configuration

#%%
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

#%%
# =============================
# Configuration
# =============================

@dataclass
class Config:
    """Inference configuration for Kaggle T4 x2 environment."""
    # Paths (Kaggle)
    kaggle_input: Path = Path("/kaggle/input")
    kaggle_working: Path = Path("/kaggle/working")
    
    # Model settings
    # Option 1: Set model dataset name (uploaded as Kaggle Dataset)
    model_dataset_name: Optional[str] = None  # e.g., "akkadian-byt5-v1"
    
    # Option 2: Set model path directly
    model_path: Optional[Path] = None
    
    # Inference
    max_source_length: int = 256
    max_target_length: int = 256
    batch_size: int = 8  # larger batch for inference
    num_beams: int = 4
    
    # Hardware
    fp16: bool = True  # Use FP16 for faster inference


CFG = Config()

#%% [markdown]
# ## 2. Environment Detection

#%%
def is_kaggle() -> bool:
    """Check if running on Kaggle."""
    return Path("/kaggle/input").exists()


def find_competition_data() -> Path:
    """Find competition data directory."""
    if not is_kaggle():
        # Local fallback
        local_path = Path("data")
        if local_path.exists():
            return local_path
        raise FileNotFoundError("Cannot find competition data locally")
    
    # On Kaggle: look for test.csv and sample_submission.csv
    for d in CFG.kaggle_input.iterdir():
        if (d / "test.csv").exists() and (d / "sample_submission.csv").exists():
            return d
    raise FileNotFoundError("Cannot find competition data in /kaggle/input")


def find_model_dir() -> Path:
    """Find trained model directory."""
    # Option 1: Explicit path (handle both str and Path)
    if CFG.model_path:
        model_path = Path(CFG.model_path) if isinstance(CFG.model_path, str) else CFG.model_path
        if model_path.exists():
            return model_path
    
    # Option 2: Dataset name
    if CFG.model_dataset_name:
        model_path = CFG.kaggle_input / CFG.model_dataset_name
        if model_path.exists():
            # Check if config.json is in root or subdirectory
            if (model_path / "config.json").exists():
                return model_path
            for sub in model_path.glob("**/config.json"):
                return sub.parent
        raise FileNotFoundError(f"Model dataset not found: {model_path}")
    
    # Option 3: Auto-detect from /kaggle/input
    if is_kaggle():
        for d in CFG.kaggle_input.iterdir():
            if not d.is_dir():
                continue
            # Skip competition data
            if (d / "test.csv").exists():
                continue
            # Check for model files
            if (d / "config.json").exists():
                return d
            for sub in d.glob("**/config.json"):
                return sub.parent
    
    # Option 4: Local trained model
    local_model = Path("outputs/akkadian_v1/final")
    if local_model.exists():
        return local_model
    
    raise FileNotFoundError(
        "Could not find model directory. "
        "Set CFG.model_dataset_name or CFG.model_path"
    )


COMP_DATA_DIR = find_competition_data()
MODEL_DIR = find_model_dir()

print(f"📁 Competition data: {COMP_DATA_DIR}")
print(f"🤖 Model directory: {MODEL_DIR}")
print(f"🖥️ Running on Kaggle: {is_kaggle()}")
print(f"🎮 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

#%% [markdown]
# ## 3. Data Preprocessing (same as training)

#%%
# Subscript conversion map
_SUBSCRIPT_MAP = str.maketrans({
    "\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u2084": "4",
    "\u2085": "5", "\u2086": "6", "\u2087": "7", "\u2088": "8", "\u2089": "9",
    "\u2093": "x",
})


def normalize_transliteration(text: str) -> str:
    """Normalize Akkadian transliteration for model input.
    
    Important: This must match the preprocessing used during training!
    """
    if pd.isna(text):
        return ""
    text = str(text)
    
    # Unicode normalization
    text = unicodedata.normalize("NFC", text)
    
    # Normalize special H character
    text = text.replace("\u1E2A", "H").replace("\u1E2B", "h")
    
    # Convert subscripts to numbers
    text = text.translate(_SUBSCRIPT_MAP)
    
    # Handle gaps and damaged portions
    text = text.replace("\u2026", " <gap> ")  # ellipsis
    text = re.sub(r"\.\.\.+", " <gap> ", text)
    text = re.sub(r"\[([^\]]*)\]", " <gap> ", text)  # [damaged text]
    
    # Handle unknown signs
    text = re.sub(r"\bx\b", " <unk> ", text)
    
    # Remove editorial marks
    text = re.sub(r"[!?/]", " ", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


#%% [markdown]
# ## 4. Load Model

#%%
from transformers import ByT5Tokenizer

print(f"🤖 Loading model from {MODEL_DIR}")

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

# Force ByT5Tokenizer and compute extra_ids from model config.
# ByT5 vocab = 256 bytes + 3 specials + extra_ids => extra_ids = vocab_size - 259
vocab_size = getattr(model.config, "vocab_size", None)
extra_ids = 125
if isinstance(vocab_size, int):
    extra_ids = max(vocab_size - 259, 0)
tokenizer = ByT5Tokenizer(extra_ids=extra_ids)
print(f"   ✅ Using ByT5Tokenizer(extra_ids={extra_ids})")

# Sanity check for potential mismatch
if isinstance(vocab_size, int) and tokenizer.vocab_size != vocab_size:
    print(
        f"   ⚠️ Tokenizer vocab_size ({tokenizer.vocab_size}) != model vocab_size ({vocab_size})"
    )

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Use FP16 for faster inference
if CFG.fp16 and device.type == "cuda":
    model = model.half()
    print("   ✅ Using FP16 for inference")

model.eval()
print(f"   ✅ Model loaded on {device}")

#%% [markdown]
# ## 5. Inference Functions

#%%
@torch.no_grad()
def generate_batch(texts: List[str]) -> List[str]:
    """Generate translations for a batch of texts."""
    # Tokenize
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=CFG.max_source_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_length=CFG.max_target_length,
        num_beams=CFG.num_beams,
        early_stopping=True,
    )
    
    # Decode
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return translations


def translate_all(texts: List[str], batch_size: int = None) -> List[str]:
    """Translate all texts with batching and progress bar."""
    if batch_size is None:
        batch_size = CFG.batch_size
    
    all_translations = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i:i + batch_size]
        translations = generate_batch(batch)
        all_translations.extend(translations)
    
    return all_translations


#%% [markdown]
# ## 6. Load Test Data & Run Inference

#%%
print("📖 Loading test data...")
test_df = pd.read_csv(COMP_DATA_DIR / "test.csv")
print(f"   Test samples: {len(test_df)}")

# Check columns
required_cols = {"id", "transliteration"}
if not required_cols.issubset(test_df.columns):
    raise ValueError(f"Test data missing columns: {required_cols - set(test_df.columns)}")

# Normalize input
print("🔧 Normalizing transliterations...")
normalized_texts = [
    normalize_transliteration(t) 
    for t in test_df["transliteration"].tolist()
]

# Show sample
print("\nSample input (normalized):")
for i in range(min(2, len(normalized_texts))):
    print(f"  [{i}] {normalized_texts[i][:100]}...")

#%%
print("\n🚀 Running inference...")
translations = translate_all(normalized_texts)

# Show sample outputs
print("\nSample outputs:")
for i in range(min(2, len(translations))):
    print(f"  [{i}] {translations[i][:150]}...")

#%% [markdown]
# ## 7. Create Submission

#%%
submission = pd.DataFrame({
    "id": test_df["id"],
    "translation": translations,
})

# Validate
assert len(submission) == len(test_df), "Submission length mismatch!"
assert submission["translation"].notna().all(), "Found NaN translations!"

# Save
submission_path = CFG.kaggle_working / "submission.csv" if is_kaggle() else Path("submission.csv")
submission.to_csv(submission_path, index=False)

print(f"\n✅ Submission saved to: {submission_path}")
print(f"   Total predictions: {len(submission)}")

#%%
# Preview submission
print("\n📄 Submission preview:")
print(submission.head())

#%% [markdown]
# ## 8. Sanity Check

#%%
# Load sample submission for comparison
sample_sub = pd.read_csv(COMP_DATA_DIR / "sample_submission.csv")

print("\n🔍 Comparison with sample submission:")
print(f"   Sample submission shape: {sample_sub.shape}")
print(f"   Our submission shape: {submission.shape}")

# Check if IDs match
if set(submission["id"]) == set(sample_sub["id"]):
    print("   ✅ IDs match!")
else:
    print("   ❌ ID mismatch!")

# Show comparison
print("\n📊 Side-by-side comparison (first 2):")
for i in range(min(2, len(submission))):
    print(f"\n[ID: {submission.iloc[i]['id']}]")
    print(f"  Sample: {sample_sub.iloc[i]['translation'][:100]}...")
    print(f"  Ours:   {submission.iloc[i]['translation'][:100]}...")

#%% [markdown]
# ## 9. Done!
#
# Your `submission.csv` is ready for submission to Kaggle.
#
# **Next Steps:**
# 1. Click "Save Version" in Kaggle
# 2. Go to the competition page
# 3. Submit the output file

#%%
print("\n" + "=" * 60)
print("🎉 Inference complete!")
print("=" * 60)
print(f"\n📁 Submission file: {submission_path}")
print("\nTo submit:")
print("1. Save this notebook version")
print("2. Go to competition page → Submit")
print("3. Select this notebook's output")
