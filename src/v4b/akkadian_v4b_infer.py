#%% [markdown]
# # Akkadian V4b Inference
#
# **Key Features:**
# - Competition-compliant preprocessing
#   - `[content]` → content (keep brackets content)
#   - `[x]` → `<gap>`
#   - `…` → `<big_gap>`
# - NO repetition_penalty (hurts BLEU)
# - ByT5Tokenizer for consistency
#
# **Environment**: Kaggle T4/P100 GPU

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
    """V4b Inference configuration."""
    kaggle_input: Path = Path("/kaggle/input")
    kaggle_working: Path = Path("/kaggle/working")
    
    model_size: str = "base"
    model_path: Path = field(init=False)
    
    max_source_length: int = 512
    max_target_length: int = 512
    batch_size: int = 4
    num_beams: int = 4
    
    def __post_init__(self):
        if self.model_size == "base":
            self.model_path = self.kaggle_input / "akkadian-v4b-base/pytorch/default/1"
        else:
            self.model_path = self.kaggle_input / "akkadian-v4b-large/pytorch/default/1"


CFG = Config(model_size="base")

#%% [markdown]
# ## 2. Helper Functions

#%%
def is_kaggle():
    return CFG.kaggle_input.exists()


def find_competition_data():
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


def find_model():
    if not is_kaggle():
        local = Path(f"outputs/akkadian_v4b_{CFG.model_size}/model")
        if local.exists():
            return local
        raise FileNotFoundError("Local model not found")
    
    if CFG.model_path.exists():
        return CFG.model_path
    
    for d in CFG.kaggle_input.iterdir():
        if "v4b" in d.name.lower() and CFG.model_size in d.name.lower():
            if (d / "config.json").exists():
                return d
            for sub in d.glob("**/config.json"):
                return sub.parent
    
    raise FileNotFoundError(f"V4b-{CFG.model_size} model not found")

#%% [markdown]
# ## 3. Competition-Compliant Normalization

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

_QUOTE_MAP = {
    '\u201e': '"', '\u201c': '"', '\u201d': '"',
    '\u2018': "'", '\u2019': "'", '\u201a': "'",
    '\u02be': "'", '\u02bf': "'",
}

_SUBSCRIPT_MAP = str.maketrans({
    '\u2080': '0', '\u2081': '1', '\u2082': '2', '\u2083': '3', '\u2084': '4',
    '\u2085': '5', '\u2086': '6', '\u2087': '7', '\u2088': '8', '\u2089': '9',
    '\u2093': 'x',
})

_FULL_MAP = str.maketrans({**_VOWEL_MAP, **_CONSONANT_MAP, **_QUOTE_MAP})


def normalize_transliteration(text) -> str:
    """
    Normalize following competition guidelines.
    
    Key rules:
    - [x] → <gap>
    - … or [… …] → <big_gap>
    - [content] → content (remove brackets, keep content)
    - <content> → content
    - ˹ ˺ removed
    - ! ? / : removed (scribal notations, word dividers)
    
    NOTE: Line numbers NOT removed - train.csv has quantities like "1 TÚG".
    """
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    
    # Protect literal gap tokens before removing <content> blocks
    text = text.replace("<gap>", "__LIT_GAP__")
    text = text.replace("<big_gap>", "__LIT_BIG_GAP__")
    
    # Remove only apostrophe-style line numbers (e.g., 1', 1'')
    text = re.sub(r"\b\d+'{1,2}\b", " ", text)
    
    # 1. Handle <content> → content (scribal insertions) FIRST
    #    Do this before gap tokens are created to avoid removing them!
    text = re.sub(r'<<([^>]+)>>', r'\1', text)  # errant signs
    text = re.sub(r'<([^>]+)>', r'\1', text)    # scribal insertions
    
    # 2. Large gaps: [… …] or [...] → __BIG_GAP__
    text = re.sub(r'\[\s*…+\s*…*\s*\]', ' __BIG_GAP__ ', text)
    text = re.sub(r'\[\s*\.\.\.+\s*\.\.\.+\s*\]', ' __BIG_GAP__ ', text)
    
    # 3. Ellipsis → __BIG_GAP__
    text = text.replace('\u2026', ' __BIG_GAP__ ')
    text = re.sub(r'\.\.\.+', ' __BIG_GAP__ ', text)
    
    # 4. [x] → __GAP__
    text = re.sub(r'\[\s*x\s*\]', ' __GAP__ ', text, flags=re.IGNORECASE)
    
    # 5. [content] → content
    text = re.sub(r'\[([^\]]+)\]', r'\1', text)
    
    # 6. Remove half brackets (all variations)
    text = text.replace('\u2039', '').replace('\u203a', '')  # ‹ ›
    text = text.replace('\u2308', '').replace('\u2309', '')  # ⌈ ⌉
    text = text.replace('\u230a', '').replace('\u230b', '')  # ⌊ ⌋
    text = text.replace('˹', '').replace('˺', '')  # literal
    
    # 7. Character maps
    text = text.translate(_FULL_MAP)
    text = text.translate(_SUBSCRIPT_MAP)
    
    # 8. Remove scribal notations AND word dividers
    text = re.sub(r'[!?/]', ' ', text)
    text = re.sub(r'\s*:\s*', ' ', text)  # : word divider
    
    # 9. Standalone x → __GAP__
    text = re.sub(r'\bx\b', ' __GAP__ ', text, flags=re.IGNORECASE)
    
    # 10. Convert placeholders to actual tokens
    text = text.replace('__GAP__', '<gap>')
    text = text.replace('__BIG_GAP__', '<big_gap>')
    
    # Restore protected literal tokens
    text = text.replace("__LIT_GAP__", "<gap>")
    text = text.replace("__LIT_BIG_GAP__", "<big_gap>")
    
    # 11. Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

#%% [markdown]
# ## 4. Setup

#%%
print("=" * 60)
print(f"🚀 Akkadian V4b Inference: {CFG.model_size.upper()}")
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

tokenizer = ByT5Tokenizer(extra_ids=125)
print(f"   Tokenizer vocab: {len(tokenizer)}")

model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))
print(f"   Model vocab: {model.config.vocab_size}")
print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")

assert len(tokenizer) == model.config.vocab_size, "Vocab mismatch!"
print("   ✅ Vocab match")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"   ✅ Model on {device}")

#%% [markdown]
# ## 6. Inference

#%%
@torch.no_grad()
def generate_batch(texts, debug=False):
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
    
    # NO repetition_penalty - hurts BLEU
    outputs = model.generate(
        **inputs,
        max_length=CFG.max_target_length,
        num_beams=CFG.num_beams,
        early_stopping=True,
    )
    
    if debug:
        print(f"   [DEBUG] Output shape: {outputs.shape}")
    
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    if debug:
        print(f"   [DEBUG] Decoded: '{decoded[0][:100]}'")
    
    return decoded


def translate_all(texts, batch_size=None):
    if batch_size is None:
        batch_size = CFG.batch_size
    
    translations = []
    pbar = tqdm(range(0, len(texts), batch_size), desc="🔮 Translating", unit="batch")
    
    for i in pbar:
        batch = texts[i:i + batch_size]
        results = generate_batch(batch)
        translations.extend(results)
    
    return translations

#%% [markdown]
# ## 7. Run Inference

#%%
print("\n📖 Loading test data...")
test_df = pd.read_csv(COMP_DIR / "test.csv")
print(f"   Test samples: {len(test_df):,}")

print("\n🔧 Normalizing (Competition Guidelines)...")
normalized = [normalize_transliteration(t) for t in tqdm(test_df["transliteration"], desc="Normalizing")]

print(f"\n📝 Sample normalized:")
for i in range(min(2, len(normalized))):
    print(f"   [{i}] {normalized[i][:100]}...")

#%%
print("\n🚀 Running inference...")

print("\n[DEBUG] First sample test...")
test_result = generate_batch([normalized[0]], debug=True)
print(f"[DEBUG] Translation: '{test_result[0][:100]}...'")

translations = translate_all(normalized)

empty_count = sum(1 for t in translations if not t or not t.strip())
if empty_count > 0:
    print(f"\n⚠️ WARNING: {empty_count} empty translations!")

print(f"\n📝 Sample outputs:")
for i in range(min(3, len(translations))):
    print(f"   [{i}] {translations[i][:150]}...")

#%% [markdown]
# ## 8. Submission

#%%
submission = pd.DataFrame({
    "id": test_df["id"],
    "translation": translations,
})

assert len(submission) == len(test_df), "Length mismatch!"
assert submission["translation"].notna().all(), "NaN values!"

output_path = CFG.kaggle_working / "submission.csv" if is_kaggle() else Path("submission.csv")
submission.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print(f"✅ V4b-{CFG.model_size.upper()} Inference Complete!")
print("=" * 60)
print(f"📁 Saved: {output_path}")
print(f"   Rows: {len(submission)}")
print()
print(submission.head())
print("=" * 60)
