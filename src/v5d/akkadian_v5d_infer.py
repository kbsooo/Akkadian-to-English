# %% [markdown]
# # Akkadian V5d Inference
# 
# - Model: ByT5-small (fine-tuned)
# - Tokenizer: Loaded from Kaggle Models (Internet OFF compatible)
# - Features: Glossary + TM Retrieval

# %% [markdown]
# ## 1. Imports & Configuration

# %%
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

# %%
@dataclass
class Config:
    kaggle_input: Path = Path("/kaggle/input")
    kaggle_working: Path = Path("/kaggle/working")
    
    # Model paths on Kaggle
    model_name: str = "akkadian-v5d"
    tokenizer_name: str = "byt5-small"  # Kaggle model name for tokenizer
    
    model_path: Path = field(init=False)
    tokenizer_path: Path = field(init=False)
    
    # Inference settings
    max_source_length: int = 256
    max_target_length: int = 256
    batch_size: int = 8  # T4 x 2 has enough memory for larger batches
    num_beams: int = 4
    
    # Retrieval + glossary
    tm_k: int = 5
    glossary_max_items: int = 8
    max_prompt_chars: int = 512
    
    def __post_init__(self):
        self.model_path = self.kaggle_input / f"{self.model_name}/pytorch/default/1"
        self.tokenizer_path = self.kaggle_input / f"{self.tokenizer_name}/pytorch/default/1"


CFG = Config()

# %% [markdown]
# ## 2. Normalization

# %%
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
    """Normalize Akkadian transliteration."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)

    text = text.replace("<gap>", "__LIT_GAP__")
    text = text.replace("<big_gap>", "__LIT_BIG_GAP__")
    text = re.sub(r"\b\d+'{1,2}\b", " ", text)
    text = re.sub(r"<<([^>]+)>>", r"\1", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)
    text = re.sub(r"\[\s*…+\s*…*\s*\]", " __BIG_GAP__ ", text)
    text = re.sub(r"\[\s*\.\.\.+\s*\.\.\.+\s*\]", " __BIG_GAP__ ", text)
    text = text.replace("…", " __BIG_GAP__ ")
    text = re.sub(r"\.\.\.+", " __BIG_GAP__ ", text)
    text = re.sub(r"\[\s*x\s*\]", " __GAP__ ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)
    for char in "‹›⌈⌉⌊⌋˹˺":
        text = text.replace(char, "")
    text = text.translate(_TRANS_TABLE)
    text = re.sub(r"[!?/]", " ", text)
    text = re.sub(r"\s*:\s*", " ", text)
    text = re.sub(r"\bx\b", " __GAP__ ", text, flags=re.IGNORECASE)
    text = text.replace("__GAP__", "<gap>")
    text = text.replace("__BIG_GAP__", "<big_gap>")
    text = text.replace("__LIT_GAP__", "<gap>")
    text = text.replace("__LIT_BIG_GAP__", "<big_gap>")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# %% [markdown]
# ## 3. Glossary & Retrieval

# %%
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
    if not tm_pairs or retriever is None:
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

# %% [markdown]
# ## 4. Setup & Path Discovery

# %%
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
    if not is_kaggle():
        local = Path("data/v5d")
        if local.exists():
            return local
        return None
    for d in CFG.kaggle_input.iterdir():
        if (d / "v5d_glossary.json").exists() or (d / "v5d_tm_pairs.jsonl").exists():
            return d
    return None


def find_model() -> Path:
    if not is_kaggle():
        local = Path("outputs/v5d/model")
        if local.exists():
            return local
        raise FileNotFoundError("Local model not found")
    if CFG.model_path.exists():
        return CFG.model_path
    for d in CFG.kaggle_input.iterdir():
        if "v5d" in d.name.lower():
            if (d / "config.json").exists():
                return d
            for sub in d.glob("**/config.json"):
                return sub.parent
    raise FileNotFoundError("V5d model not found")


def find_tokenizer() -> Path:
    """Find tokenizer path - must be loaded from Kaggle Models (Internet OFF)"""
    if not is_kaggle():
        # Local: use HuggingFace
        return Path("google/byt5-small")
    
    if CFG.tokenizer_path.exists():
        return CFG.tokenizer_path
    
    # Search for byt5 tokenizer in Kaggle inputs
    for d in CFG.kaggle_input.iterdir():
        if "byt5" in d.name.lower():
            if (d / "tokenizer_config.json").exists():
                return d
            for sub in d.glob("**/tokenizer_config.json"):
                return sub.parent
    
    raise FileNotFoundError(
        "Tokenizer not found! Upload 'google/byt5-small' to Kaggle Models "
        "and add it as a data source."
    )

# %%
print("=" * 60)
print("🚀 Akkadian V5d Inference")
print("=" * 60)

COMP_DIR = find_competition_data()
MODEL_DIR = find_model()
ASSETS_DIR = find_assets_dir()
TOKENIZER_DIR = find_tokenizer()

print(f"📁 Competition data: {COMP_DIR}")
print(f"🤖 Model: {MODEL_DIR}")
print(f"🔤 Tokenizer: {TOKENIZER_DIR}")
print(f"🧠 Assets: {ASSETS_DIR if ASSETS_DIR else 'not found'}")
print(f"🎮 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# %% [markdown]
# ## 5. Load Tokenizer

# %%
print(f"🔤 Loading tokenizer from: {TOKENIZER_DIR}")

# Load from local path (Kaggle) or HuggingFace name (local dev)
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=is_kaggle())
print(f"   Tokenizer vocab: {len(tokenizer)}")
print(f"   ✅ Tokenizer loaded")

# %% [markdown]
# ## 6. Load Model

# %%
print(f"🤖 Loading model from: {MODEL_DIR}")

from transformers import T5Config, T5ForConditionalGeneration
from safetensors.torch import load_file

config = T5Config.from_pretrained(str(MODEL_DIR))
print(f"   Config tie_word_embeddings: {config.tie_word_embeddings}")
config.tie_word_embeddings = False

# Initialize model
model = T5ForConditionalGeneration(config)

# T5 always ties encoder/decoder embed_tokens to shared - we need to untie them
# before loading weights so they can receive their own trained values
def ensure_untied_embeddings(model, config):
    """Create separate embedding modules for encoder/decoder (not tied to shared)."""
    # Create brand new embedding modules
    model.encoder.embed_tokens = torch.nn.Embedding(config.vocab_size, config.d_model)
    model.decoder.embed_tokens = torch.nn.Embedding(config.vocab_size, config.d_model)
    
    # Also ensure lm_head is separate
    model.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
    
    print(f"   Created separate encoder/decoder/lm_head embeddings")

ensure_untied_embeddings(model, config)
print(f"   Model initialized with untied embeddings")

# Load weights
weights_path = MODEL_DIR / "model.safetensors"
if weights_path.exists():
    state_dict = load_file(str(weights_path))
    model.load_state_dict(state_dict, strict=True)
    print(f"   ✅ Loaded weights from safetensors")
else:
    weights_path = MODEL_DIR / "pytorch_model.bin"
    state_dict = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    print(f"   ✅ Loaded weights from pytorch_model.bin")

print(f"   Model vocab: {model.config.vocab_size}")
print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")

# Verify all 4 embeddings are separate
embed_params = {name: p.numel() for name, p in model.named_parameters() 
                if 'embed' in name or 'lm_head' in name or 'shared' in name}
print(f"   Embedding params: {embed_params}")

if len(embed_params) != 4:
    raise RuntimeError(f"Expected 4 embedding params, got {len(embed_params)}!")

if len(tokenizer) != model.config.vocab_size:
    print(f"   ⚠️ WARNING: Vocab mismatch!")
else:
    print("   ✅ Vocab match")

# GPU setup (note: generate() doesn't support DataParallel, so we use single GPU)
n_gpus = torch.cuda.device_count()
print(f"   Available GPUs: {n_gpus}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Use first GPU for inference
else:
    device = torch.device("cpu")

model = model.to(device)
model.eval()
print(f"   ✅ Model on {device}")
if n_gpus > 1:
    print(f"   ℹ️ Note: Using larger batch_size={CFG.batch_size} to utilize GPU memory")

# %% [markdown]
# ## 7. Load TM & Glossary

# %%
TM_PAIRS = []
GLOSSARY = None
RETRIEVER = None

if ASSETS_DIR:
    tm_path = ASSETS_DIR / "v5d_tm_pairs.jsonl"
    glossary_path = ASSETS_DIR / "v5d_glossary.json"

    if tm_path.exists():
        TM_PAIRS = load_tm_pairs(tm_path)
        print(f"🧠 TM pairs: {len(TM_PAIRS):,}")
    else:
        print("🧠 TM pairs: not found")

    if glossary_path.exists():
        GLOSSARY = load_glossary(glossary_path)
        print(f"🧠 Glossary size: {len(GLOSSARY):,}")
    else:
        print("🧠 Glossary: not found")

if TM_PAIRS:
    RETRIEVER = JaccardRetriever([p.get("src", "") for p in TM_PAIRS])
    print("🧠 Retriever built")

# %% [markdown]
# ## 8. Sanity Check

# %%
print("🔍 Sanity check...")
test_input = "um-ma"
inputs = tokenizer(test_input, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50, num_beams=4)

test_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"   Input: '{test_input}'")
print(f"   Output: '{test_output}'")

if not test_output or test_output.strip() == "":
    print("   ⚠️ WARNING: Empty output!")
else:
    print("   ✅ Model produces non-empty output")

# %% [markdown]
# ## 9. Load Test Data

# %%
print("📖 Loading test data...")
test_df = pd.read_csv(COMP_DIR / "test.csv")
print(f"   Test samples: {len(test_df):,}")
print(test_df.head())

# %% [markdown]
# ## 10. Normalize & Build Prompts

# %%
print("🔧 Normalizing...")
normalized = [normalize_transliteration(t) for t in tqdm(test_df["transliteration"], desc="Normalizing")]

print(f"\n📝 Sample normalized:")
for i in range(min(2, len(normalized))):
    print(f"   [{i}] {normalized[i][:100]}...")

# %%
print("🧠 Building glossary prompts...")
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

print(f"\n📝 Sample prompts:")
for i in range(min(2, len(prompts))):
    print(f"   [{i}] {prompts[i][:120]}...")

# %% [markdown]
# ## 11. Inference

# %%

@torch.no_grad()
def generate_batch(texts: list[str]) -> list[str]:
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

    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Fallback for empty outputs
    return [r if r and r.strip() else "[translation unavailable]" for r in results]


def translate_all(texts: list[str]) -> list[str]:
    translations = []
    pbar = tqdm(range(0, len(texts), CFG.batch_size), desc="🔮 Translating", unit="batch")
    for i in pbar:
        batch = texts[i : i + CFG.batch_size]
        translations.extend(generate_batch(batch))
    return translations

# %%
print("🚀 Running inference...")
translations = translate_all(prompts)

empty_count = sum(1 for t in translations if t == "[translation unavailable]")
if empty_count > 0:
    print(f"\n⚠️ WARNING: {empty_count} fallback translations!")

print(f"\n📝 Sample outputs:")
for i in range(min(3, len(translations))):
    print(f"   [{i}] {translations[i][:150]}...")

# %% [markdown]
# ## 12. Submission

# %%
submission = pd.DataFrame({
    "id": test_df["id"],
    "translation": translations,
})

assert len(submission) == len(test_df), "Length mismatch!"
assert submission["translation"].notna().all(), "NaN values!"

print(submission.head())

# %%
output_path = CFG.kaggle_working / "submission.csv" if is_kaggle() else Path("submission.csv")
submission.to_csv(output_path, index=False)

print("=" * 60)
print("✅ V5d Inference Complete!")
print("=" * 60)
print(f"📁 Saved: {output_path}")
print(f"   Rows: {len(submission)}")
