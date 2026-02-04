# %% [markdown]
# # Akkadian V5b Inference (Retrieval + Glossary Prompt)

# %% [markdown]
# ## 1. Imports & Configuration

# %%
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import shutil
import tempfile
import unicodedata
from collections import Counter

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

    model_size: str = "small"  # "small", "base" or "large"
    model_path: Path = field(init=False)

    max_source_length: int = 256
    max_target_length: int = 256
    batch_size: int = 4
    num_beams: int = 4

    # Retrieval + glossary
    tm_k: int = 5
    glossary_max_items: int = 8
    max_prompt_chars: int = 512
    prefer_tfidf: bool = True
    jaccard_max_candidates: int = 500

    def __post_init__(self):
        if self.model_size == "small":
            self.model_path = self.kaggle_input / "akkadian-v5b-small/pytorch/default/1"
        elif self.model_size == "base":
            self.model_path = self.kaggle_input / "akkadian-v5b-base/pytorch/default/1"
        else:
            self.model_path = self.kaggle_input / "akkadian-v5b-large/pytorch/default/1"


CFG = Config(model_size="small")

# -----------------------------
# V5 normalization (inline for Kaggle)
# -----------------------------

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


def normalize_transliteration(text) -> str:
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)

    # Protect literal gap tokens before removing <content> blocks
    text = text.replace("<gap>", "__LIT_GAP__")
    text = text.replace("<big_gap>", "__LIT_BIG_GAP__")

    # Remove apostrophe line numbers only (1', 1'')
    text = re.sub(r"\b\d+'{1,2}\b", " ", text)

    # Remove <content> blocks first
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
    text = text.replace("\u2039", "").replace("\u203A", "")
    text = text.replace("\u2308", "").replace("\u2309", "")
    text = text.replace("\u230A", "").replace("\u230B", "")
    text = text.replace("\u02F9", "").replace("\u02FA", "")

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


# -----------------------------
# Tokenization utilities
# -----------------------------

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


# -----------------------------
# Retrieval
# -----------------------------


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
        cand = set()
        freq = Counter()
        for g in grams:
            for idx in self.inverted.get(g, []):
                freq[idx] += 1
        if freq:
            for idx, _ in freq.most_common(self.max_candidates):
                cand.add(idx)
        else:
            cand = set(range(len(self.texts)))

        scores = []
        for idx in cand:
            inter = len(grams & self.grams[idx])
            union = len(grams) + len(self.grams[idx]) - inter
            score = inter / union if union else 0.0
            scores.append((score, idx))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [idx for _, idx in scores[:k]]


class TfidfRetriever:
    def __init__(self, texts: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, k: int = 5) -> list[int]:
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        if k >= len(scores):
            idxs = np.argsort(-scores)
        else:
            idxs = np.argpartition(-scores, k - 1)[:k]
            idxs = idxs[np.argsort(-scores[idxs])]
        return idxs.tolist()


def build_retriever(texts: list[str], prefer_tfidf: bool = True) -> object:
    if prefer_tfidf:
        try:
            return TfidfRetriever(texts)
        except Exception:
            pass
    return JaccardRetriever(texts, max_candidates=CFG.jaccard_max_candidates)


# -----------------------------
# Glossary prompt (local + global)
# -----------------------------


def build_prompt_with_retrieval(
    src: str,
    tm_pairs: list[dict],
    retriever: object | None,
    glossary: dict[str, list[str]] | None,
    max_items: int,
    max_prompt_chars: int,
    tm_k: int,
) -> str:
    if not tm_pairs or retriever is None:
        return src

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


# %% [markdown]
# ## 2. Helper Functions

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
    # Prefer a dataset containing v5b assets
    if not is_kaggle():
        local = Path("data/v5b")
        if local.exists():
            return local
        return None

    for d in CFG.kaggle_input.iterdir():
        if (d / "v5b_glossary.json").exists() or (d / "v5b_tm_pairs.jsonl").exists():
            return d
    return None


def find_model() -> Path:
    if not is_kaggle():
        local = Path(f"outputs/akkadian_v5b_{CFG.model_size}/model")
        if local.exists():
            return local
        raise FileNotFoundError("Local model not found")

    if CFG.model_path.exists():
        return CFG.model_path

    for d in CFG.kaggle_input.iterdir():
        if "v5b" in d.name.lower() and CFG.model_size in d.name.lower():
            if (d / "config.json").exists():
                return d
            for sub in d.glob("**/config.json"):
                return sub.parent

    raise FileNotFoundError(f"V5b-{CFG.model_size} model not found")


def load_tm_pairs(path: Path, max_rows: int | None = None) -> list[dict]:
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs


def load_glossary(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: list(v) for k, v in data.items()}


# %% [markdown]
# ## 3. Setup

# %%
print("=" * 60)
print(f"🚀 Akkadian V5b Inference: {CFG.model_size.upper()}")
print("=" * 60)

COMP_DIR = find_competition_data()
MODEL_DIR = find_model()
ASSETS_DIR = find_assets_dir()

print(f"📁 Competition data: {COMP_DIR}")
print(f"🤖 Model: {MODEL_DIR}")
print(f"🧠 Assets: {ASSETS_DIR if ASSETS_DIR else 'not found'}")
print(f"🎮 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)


# %% [markdown]
# ## 4. Load Model

# %%
print(f"\n🤖 Loading model from: {MODEL_DIR}")

model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))
print(f"   Model vocab: {model.config.vocab_size}")
print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")

def load_autotokenizer_with_fix(model_dir: Path):
    """Load AutoTokenizer from local files only.
    If tokenizer_config.json is malformed (extra_special_tokens as list),
    patch it in a temp directory and retry.
    """
    try:
        return AutoTokenizer.from_pretrained(str(model_dir), use_fast=False, local_files_only=True)
    except Exception as e:
        print(f"   ⚠️ AutoTokenizer local load failed: {e}")
        src_cfg = model_dir / "tokenizer_config.json"
        if not src_cfg.exists():
            raise

        with src_cfg.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        extra = cfg.get("extra_special_tokens")
        if isinstance(extra, list):
            # transformers expects dict here; list triggers AttributeError(keys)
            cfg["extra_special_tokens"] = {}
        else:
            raise

        tmp_dir = Path(tempfile.mkdtemp(prefix="tokfix_"))
        for name in ["tokenizer_config.json", "special_tokens_map.json", "config.json"]:
            src = model_dir / name
            if src.exists():
                shutil.copy2(src, tmp_dir / name)
        with (tmp_dir / "tokenizer_config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        print("   ↪ Patched tokenizer_config.json (extra_special_tokens list -> dict) and retrying AutoTokenizer")
        return AutoTokenizer.from_pretrained(str(tmp_dir), use_fast=False, local_files_only=True)


tokenizer = load_autotokenizer_with_fix(MODEL_DIR)

print(f"   Tokenizer vocab: {len(tokenizer)}")
print(f"   Tokenizer class: {tokenizer.__class__.__name__}")

assert len(tokenizer) == model.config.vocab_size, "Vocab mismatch!"
print("   ✅ Vocab match")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"   ✅ Model on {device}")


# %% [markdown]
# ## 5. Load TM + Glossary

# %%
TM_PAIRS = []
GLOSSARY = None
RETRIEVER = None

if ASSETS_DIR:
    tm_path = ASSETS_DIR / "v5b_tm_pairs.jsonl"
    glossary_path = ASSETS_DIR / "v5b_glossary.json"

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
    RETRIEVER = build_retriever([p.get("src", "") for p in TM_PAIRS], prefer_tfidf=CFG.prefer_tfidf)


# %% [markdown]
# ## 6. Inference

# %%
@torch.no_grad()
def generate_batch(texts, debug: bool = False):
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

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def translate_all(texts, batch_size=None):
    if batch_size is None:
        batch_size = CFG.batch_size

    translations = []
    pbar = tqdm(range(0, len(texts), batch_size), desc="🔮 Translating", unit="batch")
    for i in pbar:
        batch = texts[i : i + batch_size]
        translations.extend(generate_batch(batch))
    return translations


# %% [markdown]
# ## 7. Run Inference

# %%
print("\n📖 Loading test data...")
test_df = pd.read_csv(COMP_DIR / "test.csv")
print(f"   Test samples: {len(test_df):,}")

print("\n🔧 Normalizing (V5b)...")
normalized = [normalize_transliteration(t) for t in tqdm(test_df["transliteration"], desc="Normalizing")]

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

print(f"\n📝 Sample prompts:")
for i in range(min(2, len(prompts))):
    print(f"   [{i}] {prompts[i][:120]}...")

print("\n🚀 Running inference...")
print("\n[DEBUG] First sample test...")
_test = generate_batch([prompts[0]], debug=True)
print(f"[DEBUG] Translation: '{_test[0][:100]}...'")

translations = translate_all(prompts)

empty_count = sum(1 for t in translations if not t or not t.strip())
if empty_count > 0:
    print(f"\n⚠️ WARNING: {empty_count} empty translations!")

print(f"\n📝 Sample outputs:")
for i in range(min(3, len(translations))):
    print(f"   [{i}] {translations[i][:150]}...")


# %% [markdown]
# ## 8. Submission

# %%
submission = pd.DataFrame({
    "id": test_df["id"],
    "translation": translations,
})

assert len(submission) == len(test_df), "Length mismatch!"
assert submission["translation"].notna().all(), "NaN values!"

output_path = CFG.kaggle_working / "submission.csv" if is_kaggle() else Path("submission.csv")
submission.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print(f"✅ V5b-{CFG.model_size.upper()} Inference Complete!")
print("=" * 60)
print(f"📁 Saved: {output_path}")
print(f"   Rows: {len(submission)}")
print()
print(submission.head())
print("=" * 60)
