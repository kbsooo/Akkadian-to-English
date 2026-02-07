# %% [markdown]
# # V7 Inference — Akkadian→English (Kaggle T4×2)

# %% 
# Configuration
MODEL_DATASET = "akkadian-v7-model"  # Kaggle dataset name containing model weights
ONOMASTICON_DATASET = "deeppast/old-assyrian-grammars-and-other-resources"

MAX_SOURCE_LENGTH = 384
MAX_TARGET_LENGTH = 384
MBR_N_CANDIDATES = 8  # Number of beam candidates for MBR decoding

# %%
# Imports and GPU setup
import torch
import os
import re
import unicodedata
import json
import difflib
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

n_gpus = torch.cuda.device_count()
print(f"GPUs available: {n_gpus}")
for i in range(n_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f"         VRAM: {mem:.1f} GB")

# %%
# Find paths (robust for Kaggle)
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# Find competition data
COMP_DIR = None
for d in KAGGLE_INPUT.iterdir():
    if (d / "test.csv").exists():
        COMP_DIR = d
        break
assert COMP_DIR is not None, f"Competition data not found in {KAGGLE_INPUT}"
print(f"Competition data: {COMP_DIR}")

# Find model
MODEL_DIR = None
for d in KAGGLE_INPUT.iterdir():
    if MODEL_DATASET.replace("-", "") in d.name.replace("-", ""):
        # Search for config.json
        if (d / "config.json").exists():
            MODEL_DIR = d
        else:
            for sub in d.rglob("config.json"):
                MODEL_DIR = sub.parent
                break
        if MODEL_DIR:
            break
assert MODEL_DIR is not None, f"Model not found in {KAGGLE_INPUT}"
print(f"Model: {MODEL_DIR}")
print(f"Model files: {sorted(os.listdir(MODEL_DIR))}")

# Find Onomasticon
ONOMASTICON_PATH = None
for d in KAGGLE_INPUT.iterdir():
    for f in d.rglob("Onomasticon*"):
        ONOMASTICON_PATH = f
        break
    if ONOMASTICON_PATH:
        break
print(f"Onomasticon: {ONOMASTICON_PATH}")

# %%
# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
print(f"  Tokenizer vocab: {len(tokenizer)}")

# %%
# Load model
print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR), local_files_only=True)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")
print(f"  Model vocab: {model.config.vocab_size}")

# %%
# Setup multi-GPU
model.eval()

if n_gpus >= 2:
    print("Setting up dual-GPU inference...")
    model_0 = model.to('cuda:0')
    # Reload from disk instead of deepcopy — avoids doubling CPU memory peak
    # and is more reliable across different model architectures
    model_1 = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model_1 = model_1.eval().to('cuda:1')
    print(f"  Model on cuda:0: OK")
    print(f"  Model on cuda:1: OK (loaded from disk)")
    del model  # Free CPU memory
else:
    print("Single GPU mode")
    model_0 = model.to('cuda:0')
    model_1 = None

torch.cuda.empty_cache()
for i in range(n_gpus):
    allocated = torch.cuda.memory_allocated(i) / 1e9
    print(f"  GPU {i} memory allocated: {allocated:.2f} GB")

# %%
# Sanity check
test_input = "um-ma ka-ru-um"
inputs = tokenizer(test_input, return_tensors="pt").to('cuda:0')
with torch.no_grad():
    out = model_0.generate(**inputs, max_length=50, num_beams=4)
test_output = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"Sanity check:")
print(f"  Input: '{test_input}'")
print(f"  Output: '{test_output}'")
assert test_output.strip() != "", "Model produces empty output!"
print("  OK")

# %%
# Load test data
test_df = pd.read_csv(COMP_DIR / "test.csv")
print(f"Test samples: {len(test_df)}")
print(test_df.head())
print(f"\nColumns: {list(test_df.columns)}")
print(f"text_ids: {test_df['text_id'].unique()}")

# %%
# V7 Normalization function (EXACT)
_V7_TRANS_TABLE = str.maketrans({
    "Ḫ": "H", "ḫ": "h",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9", "ₓ": "x",
    "„": '"', "\u201c": '"', "\u201d": '"',
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "ʾ": "'", "ʿ": "'",
})

def normalize_transliteration_v7(text: str) -> str:
    """V7 normalization: preserves š, ṣ, ṭ and vowel accents. Only Ḫ→H."""
    if not text or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    # Standardize determinative braces: {} → () to match test format
    text = text.replace("{", "(").replace("}", ")")
    # Protect existing gap tokens
    text = text.replace("<gap>", "\x00GAP\x00")
    text = text.replace("<big_gap>", "\x00BIGGAP\x00")
    # Remove apostrophe line numbers (1', 1'')
    text = re.sub(r"\b\d+'{1,2}\b", " ", text)
    # Remove angle-bracket content markers (keep content)
    text = re.sub(r"<<([^>]+)>>", r"\1", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)
    # Large gaps
    text = re.sub(r"\[\s*…+\s*…*\s*\]", " \x00BIGGAP\x00 ", text)
    text = re.sub(r"\[\s*\.\.\.+\s*\]", " \x00BIGGAP\x00 ", text)
    text = text.replace("…", " \x00BIGGAP\x00 ")
    text = re.sub(r"\.\.\.+", " \x00BIGGAP\x00 ", text)
    # Single gap: [x]
    text = re.sub(r"\[\s*x\s*\]", " \x00GAP\x00 ", text, flags=re.IGNORECASE)
    # Strip square brackets, keep content
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)
    # Half brackets / editorial marks
    for c in "‹›⌈⌉⌊⌋˹˺":
        text = text.replace(c, "")
    # Character map (diacritics preserved except Ḫ→H)
    text = text.translate(_V7_TRANS_TABLE)
    # Scribal notations
    text = re.sub(r"[!?/]", " ", text)
    text = re.sub(r"\s*:\s*", " ", text)
    # Standalone x → gap
    text = re.sub(r"(?<![a-zA-Z\x00])\bx\b(?![a-zA-Z])", " \x00GAP\x00 ", text)
    # Restore gap tokens
    text = text.replace("\x00GAP\x00", "<gap>")
    text = text.replace("\x00BIGGAP\x00", "<big_gap>")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# %%
# Normalize test data
normalized = [normalize_transliteration_v7(t) for t in test_df["transliteration"]]
print(f"Normalized {len(normalized)} samples")
for i, (orig, norm) in enumerate(zip(test_df["transliteration"], normalized)):
    if i < 3:  # Show first 3
        print(f"  [{i}] Original: {str(orig)[:80]}...")
        print(f"       Normalized: {norm[:80]}...")

# %%
# Define MBR decoding
from sacrebleu.metrics import CHRF
_chrf_scorer = CHRF(word_order=2)

@torch.no_grad()
def mbr_decode(model, tokenizer, source_text, n=8, device='cuda:0'):
    """
    Generate n diverse candidates via sampling, pick consensus translation.

    Beam search produces near-identical candidates (low diversity).
    Sampling with temperature creates genuinely different translations,
    which is essential for MBR consensus to work effectively.
    """
    inputs = tokenizer(source_text, return_tensors="pt", truncation=True,
                       max_length=MAX_SOURCE_LENGTH).to(device)

    try:
        # Sampling-based generation for diverse candidates
        outputs = model.generate(
            **inputs,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            num_return_sequences=n,
            max_length=MAX_TARGET_LENGTH,
        )
    except RuntimeError as e:
        # OOM fallback: reduce candidates, use greedy
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            print(f"  OOM in MBR, falling back to greedy for: '{source_text[:40]}...'")
            outputs = model.generate(
                **inputs,
                max_length=MAX_TARGET_LENGTH,
                num_beams=1,
            )
        else:
            raise

    candidates = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    # Remove empty/duplicate candidates
    candidates = list(dict.fromkeys([c for c in candidates if c.strip()]))
    if len(candidates) == 0:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    # Score each candidate against all others using chrF++
    # Precompute all pairwise scores to avoid O(n²) redundant calls
    n_cand = len(candidates)
    scores = np.zeros(n_cand)
    for i in range(n_cand):
        for j in range(n_cand):
            if i != j:
                scores[i] += _chrf_scorer.corpus_score([candidates[i]], [[candidates[j]]]).score
        scores[i] /= (n_cand - 1)

    return candidates[int(np.argmax(scores))]

# Test MBR
test_mbr = mbr_decode(model_0, tokenizer, normalized[0], n=MBR_N_CANDIDATES, device='cuda:0')
print(f"MBR test: '{test_mbr[:100]}'")

# %%
# Run inference with multi-GPU
def translate_chunk(model, tokenizer, texts, device, desc=""):
    """Translate a list of texts using MBR decoding on a specific GPU."""
    results = []
    for text in tqdm(texts, desc=desc):
        trans = mbr_decode(model, tokenizer, text, n=MBR_N_CANDIDATES, device=device)
        results.append(trans)
    return results

start = time.time()

if model_1 is not None and len(normalized) > 1:
    # Split data for dual-GPU
    mid = len(normalized) // 2
    chunk_0 = normalized[:mid]
    chunk_1 = normalized[mid:]
    
    print(f"Dual-GPU inference: {len(chunk_0)} + {len(chunk_1)} samples")
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_0 = executor.submit(translate_chunk, model_0, tokenizer, chunk_0, 'cuda:0', "GPU-0")
        future_1 = executor.submit(translate_chunk, model_1, tokenizer, chunk_1, 'cuda:1', "GPU-1")
        results_0 = future_0.result()
        results_1 = future_1.result()
    
    translations = results_0 + results_1
else:
    print(f"Single-GPU inference: {len(normalized)} samples")
    translations = translate_chunk(model_0, tokenizer, normalized, 'cuda:0', "Translating")

elapsed = time.time() - start
print(f"\nInference done in {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"Avg per sample: {elapsed/len(normalized):.1f}s")

# Show raw translations
for i, t in enumerate(translations[:3]):
    print(f"  [{i}] {t[:120]}")

# %%
# Free GPU memory
del model_0
if model_1 is not None:
    del model_1
torch.cuda.empty_cache()
print("GPU memory freed")

# %%
# Rule-based post-processing
def rule_based_postprocess(translation: str) -> str:
    if not translation:
        return ""
    # Gap marker normalization
    translation = re.sub(r'\b(?:gap)\b(?!\s*>)', '<gap>', translation, flags=re.IGNORECASE)
    translation = re.sub(r'\[(gap)\]', '<gap>', translation, flags=re.IGNORECASE)
    translation = re.sub(r'\b(?:big[_ ]gap|large[_ ]gap)\b', '<big_gap>', translation, flags=re.IGNORECASE)
    translation = re.sub(r'(<gap>\s*){2,}', '<big_gap> ', translation)
    # Number normalization
    translation = re.sub(r'\bone[- ]third\b', '0.33333', translation, flags=re.IGNORECASE)
    translation = re.sub(r'\btwo[- ]thirds?\b', '0.66666', translation, flags=re.IGNORECASE)
    translation = re.sub(r'\bone[- ]half\b', '0.5', translation, flags=re.IGNORECASE)
    # Whitespace/punctuation
    translation = re.sub(r'\s+([.,;:!?)])', r'\1', translation)
    translation = re.sub(r'([(\[])\s+', r'\1', translation)
    translation = re.sub(r'\s{2,}', ' ', translation)
    translation = re.sub(r'"{2,}', '"', translation)
    return translation.strip()

translations = [rule_based_postprocess(t) for t in translations]
print("Rule-based post-processing done")
for i, t in enumerate(translations[:3]):
    print(f"  [{i}] {t[:120]}")

# %%
# Onomasticon name repair
def load_onomasticon(path):
    if path is None:
        return {}
    df = pd.read_csv(path)
    # Try common column names
    col = df.columns[0]
    names = df[col].dropna().str.strip().unique().tolist()
    return {n.lower(): n for n in names if isinstance(n, str) and len(n) > 1}

def repair_names(translation, onomasticon):
    if not onomasticon or not translation:
        return translation
    words = translation.split()
    result = []
    for word in words:
        if not word or not word[0].isupper():
            result.append(word)
            continue
        stripped = word.rstrip('.,;:!?"\')]')
        suffix = word[len(stripped):]
        low = stripped.lower()
        if low in onomasticon:
            result.append(onomasticon[low] + suffix)
            continue
        matches = difflib.get_close_matches(low, onomasticon.keys(), n=1, cutoff=0.85)
        if matches:
            result.append(onomasticon[matches[0]] + suffix)
        else:
            result.append(word)
    return ' '.join(result)

onomasticon = load_onomasticon(ONOMASTICON_PATH)
print(f"Onomasticon loaded: {len(onomasticon)} names")
if onomasticon:
    sample_items = dict(list(onomasticon.items())[:5])
    print(f"  Sample: {sample_items}")

translations = [repair_names(t, onomasticon) for t in translations]
print("Name repair done")
for i, t in enumerate(translations[:3]):
    print(f"  [{i}] {t[:120]}")

# %%
# Document consistency
def enforce_consistency(translations, text_ids):
    doc_groups = defaultdict(list)
    for i, tid in enumerate(text_ids):
        doc_groups[tid].append(i)
    
    for tid, indices in doc_groups.items():
        if len(indices) <= 1:
            continue
        name_counts = Counter()
        for idx in indices:
            for w in translations[idx].split():
                if w and w[0].isupper() and len(w) > 2:
                    name_counts[w] += 1
        canonical = {}
        used = set()
        for name in sorted(name_counts, key=name_counts.get, reverse=True):
            if name.lower() in used:
                continue
            for other in name_counts:
                if other.lower() != name.lower() and other.lower() not in used:
                    if difflib.SequenceMatcher(None, name.lower(), other.lower()).ratio() > 0.8:
                        canonical[other] = name
                        used.add(other.lower())
            used.add(name.lower())
        for idx in indices:
            for wrong, right in canonical.items():
                # Word boundary matching to avoid partial replacement (e.g. "Aššur" inside "Puzur-Aššur")
                translations[idx] = re.sub(r'\b' + re.escape(wrong) + r'\b', right, translations[idx])
    
    return translations

text_ids = test_df["text_id"].tolist()
translations = enforce_consistency(translations, text_ids)
print("Document consistency enforced")
for i, t in enumerate(translations[:3]):
    print(f"  [{i}] {t[:120]}")

# %%
# Final validation and save submission
# Replace any empty translations
for i, t in enumerate(translations):
    if not t or t.strip() == "":
        translations[i] = "<gap>"
        print(f"  WARNING: Empty translation at index {i}, replaced with <gap>")

submission = pd.DataFrame({
    "id": test_df["id"],
    "translation": translations,
})

# Validate
assert len(submission) == len(test_df), f"Row count mismatch: {len(submission)} vs {len(test_df)}"
assert submission["translation"].notna().all(), "NaN values found!"
assert (submission["translation"].str.len() > 0).all(), "Empty translations found!"
print("Validation passed")

output_path = KAGGLE_WORKING / "submission.csv"
submission.to_csv(output_path, index=False)
print(f"\nSubmission saved to {output_path}")
print(f"  Rows: {len(submission)}")
print(submission)
