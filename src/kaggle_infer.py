#%% [markdown]
# Kaggle Inference Template (use trained MPS model)
#
# Goal: load a trained checkpoint from Kaggle Dataset and create submission.csv
#
# Steps (Kaggle UI):
# 1) Upload your trained model folder as a Kaggle Dataset
# 2) Add it as an Input to this notebook
# 3) Run all cells → /kaggle/working/submission.csv
#
# If your model dataset name is, e.g., "akkadian-byt5-mps", set MODEL_DATASET_NAME below.

#%%
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, ByT5Tokenizer

#%% [markdown]
# ## 1. Paths & Settings

#%%
KAGGLE_INPUT = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")

# Set this to your Kaggle Dataset name that contains the trained model
MODEL_DATASET_NAME = None  # e.g., "akkadian-byt5-mps"
TOKENIZER_DIR: Optional[Path] = None  # set if you uploaded tokenizer separately

# If None, we try to auto-detect a checkpoint inside /kaggle/input
MODEL_DIR: Optional[Path] = None

MAX_SRC_LEN = 256
MAX_TGT_LEN = 256
BATCH_SIZE = 8

#%% [markdown]
# ## 2. Locate competition data

#%%

def find_competition_data_dir() -> Path:
    if not KAGGLE_INPUT.exists():
        raise FileNotFoundError("/kaggle/input not found")
    for d in KAGGLE_INPUT.iterdir():
        if (d / "test.csv").exists() and (d / "sample_submission.csv").exists():
            return d
    raise FileNotFoundError("Could not locate competition dataset")


def find_model_dir() -> Path:
    if MODEL_DATASET_NAME:
        d = KAGGLE_INPUT / MODEL_DATASET_NAME
        if not d.exists():
            raise FileNotFoundError(f"Model dataset not found: {d}")
        return d

    # Auto-detect: look for config.json + model weights
    for d in KAGGLE_INPUT.iterdir():
        if not d.is_dir():
            continue
        # check root
        if (d / "config.json").exists():
            return d
        # check subdirs
        for sub in d.glob("**/config.json"):
            return sub.parent

    raise FileNotFoundError("Could not auto-detect model directory in /kaggle/input")


COMP_DATA_DIR = find_competition_data_dir()
MODEL_DIR = MODEL_DIR or find_model_dir()

print("Competition data:", COMP_DATA_DIR)
print("Model dir:", MODEL_DIR)

#%% [markdown]
# ## 3. Normalization (match preprocessing)

#%%
_SUBSCRIPT_MAP = str.maketrans({
    "\u2080": "0",
    "\u2081": "1",
    "\u2082": "2",
    "\u2083": "3",
    "\u2084": "4",
    "\u2085": "5",
    "\u2086": "6",
    "\u2087": "7",
    "\u2088": "8",
    "\u2089": "9",
    "\u2093": "x",
})


def normalize_transliteration(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u1E2A", "H").replace("\u1E2B", "h")
    text = text.translate(_SUBSCRIPT_MAP)
    text = text.replace("\u2026", " <big_gap> ")
    text = re.sub(r"\.\.\.+", " <big_gap> ", text)
    text = re.sub(r"\[([^\]]+)\]", " <gap> ", text)
    text = re.sub(r"\bx\b", " <unk_sign> ", text)
    text = re.sub(r"[!?/]", " ", text)
    text = re.sub(r"<([^>]+)>", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()

#%% [markdown]
# ## 4. Load model + tokenizer

#%%
# Prefer tokenizer from model/tokenizer dir if present; fallback to ByT5 class.
try:
    if TOKENIZER_DIR and TOKENIZER_DIR.exists():
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    elif (MODEL_DIR / "tokenizer_config.json").exists() or (MODEL_DIR / "tokenizer.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    else:
        # If no tokenizer files, use ByT5 tokenizer with defaults (no internet needed)
        tokenizer = ByT5Tokenizer()
except Exception as exc:
    print(f"Tokenizer load failed ({exc}); falling back to ByT5Tokenizer().")
    tokenizer = ByT5Tokenizer()

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if device.type == "cuda":
    model = model.half()

model.eval()

#%% [markdown]
# ## 5. Predict on test and create submission

#%%

def batch_generate(texts, batch_size=BATCH_SIZE, max_len=MAX_TGT_LEN):
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SRC_LEN,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_length=max_len)
        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs


test_df = pd.read_csv(COMP_DATA_DIR / "test.csv")
assert {"id", "transliteration"}.issubset(test_df.columns)

norm_texts = [normalize_transliteration(t) for t in test_df["transliteration"].tolist()]

preds = batch_generate(norm_texts)

sub = pd.DataFrame({"id": test_df["id"], "translation": preds})
sub_path = WORK_DIR / "submission.csv"
sub.to_csv(sub_path, index=False)

print("Saved:", sub_path)

#%% [markdown]
# ## 6. Quick sanity check
#
# ```
# !head /kaggle/working/submission.csv
# ```
