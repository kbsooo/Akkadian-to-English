#%% [markdown]
# Kaggle Notebook Template: Akkadian → English (Baseline)
#
# Pipeline: data load → preprocessing (if needed) → train → eval → submission
#
# Notes
# - Works on Kaggle with 1 or 2 GPUs (T4).
# - Uses ByT5-base by default (safe for T4).
# - Multi-GPU uses accelerate.notebook_launcher.

#%%
from __future__ import annotations

import os
import math
import json
import random
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sacrebleu.metrics import BLEU, CHRF
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
import inspect

# Optional: accelerate for multi-GPU
try:
    from accelerate import notebook_launcher
    _HAS_ACCELERATE = True
except Exception:
    _HAS_ACCELERATE = False

#%% [markdown]
# ## 0. (Optional) Install dependencies
#
# If needed in Kaggle:
# ```
# !pip -q install transformers datasets sacrebleu sentencepiece accelerate
# ```

#%% [markdown]
# ## 1. Paths & Settings
#
# Kaggle folder convention:
# - inputs: /kaggle/input/<dataset-name>/
# - outputs: /kaggle/working/

#%%
KAGGLE_INPUT = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")
OUTPUTS_DIR = WORK_DIR / "outputs"

MODEL_NAME = "google/byt5-base"  # safe default for T4
TIER_FILE = "sentence_pairs_q70_pattern.csv"  # Tier3
SEED = 42

# Training hyperparams (tune as needed)
MAX_SRC_LEN = 256
MAX_TGT_LEN = 256
BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS = 5
LR = 3e-4
WARMUP_RATIO = 0.05

USE_FP16 = True if torch.cuda.is_available() else False
GRADIENT_CHECKPOINTING = False

#%% [markdown]
# ## 2. Locate competition data

#%%

def find_competition_data_dir() -> Path:
    # find directory containing train.csv and test.csv
    if not KAGGLE_INPUT.exists():
        raise FileNotFoundError("/kaggle/input not found")

    for d in KAGGLE_INPUT.iterdir():
        if not d.is_dir():
            continue
        if (d / "train.csv").exists() and (d / "test.csv").exists():
            return d
    raise FileNotFoundError("Could not locate competition dataset with train.csv/test.csv")


def find_repo_dir() -> Path | None:
    # search for repo that contains src/data_preprocessing.py
    for d in KAGGLE_INPUT.iterdir():
        if not d.is_dir():
            continue
        cand = d / "src" / "data_preprocessing.py"
        if cand.exists():
            return d
    return None


COMP_DATA_DIR = find_competition_data_dir()
REPO_DIR = find_repo_dir()

print("Competition data:", COMP_DATA_DIR)
print("Repo dir:", REPO_DIR)

#%% [markdown]
# ## 3. Preprocessing (if needed)
#
# This expects our preprocessing script to be attached as a Kaggle Dataset.
# If the outputs already exist, it will skip.

#%%

def run_preprocessing():
    if OUTPUTS_DIR.exists() and (OUTPUTS_DIR / TIER_FILE).exists():
        print("Found preprocessed outputs.")
        return

    if REPO_DIR is None:
        raise RuntimeError("Repo with src/data_preprocessing.py not found in /kaggle/input")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    script = REPO_DIR / "src" / "data_preprocessing.py"
    cmd = [
        "python",
        str(script),
        "--data-dir",
        str(COMP_DATA_DIR),
        "--out-dir",
        str(OUTPUTS_DIR),
        "--plot",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


run_preprocessing()

#%% [markdown]
# ## 4. Load Tier3 dataset

#%%
train_df = pd.read_csv(OUTPUTS_DIR / TIER_FILE)
assert {"oare_id", "src_norm", "tgt_norm"}.issubset(train_df.columns)

print("Train rows:", len(train_df))
print(train_df.head(2))

#%% [markdown]
# ## 5. Grouped train/val split

#%%

def group_split(df: pd.DataFrame, group_col: str, val_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    groups = df[group_col].unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * val_frac))
    val_groups = set(groups[:n_val])

    train_df = df[~df[group_col].isin(val_groups)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_groups)].reset_index(drop=True)
    return train_df, val_df


train_df, val_df = group_split(train_df, "oare_id", 0.1, SEED)
print("Train/Val:", len(train_df), len(val_df))

#%% [markdown]
# ## 6. Tokenization

#%%
set_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
if GRADIENT_CHECKPOINTING:
    model.gradient_checkpointing_enable()

train_ds = Dataset.from_pandas(train_df[["src_norm", "tgt_norm"]])
val_ds = Dataset.from_pandas(val_df[["src_norm", "tgt_norm"]])


def tokenize(batch):
    inputs = tokenizer(
        batch["src_norm"],
        max_length=MAX_SRC_LEN,
        truncation=True,
    )
    labels = tokenizer(
        text_target=batch["tgt_norm"],
        max_length=MAX_TGT_LEN,
        truncation=True,
    )
    inputs["labels"] = labels["input_ids"]
    return inputs


train_ds = train_ds.map(tokenize, batched=True, remove_columns=["src_norm", "tgt_norm"])
val_ds = val_ds.map(tokenize, batched=True, remove_columns=["src_norm", "tgt_norm"])

#%% [markdown]
# ## 7. Metrics

#%%
bleu = BLEU()
chrf = CHRF(word_order=2)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds == -100, tokenizer.pad_token_id, preds)
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu_score = bleu.corpus_score(decoded_preds, [decoded_labels]).score
    chrf_score = chrf.corpus_score(decoded_preds, [decoded_labels]).score
    score = math.sqrt(max(0.0, bleu_score) * max(0.0, chrf_score))

    return {"bleu": bleu_score, "chrf": chrf_score, "score": score}

#%% [markdown]
# ## 8. Train (single or multi‑GPU)

#%%
@dataclass
class TrainCfg:
    batch_size: int = BATCH_SIZE
    grad_accum: int = GRAD_ACCUM
    epochs: int = EPOCHS
    lr: float = LR
    warmup_ratio: float = WARMUP_RATIO


def train_fn():
    set_seed(SEED)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    steps_per_epoch = math.ceil(len(train_ds) / max(1, BATCH_SIZE))
    steps_per_epoch = math.ceil(steps_per_epoch / max(1, GRAD_ACCUM))
    total_steps = max(1, steps_per_epoch * EPOCHS)
    warmup_steps = int(total_steps * WARMUP_RATIO)

    # HF arg compatibility (evaluation_strategy -> eval_strategy)
    arg_sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    eval_key = "evaluation_strategy" if "evaluation_strategy" in arg_sig.parameters else "eval_strategy"

    args_kwargs = dict(
        output_dir=str(WORK_DIR / "baseline"),
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        predict_with_generate=True,
        logging_steps=20,
        save_total_limit=2,
        fp16=USE_FP16,
        bf16=False,
        load_best_model_at_end=True,
        metric_for_best_model="score",
        greater_is_better=True,
        report_to="none",
    )
    args_kwargs[eval_key] = "epoch"

    args = Seq2SeqTrainingArguments(**args_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainer.train()
    metrics = trainer.evaluate()

    with open(WORK_DIR / "baseline" / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(metrics)


if torch.cuda.device_count() > 1 and _HAS_ACCELERATE:
    print("Multi-GPU detected. Using accelerate.notebook_launcher ...")
    notebook_launcher(train_fn, num_processes=torch.cuda.device_count())
else:
    train_fn()

#%% [markdown]
# ## 9. Predict on Test + Create Submission

#%%
# Minimal normalization for test side (matches preprocessing)
import unicodedata
import re


def normalize_transliteration(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u1E2A", "H").replace("\u1E2B", "h")
    text = re.sub(r"\.\.\.+", " <big_gap> ", text)
    text = text.replace("\u2026", " <big_gap> ")
    text = re.sub(r"\[([^\]]+)\]", " <gap> ", text)
    text = re.sub(r"\bx\b", " <unk_sign> ", text)
    return re.sub(r"\s+", " ", text).strip()


def batch_generate(texts, batch_size=8, max_len=MAX_TGT_LEN):
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SRC_LEN)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_length=max_len)
        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs


test_df = pd.read_csv(COMP_DATA_DIR / "test.csv")
assert {"id", "transliteration"}.issubset(test_df.columns)

norm_texts = [normalize_transliteration(t) for t in test_df["transliteration"].tolist()]

preds = batch_generate(norm_texts, batch_size=8)

sub = pd.DataFrame({"id": test_df["id"], "translation": preds})
sub_path = WORK_DIR / "submission.csv"
sub.to_csv(sub_path, index=False)

print("Saved:", sub_path)
