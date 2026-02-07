# %% [markdown]
# # V7 Training — ByT5-small Akkadian→English (Kaggle T4×2)
#
# **Environment:** Kaggle Notebook, GPU T4 × 2, 9h limit
# **Data:** `/kaggle/input/data-v7/` (pre-built V7 dataset)
# **Output:** `/kaggle/working/outputs_v7/final/` → upload as `akkadian-v7-model`
#
# ### Multi-GPU Strategy
# HuggingFace Trainer uses DataParallel automatically when 2+ GPUs detected.
# - `per_device_train_batch_size` applies **per GPU**
# - Effective batch = per_device × n_gpus × grad_accum = 8 × 2 × 1 = 16
# - 2× throughput vs single-GPU with identical training dynamics

# %% [markdown]
# ## Configuration

# %%
MODEL_NAME = "google/byt5-small"  # ~300M params (d_model=1472, 12+4 layers)
DATA_DIR = "/kaggle/input/data-v7"
OUTPUT_DIR = "/kaggle/working/outputs_v7"

MAX_SOURCE_LENGTH = 384
MAX_TARGET_LENGTH = 384
EPOCHS = 15
BATCH_SIZE = 8       # per GPU — with gradient checkpointing, fits in T4 16GB
GRAD_ACCUM = 1       # effective batch = 8 × 2 GPUs × 1 = 16
LR = 5e-5            # lower LR prevents overfitting on ~8K data
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.005
EARLY_STOPPING_PATIENCE = 5  # cosine LR converges slowly, needs patience
LABEL_SMOOTHING = 0.1
SEED = 42

# %% [markdown]
# ## Install & Import

# %%
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "sacrebleu", "accelerate"])

import os
# %%
# Keep transformers on PyTorch path only, and reduce TF/XLA startup noise.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
import numpy as np
import pandas as pd
import json
import re
import unicodedata
from pathlib import Path
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed,
)
from datasets import Dataset
from sacrebleu.metrics import BLEU, CHRF

set_seed(SEED)

# %% [markdown]
# ## GPU Diagnostics

# %%
n_gpus = torch.cuda.device_count()
print(f"GPUs: {n_gpus}")
for i in range(n_gpus):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  [{i}] {name} — {mem:.1f} GB")

effective_batch = BATCH_SIZE * n_gpus * GRAD_ACCUM
print(f"\nTraining config:")
print(f"  per_device_batch = {BATCH_SIZE}")
print(f"  n_gpus = {n_gpus}")
print(f"  grad_accum = {GRAD_ACCUM}")
print(f"  → effective batch = {effective_batch}")
print(f"  epochs = {EPOCHS}, LR = {LR}")
print(f"  label_smoothing = {LABEL_SMOOTHING}")

# %% [markdown]
# ## Load Data

# %%
train_df = pd.read_csv(f"{DATA_DIR}/v7_train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/v7_val.csv")
train_df = train_df.dropna(subset=["src", "tgt"]).reset_index(drop=True)
val_df = val_df.dropna(subset=["src", "tgt"]).reset_index(drop=True)

print(f"Train: {len(train_df):,}")
print(f"Val:   {len(val_df):,}")
print(f"\nSource breakdown:")
print(train_df["source"].value_counts().to_string())
print(f"\nSample:")
print(f"  src: {train_df['src'].iloc[0][:100]}...")
print(f"  tgt: {train_df['tgt'].iloc[0][:100]}...")

# Check length distribution
pct_over = (train_df["src"].str.len() > MAX_SOURCE_LENGTH).mean() * 100
print(f"\nSources > {MAX_SOURCE_LENGTH} chars: {pct_over:.1f}%")

# %% [markdown]
# ## Load Model & Tokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {MODEL_NAME}")
print(f"Parameters: {n_params:,}")

# %% [markdown]
# ## Tokenize

# %%
def tokenize_fn(examples):
    # text_target ensures decoder-side tokenization with proper BOS token handling
    model_inputs = tokenizer(
        examples["src"], max_length=MAX_SOURCE_LENGTH,
        truncation=True, padding=False,
        text_target=examples["tgt"])
    labels = model_inputs["labels"]
    model_inputs["labels"] = [l[:MAX_TARGET_LENGTH] for l in labels]
    return model_inputs

train_ds = Dataset.from_pandas(train_df[["src", "tgt"]])
train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"],
                        num_proc=2)  # parallel tokenization on Kaggle's 4-core CPU
val_ds = Dataset.from_pandas(val_df[["src", "tgt"]])
val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"],
                    num_proc=2)

print(f"Tokenized — Train: {len(train_ds)}, Val: {len(val_ds)}")
print(f"Sample input_ids len: {len(train_ds[0]['input_ids'])}")

# %% [markdown]
# ## Metrics

# %%
bleu_metric = BLEU()
chrf_metric = CHRF(word_order=2)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = [p.strip() for p in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
    decoded_labels = [[l.strip()] for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
    bleu = bleu_metric.corpus_score(decoded_preds, decoded_labels).score
    chrf = chrf_metric.corpus_score(decoded_preds, decoded_labels).score
    geo = (bleu * chrf) ** 0.5 if bleu > 0 and chrf > 0 else 0.0
    return {"bleu": bleu, "chrf": chrf, "geo_mean": geo}

# %% [markdown]
# ## Callbacks

# %%
class LogCallback(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.losses:
            avg = sum(self.losses) / len(self.losses)
            print(f"\n--- Epoch {int(state.epoch)} avg train loss: {avg:.4f} ---")
            self.losses = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"  BLEU: {metrics.get('eval_bleu', 0):.2f}  "
                  f"chrF: {metrics.get('eval_chrf', 0):.2f}  "
                  f"Geo: {metrics.get('eval_geo_mean', 0):.2f}")


class SampleCallback(TrainerCallback):
    """Generate sample translations on GPU 0 after each eval."""
    def __init__(self, tokenizer, samples):
        self.tokenizer = tokenizer
        self.samples = samples[:3]

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        # When Trainer wraps in DataParallel, unwrap to get the actual model
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.eval()
        device = next(raw_model.parameters()).device
        print("\nSample translations:")
        for i, src in enumerate(self.samples):
            inputs = self.tokenizer(src, return_tensors="pt", truncation=True,
                                    max_length=MAX_SOURCE_LENGTH)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = raw_model.generate(**inputs, max_length=128, num_beams=4)
            trans = self.tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"  [{i}] {src[:60]}...")
            print(f"      → {trans[:80]}")

# %% [markdown]
# ## Training Arguments (T4×2 Optimized)

# %%
os.makedirs(OUTPUT_DIR, exist_ok=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=f"{OUTPUT_DIR}/checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,  # eval uses less memory (no gradients)
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    label_smoothing_factor=LABEL_SMOOTHING,
    max_grad_norm=1.0,
    # ByT5 requires FP32 — FP16 causes NaN losses due to byte-level embeddings
    fp16=False,
    bf16=False,
    # Gradient checkpointing: trades ~20% speed for ~30% VRAM savings
    # Critical for batch=8 on T4 with ByT5's 384-token byte sequences
    gradient_checkpointing=True,
    # Eval & save
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_geo_mean",
    greater_is_better=True,
    # Generation during eval
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    generation_num_beams=4,
    # Kaggle has 4 CPU cores — use 2 workers per GPU for data loading
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    # Misc
    logging_steps=50,
    report_to="none",
    seed=SEED,
    # Multi-GPU: Trainer auto-detects DataParallel when CUDA_VISIBLE_DEVICES has 2+ GPUs
    # No ddp_* args needed for Kaggle's simple 2-GPU setup
)

print("Training args configured")
print(f"  Effective batch: {BATCH_SIZE} × {n_gpus} GPU × {GRAD_ACCUM} accum = {effective_batch}")
print(f"  Gradient checkpointing: ON")
print(f"  LR: {LR} (cosine → 0)")

# %% [markdown]
# ## Train

# %%
callbacks = [
    LogCallback(),
    SampleCallback(tokenizer, val_df["src"].tolist()),
    EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
]

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
)

print(f"Starting training on {n_gpus} GPU(s)...")
print(f"  Train samples: {len(train_ds):,}")
print(f"  Steps/epoch: ~{len(train_ds) // effective_batch}")
print(f"  Max epochs: {EPOCHS} (early stop patience={EARLY_STOPPING_PATIENCE})")

trainer.train()

# %% [markdown]
# ## Evaluate

# %%
results = trainer.evaluate()
print(f"\nFinal evaluation:")
print(f"  BLEU: {results.get('eval_bleu', 0):.2f}")
print(f"  chrF: {results.get('eval_chrf', 0):.2f}")
print(f"  Geo:  {results.get('eval_geo_mean', 0):.2f}")

# %% [markdown]
# ## Save Model

# %%
final_dir = f"{OUTPUT_DIR}/final"
os.makedirs(final_dir, exist_ok=True)

# Unwrap DataParallel if needed
save_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
save_model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

# Save training config for reproducibility
config_info = {
    "model_name": MODEL_NAME,
    "max_source_length": MAX_SOURCE_LENGTH,
    "max_target_length": MAX_TARGET_LENGTH,
    "epochs_trained": int(trainer.state.epoch),
    "effective_batch_size": effective_batch,
    "learning_rate": LR,
    "label_smoothing": LABEL_SMOOTHING,
    "best_metric": results.get("eval_geo_mean", 0),
    "normalization": "v7_preserve_diacritics",
    "n_gpus": n_gpus,
}
with open(f"{final_dir}/v7_config.json", "w") as f:
    json.dump(config_info, f, indent=2)

print(f"\nModel saved to {final_dir}/")
for fname in sorted(os.listdir(final_dir)):
    size = os.path.getsize(f"{final_dir}/{fname}") / 1e6
    print(f"  {fname} ({size:.1f} MB)")

# %% [markdown]
# ## Sanity Check

# %%
# Quick generation test on the saved model
save_model.eval()
device0 = torch.device("cuda:0")
save_model = save_model.to(device0)

test_input = "um-ma ka-ru-um"
inputs = tokenizer(test_input, return_tensors="pt").to(device0)
with torch.no_grad():
    out = save_model.generate(**inputs, max_length=50, num_beams=4)
translation = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"\nSanity check:")
print(f"  Input:  '{test_input}'")
print(f"  Output: '{translation}'")
assert translation.strip() != "", "Empty output!"
print("  OK")

# %% [markdown]
# ## Verify Output Files
#
# After training, create a **New Dataset** on Kaggle:
# 1. Go to kaggle.com/datasets/new
# 2. Upload all files from `/kaggle/working/outputs_v7/final/`
# 3. Name it `akkadian-v7-model`
# 4. Use as input in the inference notebook

# %%
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"  Best geo_mean: {results.get('eval_geo_mean', 0):.2f}")
print(f"  BLEU: {results.get('eval_bleu', 0):.2f}")
print(f"  chrF: {results.get('eval_chrf', 0):.2f}")
print(f"  Epochs: {int(trainer.state.epoch)}")
print(f"  GPUs used: {n_gpus}")
print(f"\n  Output: {final_dir}/")
print(f"  Files:")
for fname in sorted(os.listdir(final_dir)):
    print(f"    - {fname}")
print(f"\n  → Upload '{final_dir}/' as Kaggle Dataset 'akkadian-v7-model'")
print("=" * 60)
