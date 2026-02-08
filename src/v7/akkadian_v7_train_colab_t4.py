# %% [markdown]
# # V7 Training — ByT5-small Akkadian→English (Colab T4 16GB)
#
# **Environment:** Google Colab Free/Pro, single T4 16GB
#
# ### T4 Optimization Strategy
# | Constraint | Solution |
# |------------|----------|
# | 16GB VRAM | Gradient checkpointing ON (~30% VRAM savings) |
# | No BF16 | FP32 only — ByT5 NaN with FP16, T4 lacks BF16 |
# | Slower matmul | Batch=8 + grad_accum=2 → effective=16 |
# | Slow eval | Greedy decoding + short gen length (eval ≠ final inference) |
# | Colab timeout | Early stopping patience=5, ~4-5h total |

# %% [markdown]
# ## Configuration

# %%
KAGGLE_USERNAME = "your-username"  # EDIT THIS
V7_DATA_DATASET = f"{KAGGLE_USERNAME}/akkadian-v7-data"
MODEL_NAME = "google/byt5-small"  # ~300M params (d_model=1472, 12+4 layers)
OUTPUT_DIR = "outputs_v7"

MAX_SOURCE_LENGTH = 384
MAX_TARGET_LENGTH = 384
EPOCHS = 15
BATCH_SIZE = 8        # T4 16GB + FP32 + gradient checkpointing → batch=8 safe
GRAD_ACCUM = 2        # effective batch = 8 × 2 = 16
LR = 5e-5             # standard for effective batch=16
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.005
EARLY_STOPPING_PATIENCE = 5
LABEL_SMOOTHING = 0.1
SEED = 42

print("Configuration (Colab T4 16GB):")
print(f"  Model: {MODEL_NAME}")
print(f"  Max length: {MAX_SOURCE_LENGTH}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch: {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM} effective")
print(f"  LR: {LR}")
print(f"  Label smoothing: {LABEL_SMOOTHING}")
print(f"  Seed: {SEED}")

# %% [markdown]
# ## Install & Import

# %%
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "kagglehub", "transformers", "datasets", "sacrebleu", "accelerate"])
print("Dependencies installed")

# %%
import torch
import numpy as np
import pandas as pd
import json
import os
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gpu_mem:.1f} GB")

# %% [markdown]
# ## Download V7 Data

# %%
import kagglehub

try:
    data_path = kagglehub.dataset_download(V7_DATA_DATASET)
    print(f"Data from Kaggle: {data_path}")
except Exception as e:
    print(f"Kaggle download failed: {e}")
    print("Trying Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive')
    data_path = "/content/drive/MyDrive/akkadian/data_v7"

print("\nData files:")
for f in sorted(os.listdir(data_path)):
    print(f"  {f}")

# %% [markdown]
# ## Load Data

# %%
train_df = pd.read_csv(f"{data_path}/v7_train.csv")
val_df = pd.read_csv(f"{data_path}/v7_val.csv")
train_df = train_df.dropna(subset=["src", "tgt"]).reset_index(drop=True)
val_df = val_df.dropna(subset=["src", "tgt"]).reset_index(drop=True)

print(f"Train: {len(train_df):,}")
print(f"Val:   {len(val_df):,}")
print(f"\nSource breakdown:")
print(train_df["source"].value_counts().to_string())
print(f"\nSample:")
print(f"  src: {train_df['src'].iloc[0][:100]}...")
print(f"  tgt: {train_df['tgt'].iloc[0][:100]}...")

pct_over = (train_df["src"].str.len() > MAX_SOURCE_LENGTH).mean() * 100
print(f"\nSources > {MAX_SOURCE_LENGTH} chars: {pct_over:.1f}%")

# %% [markdown]
# ## V7 Normalization

# %%
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
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("<gap>", "\x00GAP\x00")
    text = text.replace("<big_gap>", "\x00BIGGAP\x00")
    text = re.sub(r"\b\d+'{1,2}\b", " ", text)
    text = re.sub(r"<<([^>]+)>>", r"\1", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)
    text = re.sub(r"\[\s*…+\s*…*\s*\]", " \x00BIGGAP\x00 ", text)
    text = re.sub(r"\[\s*\.\.\.+\s*\]", " \x00BIGGAP\x00 ", text)
    text = text.replace("…", " \x00BIGGAP\x00 ")
    text = re.sub(r"\.\.\.+", " \x00BIGGAP\x00 ", text)
    text = re.sub(r"\[\s*x\s*\]", " \x00GAP\x00 ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)
    for c in "‹›⌈⌉⌊⌋˹˺":
        text = text.replace(c, "")
    text = text.translate(_V7_TRANS_TABLE)
    text = re.sub(r"[!?/]", " ", text)
    text = re.sub(r"\s*:\s*", " ", text)
    text = re.sub(r"(?<![a-zA-Z\x00])\bx\b(?![a-zA-Z])", " \x00GAP\x00 ", text)
    text = text.replace("\x00GAP\x00", "<gap>")
    text = text.replace("\x00BIGGAP\x00", "<big_gap>")
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("V7 normalization defined (preserves š, ṣ, ṭ, vowel accents)")

# %% [markdown]
# ## Load Model & Tokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# FP32: ByT5 byte embeddings cause NaN under FP16 (5-bit exponent, max 65504)
# T4 lacks BF16 support, so FP32 is the only safe option
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {MODEL_NAME}")
print(f"Parameters: {n_params:,}")
print(f"Dtype: {next(model.parameters()).dtype}")

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
                        num_proc=2)
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
    """Generate sample translations after each eval."""
    def __init__(self, tokenizer, samples):
        self.tokenizer = tokenizer
        self.samples = samples[:3]

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model.eval()
        device = next(model.parameters()).device
        print("\nSample translations:")
        for i, src in enumerate(self.samples):
            inputs = self.tokenizer(src, return_tensors="pt", truncation=True,
                                    max_length=MAX_SOURCE_LENGTH)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_length=128, num_beams=1)
            trans = self.tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"  [{i}] {src[:60]}...")
            print(f"      -> {trans[:80]}")

# %% [markdown]
# ## Training Arguments (T4 16GB Optimized)

# %%
os.makedirs(OUTPUT_DIR, exist_ok=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# Steps per epoch (for eval scheduling)
steps_per_epoch = len(train_ds) // (BATCH_SIZE * GRAD_ACCUM)
# Eval every 2 epochs to minimize ByT5's slow autoregressive eval generation
eval_interval = steps_per_epoch * 2

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
    # ByT5 requires FP32 — FP16 causes NaN, T4 lacks BF16
    fp16=False,
    bf16=False,
    # Gradient checkpointing: trades ~20% speed for ~30% VRAM savings
    # Essential for batch=8 on T4 16GB with ByT5's long byte sequences
    gradient_checkpointing=True,
    # Eval every ~2 epochs: ByT5 byte-level generation is the dominant time cost
    # (up to 384 autoregressive steps per sample). Training forward/backward is fast;
    # eval generation is memory-bandwidth-bound and barely benefits from faster GPUs.
    eval_strategy="steps",
    eval_steps=eval_interval,
    save_strategy="steps",
    save_steps=eval_interval,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_geo_mean",
    greater_is_better=True,
    # Generation during eval — greedy + short length to keep eval fast
    # Final inference uses beam search separately; eval only needs metric tracking
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=1,
    # Colab has 2 CPU cores — 2 workers for data loading
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    logging_steps=50,
    report_to="none",
    seed=SEED,
)

print("Training args configured (T4 16GB)")
print(f"  Effective batch: {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Gradient checkpointing: ON")
print(f"  Precision: FP32")
print(f"  Eval every: ~2 epochs ({eval_interval} steps)")
print(f"  Eval generation: greedy, max_length=128")
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

print(f"Starting training on T4 16GB...")
print(f"  Train samples: {len(train_ds):,}")
print(f"  Steps/epoch: ~{steps_per_epoch}")
print(f"  Max epochs: {EPOCHS} (early stop patience={EARLY_STOPPING_PATIENCE})")
print(f"  Estimated time: ~4-5 hours")

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

trainer.model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

config_info = {
    "model_name": MODEL_NAME,
    "max_source_length": MAX_SOURCE_LENGTH,
    "max_target_length": MAX_TARGET_LENGTH,
    "epochs_trained": int(trainer.state.epoch),
    "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
    "learning_rate": LR,
    "label_smoothing": LABEL_SMOOTHING,
    "normalization": "v7_preserve_diacritics",
    "gpu": "T4 16GB",
    "precision": "FP32",
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
model = trainer.model
model.eval()
device0 = next(model.parameters()).device

test_input = "um-ma ka-ru-um"
inputs = tokenizer(test_input, return_tensors="pt").to(device0)
with torch.no_grad():
    out = model.generate(**inputs, max_length=50, num_beams=4)
translation = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"\nSanity check:")
print(f"  Input:  '{test_input}'")
print(f"  Output: '{translation}'")
assert translation.strip() != "", "Empty output!"
print("  OK")

# %% [markdown]
# ## Done
#
# Upload `outputs_v7/final/` to Kaggle as Dataset `akkadian-v7-model`.
#
# Required files:
# - config.json, model.safetensors
# - tokenizer.json, tokenizer_config.json, special_tokens_map.json
# - v7_config.json

# %%
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"  Best geo_mean: {results.get('eval_geo_mean', 0):.2f}")
print(f"  BLEU: {results.get('eval_bleu', 0):.2f}")
print(f"  chrF: {results.get('eval_chrf', 0):.2f}")
print(f"  Epochs: {int(trainer.state.epoch)}")
print(f"  GPU: T4 16GB (FP32)")
print(f"\n  Output: {final_dir}/")
print(f"  -> Upload to Kaggle Dataset 'akkadian-v7-model'")
print("=" * 60)
