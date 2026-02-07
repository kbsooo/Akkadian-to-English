# %% [markdown]
# # V7 Training — ByT5-small Akkadian→English

# %% [markdown]
# Configuration for V7 training. Edit KAGGLE_USERNAME to your username.

# %%
KAGGLE_USERNAME = "your-username"  # EDIT THIS
V7_DATA_DATASET = f"{KAGGLE_USERNAME}/akkadian-v7-data"
MODEL_NAME = "google/byt5-small"
OUTPUT_DIR = "outputs_v7"

MAX_SOURCE_LENGTH = 384
MAX_TARGET_LENGTH = 384
EPOCHS = 15
BATCH_SIZE = 4
GRAD_ACCUM = 4  # effective batch = 16
LR = 1e-4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 3
SEED = 42

print("Configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Max source length: {MAX_SOURCE_LENGTH}")
print(f"  Max target length: {MAX_TARGET_LENGTH}")
print(f"  Epochs: {EPOCHS}")
print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Learning rate: {LR}")
print(f"  Seed: {SEED}")

# %% [markdown]
# Install dependencies

# %%
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub", "transformers", "datasets", "sacrebleu", "accelerate"])
print("Dependencies installed")

# %% [markdown]
# Import libraries and setup device

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
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# Download V7 data from Kaggle

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
# Load and inspect data

# %%
train_df = pd.read_csv(f"{data_path}/v7_train.csv")
val_df = pd.read_csv(f"{data_path}/v7_val.csv")
train_df = train_df.dropna(subset=["src", "tgt"]).reset_index(drop=True)
val_df = val_df.dropna(subset=["src", "tgt"]).reset_index(drop=True)

print(f"Train: {len(train_df):,}")
print(f"Val: {len(val_df):,}")
print(f"\nSource breakdown (train):")
print(train_df["source"].value_counts())
print(f"\nSample src: {train_df['src'].iloc[0][:120]}")
print(f"Sample tgt: {train_df['tgt'].iloc[0][:120]}")

# %% [markdown]
# V7 Normalization (exact - must match build and infer scripts)

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

print("V7 normalization function defined")
print("  Preserves: š, ṣ, ṭ, vowel accents")
print("  Only removes: Ḫ→H, smart quotes, special punct")

# %% [markdown]
# Check data characteristics

# %%
print("Source length stats (chars):")
print(train_df["src"].str.len().describe())
print(f"\nSources > 384 chars: {(train_df['src'].str.len() > 384).sum()} ({(train_df['src'].str.len() > 384).mean()*100:.1f}%)")
print(f"\nTarget length stats (chars):")
print(train_df["tgt"].str.len().describe())

# %% [markdown]
# Load model and tokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {MODEL_NAME}")
print(f"Parameters: {n_params:,}")
print(f"Tokenizer vocab: {len(tokenizer)}")

# %% [markdown]
# Tokenize datasets

# %%
def tokenize_fn(examples):
    model_inputs = tokenizer(
        examples["src"], max_length=MAX_SOURCE_LENGTH,
        truncation=True, padding=False)
    labels = tokenizer(
        examples["tgt"], max_length=MAX_TARGET_LENGTH,
        truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_ds = Dataset.from_pandas(train_df[["src", "tgt"]])
train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"])
val_ds = Dataset.from_pandas(val_df[["src", "tgt"]])
val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"])

print(f"Train dataset: {len(train_ds)}")
print(f"Val dataset: {len(val_ds)}")
print(f"Train sample keys: {train_ds[0].keys()}")
print(f"Train sample input_ids length: {len(train_ds[0]['input_ids'])}")

# %% [markdown]
# Define metrics

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
    bleu_score = bleu_metric.corpus_score(decoded_preds, decoded_labels).score
    chrf_score = chrf_metric.corpus_score(decoded_preds, decoded_labels).score
    geo = (bleu_score * chrf_score) ** 0.5 if bleu_score > 0 and chrf_score > 0 else 0.0
    return {"bleu": bleu_score, "chrf": chrf_score, "geo_mean": geo}

print("Metrics defined: BLEU, chrF++ (word_order=2), geo_mean")

# %% [markdown]
# Define callbacks

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
            print(f"  BLEU: {metrics.get('eval_bleu', 0):.2f}")
            print(f"  chrF: {metrics.get('eval_chrf', 0):.2f}")
            print(f"  Geo:  {metrics.get('eval_geo_mean', 0):.2f}")

class SampleCallback(TrainerCallback):
    def __init__(self, tokenizer, samples, device):
        self.tokenizer = tokenizer
        self.samples = samples[:3]
        self.device = device
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model.eval()
        print("\nSample translations:")
        for i, src in enumerate(self.samples):
            inputs = self.tokenizer(src, return_tensors="pt", truncation=True, max_length=384)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_length=128, num_beams=4)
            trans = self.tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"  [{i}] {src[:60]}...")
            print(f"      -> {trans[:80]}")

print("Callbacks defined: LogCallback, SampleCallback")

# %% [markdown]
# Configure training arguments

# %%
os.makedirs(OUTPUT_DIR, exist_ok=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=f"{OUTPUT_DIR}/checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    max_grad_norm=1.0,
    fp16=False,  # ByT5 requires FP32
    bf16=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_geo_mean",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    dataloader_num_workers=2,
    logging_steps=50,
    report_to="none",
    seed=SEED,
)

print("Training arguments configured")
print(f"  Epochs: {EPOCHS}")
print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Max length: {MAX_SOURCE_LENGTH}")
print(f"  fp16: False (ByT5 requires FP32)")

# %% [markdown]
# Create trainer and start training

# %%
callbacks = [
    LogCallback(),
    SampleCallback(tokenizer, val_df["src"].tolist(), device),
    EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
]

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
)

print("Starting training...")
trainer.train()

# %% [markdown]
# Evaluate final model

# %%
results = trainer.evaluate()
print(f"\nFinal evaluation:")
print(f"  BLEU: {results.get('eval_bleu', 0):.2f}")
print(f"  chrF: {results.get('eval_chrf', 0):.2f}")
print(f"  Geo:  {results.get('eval_geo_mean', 0):.2f}")

# %% [markdown]
# Save model and tokenizer

# %%
final_dir = f"{OUTPUT_DIR}/final"
os.makedirs(final_dir, exist_ok=True)
trainer.model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

# Save config for reference
config_info = {
    "model_name": MODEL_NAME,
    "max_source_length": MAX_SOURCE_LENGTH,
    "max_target_length": MAX_TARGET_LENGTH,
    "epochs": EPOCHS,
    "normalization": "v7_preserve_diacritics",
}
with open(f"{final_dir}/v7_config.json", "w") as f:
    json.dump(config_info, f, indent=2)

print(f"Model saved to {final_dir}/")
for f in sorted(os.listdir(final_dir)):
    size = os.path.getsize(f"{final_dir}/{f}") / 1e6
    print(f"  {f} ({size:.1f} MB)")

# %% [markdown]
# Sanity check: Generate a sample translation

# %%
model.eval()
test_input = "um-ma ka-ru-um"
inputs = tokenizer(test_input, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(**inputs, max_length=50, num_beams=4)
translation = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"Sanity check:")
print(f"  Input: '{test_input}'")
print(f"  Output: '{translation}'")
assert translation.strip() != "", "WARNING: Empty output!"
print("  OK - model produces non-empty output")

# %% [markdown]
# Upload instructions
#
# Upload the `outputs_v7/final/` folder to Kaggle as a Model or Dataset named `akkadian-v7-model`.
#
# Required files:
# - config.json
# - model.safetensors
# - tokenizer.json
# - tokenizer_config.json
# - special_tokens_map.json
# - v7_config.json
