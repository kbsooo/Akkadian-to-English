#%% [markdown]
# # Akkadian V2 Training
#
# **Key Changes from V1:**
# - Unified ASCII normalization (Train/Test style mismatch fixed)
# - All diacritics converted to ASCII (š→s, à→a, etc.)
#
# **Environment**: Kaggle T4 GPU x2
#
# **Usage:**
# ```bash
# uv run jupytext --to notebook src/v2/akka_v2_train.py
# ```

#%% [markdown]
# ## 1. Imports & Configuration

#%%
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

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

#%%
@dataclass
class Config:
    """Training configuration for Kaggle T4 x2."""
    # Model
    model_name: str = "google/byt5-base"
    
    # Paths
    kaggle_input: Path = Path("/kaggle/input")
    kaggle_working: Path = Path("/kaggle/working")
    
    # Training
    seed: int = 42
    max_source_length: int = 512   # Increased for longer documents
    max_target_length: int = 512
    batch_size: int = 2            # Per GPU
    gradient_accumulation_steps: int = 8
    epochs: int = 10               # More epochs
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    
    # Hardware
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2


CFG = Config()

#%% [markdown]
# ## 2. Environment Detection

#%%
def is_kaggle() -> bool:
    return Path("/kaggle/input").exists()


def find_data_dir() -> Path:
    """Find V2 preprocessed data."""
    if is_kaggle():
        # On Kaggle: look for v2_train.csv in input datasets
        for d in CFG.kaggle_input.iterdir():
            if (d / "v2_train.csv").exists():
                return d
        # Fallback: run preprocessing
        raise FileNotFoundError("V2 data not found. Upload v2_train.csv and v2_val.csv as a dataset.")
    else:
        # Local
        local = Path("data/v2")
        if local.exists():
            return local
        raise FileNotFoundError("Run build_dataset.py first")


def get_output_dir() -> Path:
    if is_kaggle():
        return CFG.kaggle_working / "akkadian_v2"
    return Path("outputs/akkadian_v2")


DATA_DIR = find_data_dir()
OUTPUT_DIR = get_output_dir()

print(f"📁 Data directory: {DATA_DIR}")
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"🖥️ Kaggle: {is_kaggle()}")
print(f"🎮 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPUs: {torch.cuda.device_count()}")

#%% [markdown]
# ## 3. Load Data

#%%
print("📖 Loading preprocessed data...")
train_df = pd.read_csv(DATA_DIR / "v2_train.csv")
val_df = pd.read_csv(DATA_DIR / "v2_val.csv")

print(f"   Train: {len(train_df)}, Val: {len(val_df)}")
print(f"\n📝 Sample:")
print(f"   src: {train_df.iloc[0]['src'][:80]}...")
print(f"   tgt: {train_df.iloc[0]['tgt'][:80]}...")

#%% [markdown]
# ## 4. Tokenization

#%%
print(f"🤖 Loading model: {CFG.model_name}")
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name)

if CFG.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    print("   ✅ Gradient checkpointing enabled")

#%%
def tokenize_fn(examples):
    model_inputs = tokenizer(
        examples["src"],
        max_length=CFG.max_source_length,
        truncation=True,
        padding=False,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["tgt"],
            max_length=CFG.max_target_length,
            truncation=True,
            padding=False,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_ds = Dataset.from_pandas(train_df[["src", "tgt"]])
val_ds = Dataset.from_pandas(val_df[["src", "tgt"]])

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"], desc="Tokenizing train")
val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"], desc="Tokenizing val")

print(f"   Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

#%% [markdown]
# ## 5. Metrics

#%%
def build_compute_metrics(tokenizer):
    bleu = BLEU()
    chrf = CHRF(word_order=2)
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]
        
        bleu_score = bleu.corpus_score(decoded_preds, decoded_labels).score
        chrf_score = chrf.corpus_score(decoded_preds, decoded_labels).score
        geo_mean = np.sqrt(bleu_score * chrf_score) if bleu_score > 0 and chrf_score > 0 else 0.0
        
        return {"bleu": bleu_score, "chrf": chrf_score, "geo_mean": geo_mean}
    
    return compute_metrics

#%% [markdown]
# ## 6. Training

#%%
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=CFG.epochs,
    per_device_train_batch_size=CFG.batch_size,
    per_device_eval_batch_size=CFG.batch_size * 2,
    gradient_accumulation_steps=CFG.gradient_accumulation_steps,
    learning_rate=CFG.learning_rate,
    weight_decay=CFG.weight_decay,
    warmup_ratio=CFG.warmup_ratio,
    fp16=CFG.fp16 and torch.cuda.is_available(),
    bf16=CFG.bf16,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_bleu",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=CFG.max_target_length,
    dataloader_num_workers=CFG.dataloader_num_workers,
    logging_steps=50,
    report_to="none",
    seed=CFG.seed,
    ddp_find_unused_parameters=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=build_compute_metrics(tokenizer),
)

#%%
print("\n🏋️ Training...")
trainer.train()

#%%
# Save final model
final_dir = OUTPUT_DIR / "final"
print(f"\n💾 Saving to {final_dir}")
trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))

# Evaluate
print("\n📈 Final evaluation:")
results = trainer.evaluate()
print(f"   BLEU: {results.get('eval_bleu', 0):.2f}")
print(f"   chrF++: {results.get('eval_chrf', 0):.2f}")
print(f"   Geo Mean: {results.get('eval_geo_mean', 0):.2f}")

print("\n✅ Training complete!")

#%% [markdown]
# ## 7. Create Model Archive
#
# For Kaggle Models upload:

#%%
import shutil

if is_kaggle():
    zip_path = CFG.kaggle_working / "akkadian_v2_model"
    shutil.make_archive(str(zip_path), 'zip', final_dir)
    print(f"📦 Model archived: {zip_path}.zip")
