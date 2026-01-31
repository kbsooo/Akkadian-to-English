#%% [markdown]
# # Akkadian V2 Training (Colab Version)
#
# **Key Changes from V1:**
# - Unified ASCII normalization (Train/Test style mismatch fixed)
# - All diacritics converted to ASCII (š→s, à→a, etc.)
#
# **Environment**: Google Colab with GPU
#
# **Output**: Saved to Google Drive `/content/drive/MyDrive/akkadian/v2`

#%% [markdown]
# ## 0. Setup: Kaggle Data Download

#%%
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

#%%
# Kaggle Hub login and data download
import kagglehub
kagglehub.login()

#%%
# Download data from Kaggle
kbsooo_akkadian_v2_data_path = kagglehub.dataset_download('kbsooo/akkadian-v2-data')
print(f'Data downloaded to: {kbsooo_akkadian_v2_data_path}')

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
    """Training configuration for Colab GPU."""
    # Model
    model_name: str = "google/byt5-base"
    
    # Paths (Colab + Google Drive)
    data_dir: Path = None  # Set after kagglehub download
    output_dir: Path = Path("/content/drive/MyDrive/akkadian/v2")
    
    # Training
    seed: int = 42
    max_source_length: int = 512
    max_target_length: int = 512
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    epochs: int = 10
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    
    # Hardware
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2


CFG = Config()

# Set data directory from kagglehub download
CFG.data_dir = Path(kbsooo_akkadian_v2_data_path)

# Ensure output directory exists
CFG.output_dir.mkdir(parents=True, exist_ok=True)

print(f"📁 Data directory: {CFG.data_dir}")
print(f"📁 Output directory: {CFG.output_dir}")
print(f"🎮 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

#%% [markdown]
# ## 2. Load Data

#%%
print("📖 Loading preprocessed data...")
train_df = pd.read_csv(CFG.data_dir / "v2_train_augmented.csv")
val_df = pd.read_csv(CFG.data_dir / "v2_val.csv")

print(f"   Train: {len(train_df)}, Val: {len(val_df)}")
print(f"\n📝 Sample:")
print(f"   src: {train_df.iloc[0]['src'][:80]}...")
print(f"   tgt: {train_df.iloc[0]['tgt'][:80]}...")

#%% [markdown]
# ## 3. Tokenization

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
    # ByT5: encoder/decoder share same vocab, no as_target_tokenizer() needed
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
# ## 4. Metrics

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
# ## 5. Training

#%%
from transformers import TrainerCallback

class LoggingCallback(TrainerCallback):
    """Custom callback for cleaner training logs."""
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.log_history:
            last_log = state.log_history[-1]
            epoch = last_log.get('epoch', 0)
            train_loss = last_log.get('loss', 'N/A')
            print(f"\n{'='*50}")
            print(f"📊 Epoch {int(epoch)} Complete")
            print(f"   Train Loss: {train_loss:.4f}" if isinstance(train_loss, float) else f"   Train Loss: {train_loss}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\n📈 Validation Results:")
            print(f"   Loss: {metrics.get('eval_loss', 'N/A'):.4f}" if metrics.get('eval_loss') else "   Loss: N/A")
            print(f"   BLEU: {metrics.get('eval_bleu', 0):.2f}")
            print(f"   chrF++: {metrics.get('eval_chrf', 0):.2f}")
            print(f"   Geo Mean: {metrics.get('eval_geo_mean', 0):.2f}")
            print(f"{'='*50}")


#%%
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(CFG.output_dir / "checkpoints"),
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
    logging_steps=100,  # Less frequent logging
    logging_first_step=True,
    report_to="none",
    seed=CFG.seed,
    disable_tqdm=False,  # Keep progress bar
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=build_compute_metrics(tokenizer),
    callbacks=[LoggingCallback()],
)

#%%
print("\n🏋️ Training...")
trainer.train()

#%%
# Save final model to Google Drive
final_dir = CFG.output_dir / "final"
print(f"\n💾 Saving to Google Drive: {final_dir}")
trainer.save_model(str(final_dir))
tokenizer.save_pretrained(str(final_dir))

# Evaluate
print("\n📈 Final evaluation:")
results = trainer.evaluate()
print(f"   BLEU: {results.get('eval_bleu', 0):.2f}")
print(f"   chrF++: {results.get('eval_chrf', 0):.2f}")
print(f"   Geo Mean: {results.get('eval_geo_mean', 0):.2f}")

print("\n✅ Training complete!")
print(f"📁 Model saved to: {final_dir}")

#%% [markdown]
# ## 6. Create Model Archive (Optional)

#%%
import shutil

zip_path = CFG.output_dir / "akkadian_v2_model"
shutil.make_archive(str(zip_path), 'zip', final_dir)
print(f"📦 Model archived: {zip_path}.zip")
