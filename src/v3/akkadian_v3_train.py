#%% [markdown]
# # Akkadian V3 Training: ByT5-Large + LoRA
#
# **Key Features:**
# - ByT5-Large (1.2B params) with LoRA for parameter-efficient fine-tuning
# - Sentence-level data for better train/test distribution match
# - tqdm-based progress tracking for better visibility
#
# **Environment**: Google Colab with A100 GPU
#
# **Output**: Saved to Google Drive `/content/drive/MyDrive/akkadian/v3`

#%% [markdown]
# ## 0. Setup: Install Dependencies & Mount Drive

#%%
# Install PEFT for LoRA
!pip install -q peft accelerate

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from sacrebleu.metrics import BLEU, CHRF
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)

#%%
@dataclass
class Config:
    """Training configuration for ByT5-Large + LoRA on A100."""
    
    # Model
    model_name: str = "google/byt5-large"
    
    # Paths (Colab + Google Drive)
    data_dir: Path = None  # Set after kagglehub download
    output_dir: Path = Path("/content/drive/MyDrive/akkadian/v3")

    # Data files (sentence-level, already normalized)
    train_file: str = "v2_sentence_train.csv"
    val_file: str = "v2_sentence_val.csv"
    
    # LoRA Configuration (from V3 strategy doc)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q", "v"])
    
    # Training
    seed: int = 42
    max_source_length: int = 256
    max_target_length: int = 256
    batch_size: int = 4  # A100 can handle more
    gradient_accumulation_steps: int = 4
    epochs: int = 10
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Hardware - FP16 OFF for ByT5 numerical stability
    fp16: bool = False
    bf16: bool = False  # Can try True on A100
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2


CFG = Config()

# Set data directory from kagglehub download
CFG.data_dir = Path(kbsooo_akkadian_v2_data_path)

# Ensure output directory exists
CFG.output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("🚀 Akkadian V3: ByT5-Large + LoRA Training")
print("=" * 60)
print(f"📁 Data directory: {CFG.data_dir}")
print(f"📁 Output directory: {CFG.output_dir}")
print(f"🤖 Model: {CFG.model_name}")
print(f"🔧 LoRA: r={CFG.lora_r}, alpha={CFG.lora_alpha}")
print(f"🎮 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60)

# Reproducibility
set_seed(CFG.seed)

#%% [markdown]
# ## 2. Load Data

#%%
def load_data():
    """Load pre-normalized sentence-level data."""
    train_path = CFG.data_dir / CFG.train_file
    val_path = CFG.data_dir / CFG.val_file
    
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val file not found: {val_path}")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Validate columns
    required_cols = {"src", "tgt"}
    for name, df in [("train", train_df), ("val", val_df)]:
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing columns in {name}: {missing}")
    
    # Drop NaN
    train_df = train_df.dropna(subset=["src", "tgt"]).reset_index(drop=True)
    val_df = val_df.dropna(subset=["src", "tgt"]).reset_index(drop=True)
    
    return train_df, val_df


print("📖 Loading preprocessed sentence-level data...")
train_df, val_df = load_data()

print(f"   Train: {len(train_df):,} samples")
print(f"   Val: {len(val_df):,} samples")

print(f"\n📝 Sample:")
print(f"   src: {train_df.iloc[0]['src'][:80]}...")
print(f"   tgt: {train_df.iloc[0]['tgt'][:80]}...")

# Truncation risk check
src_over = (train_df["src"].str.len() > CFG.max_source_length).mean()
tgt_over = (train_df["tgt"].str.len() > CFG.max_target_length).mean()
print(f"\n⚠️ Truncation risk: src>{CFG.max_source_length}: {src_over:.1%}, tgt>{CFG.max_target_length}: {tgt_over:.1%}")

#%% [markdown]
# ## 3. Model & LoRA Setup

#%%
print(f"\n🤖 Loading base model: {CFG.model_name}")
print("   This may take a few minutes...")

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name)

print(f"   Base model loaded: {sum(p.numel() for p in base_model.parameters()):,} parameters")

#%%
# Apply LoRA
print(f"\n🔧 Applying LoRA (r={CFG.lora_r}, alpha={CFG.lora_alpha})...")

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=CFG.lora_r,
    lora_alpha=CFG.lora_alpha,
    lora_dropout=CFG.lora_dropout,
    target_modules=CFG.lora_target_modules,
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

if CFG.gradient_checkpointing:
    model.enable_input_require_grads()  # Required for LoRA + gradient checkpointing
    model.gradient_checkpointing_enable()
    print("   ✅ Gradient checkpointing enabled")

#%% [markdown]
# ## 4. Tokenization

#%%
def tokenize_fn(examples):
    """Tokenize source and target texts."""
    model_inputs = tokenizer(
        examples["src"],
        max_length=CFG.max_source_length,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        examples["tgt"],
        max_length=CFG.max_target_length,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("\n🔤 Tokenizing datasets...")
train_ds = Dataset.from_pandas(train_df[["src", "tgt"]])
val_ds = Dataset.from_pandas(val_df[["src", "tgt"]])

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"], desc="Tokenizing train")
val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"], desc="Tokenizing val")

print(f"   Train: {len(train_ds):,} samples")
print(f"   Val: {len(val_ds):,} samples")

#%% [markdown]
# ## 5. Metrics

#%%
def build_compute_metrics(tokenizer):
    """Build metrics computation function."""
    bleu = BLEU()
    chrf = CHRF(word_order=2)
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Replace -100 with pad token
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Decode
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]  # List of references
        
        # Compute scores
        bleu_score = bleu.corpus_score(decoded_preds, decoded_labels).score
        chrf_score = chrf.corpus_score(decoded_preds, decoded_labels).score
        geo_mean = np.sqrt(bleu_score * chrf_score) if bleu_score > 0 and chrf_score > 0 else 0.0
        
        return {"bleu": bleu_score, "chrf": chrf_score, "geo_mean": geo_mean}
    
    return compute_metrics

#%% [markdown]
# ## 6. Custom Callbacks for Better Logging

#%%
class TqdmLoggingCallback(TrainerCallback):
    """Enhanced logging with tqdm-style progress and clear metrics display."""
    
    def __init__(self):
        self.current_epoch = 0
        self.train_loss = []
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch = int(state.epoch) if state.epoch else 0
        self.train_loss = []
        print(f"\n{'='*60}")
        print(f"📊 Epoch {self.current_epoch + 1}/{args.num_train_epochs}")
        print(f"{'='*60}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_loss.append(logs["loss"])
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.train_loss:
            avg_loss = sum(self.train_loss) / len(self.train_loss)
            print(f"\n📉 Epoch {self.current_epoch + 1} Train Loss: {avg_loss:.4f}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\n{'─'*40}")
            print(f"📈 Validation Results (Epoch {self.current_epoch + 1})")
            print(f"{'─'*40}")
            print(f"   Loss:     {metrics.get('eval_loss', 0):.4f}")
            print(f"   BLEU:     {metrics.get('eval_bleu', 0):.2f}")
            print(f"   chrF++:   {metrics.get('eval_chrf', 0):.2f}")
            print(f"   Geo Mean: {metrics.get('eval_geo_mean', 0):.2f}")
            print(f"{'─'*40}")
    
    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n{'='*60}")
        print("🎉 Training Complete!")
        print(f"   Total steps: {state.global_step:,}")
        print(f"   Best metric: {state.best_metric:.2f}" if state.best_metric else "")
        print(f"{'='*60}")

#%% [markdown]
# ## 7. Training

#%%
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

training_kwargs = dict(
    output_dir=str(CFG.output_dir / "checkpoints"),
    num_train_epochs=CFG.epochs,
    per_device_train_batch_size=CFG.batch_size,
    per_device_eval_batch_size=CFG.batch_size * 2,
    gradient_accumulation_steps=CFG.gradient_accumulation_steps,
    learning_rate=CFG.learning_rate,
    weight_decay=CFG.weight_decay,
    warmup_ratio=CFG.warmup_ratio,
    max_grad_norm=CFG.max_grad_norm,
    fp16=CFG.fp16 and torch.cuda.is_available(),
    bf16=CFG.bf16 and torch.cuda.is_available(),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_geo_mean",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=CFG.max_target_length,
    dataloader_num_workers=CFG.dataloader_num_workers,
    logging_steps=50,
    logging_first_step=True,
    report_to="none",
    seed=CFG.seed,
    disable_tqdm=False,
)

# Handle API version differences
try:
    training_args = Seq2SeqTrainingArguments(**training_kwargs)
except TypeError:
    training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

#%%
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=build_compute_metrics(tokenizer),
    callbacks=[TqdmLoggingCallback()],
)

#%%
print("\n🏋️ Starting training...")
print(f"   Epochs: {CFG.epochs}")
print(f"   Batch size: {CFG.batch_size} x {CFG.gradient_accumulation_steps} = {CFG.batch_size * CFG.gradient_accumulation_steps}")
print(f"   Learning rate: {CFG.learning_rate}")
print()

trainer.train()

#%% [markdown]
# ## 8. Save Model

#%%
# Save LoRA adapter to Google Drive
adapter_dir = CFG.output_dir / "lora_adapter"
print(f"\n💾 Saving LoRA adapter to: {adapter_dir}")
model.save_pretrained(str(adapter_dir))
tokenizer.save_pretrained(str(adapter_dir))

#%%
# Final evaluation
print("\n📈 Final Evaluation:")
results = trainer.evaluate()
print(f"   BLEU:     {results.get('eval_bleu', 0):.2f}")
print(f"   chrF++:   {results.get('eval_chrf', 0):.2f}")
print(f"   Geo Mean: {results.get('eval_geo_mean', 0):.2f}")

#%% [markdown]
# ## 9. Create Archive

#%%
import shutil

# Create ZIP archive for easy download
zip_path = CFG.output_dir / "akkadian_v3_lora"
shutil.make_archive(str(zip_path), 'zip', adapter_dir)
print(f"\n📦 Model archived: {zip_path}.zip")

print("\n" + "=" * 60)
print("✅ V3 Training Complete!")
print("=" * 60)
print(f"📁 LoRA adapter: {adapter_dir}")
print(f"📦 Archive: {zip_path}.zip")
print("\nNext steps:")
print("1. Download the archive from Google Drive")
print("2. Upload to Kaggle as a dataset for inference")
print("=" * 60)
