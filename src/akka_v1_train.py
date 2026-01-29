#%% [markdown]
# # Akkadian → English Translation: Training (V1)
#
# **Environment**: Kaggle T4 GPU x2
#
# **Model**: ByT5-base (byte-level tokenization, good for low-resource languages)
#
# **Workflow**:
# 1. Load preprocessed sentence pairs from competition data
# 2. Train ByT5 with grouped train/val split
# 3. Save model to Kaggle Models
#
# **Usage (convert to notebook)**:
# ```bash
# uv run jupytext --to notebook src/akka_v1_train.py
# ```

#%% [markdown]
# ## 1. Imports & Configuration

#%%
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
# =============================
# Configuration
# =============================

@dataclass
class Config:
    """Training configuration for Kaggle T4 x2 environment."""
    # Model
    model_name: str = "google/byt5-base"
    
    # Paths (Kaggle)
    kaggle_input: Path = Path("/kaggle/input")
    kaggle_working: Path = Path("/kaggle/working")
    
    # Data
    train_file: str = "train.csv"  # from competition data
    
    # Training
    seed: int = 42
    val_frac: float = 0.1
    max_source_length: int = 256
    max_target_length: int = 256
    batch_size: int = 4  # per GPU
    gradient_accumulation_steps: int = 4
    epochs: int = 5
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    
    # Hardware
    fp16: bool = True  # T4 supports FP16 well
    bf16: bool = False  # T4 doesn't support BF16
    gradient_checkpointing: bool = True  # save memory
    dataloader_num_workers: int = 2
    
    # Evaluation
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_bleu"
    greater_is_better: bool = True


CFG = Config()

#%% [markdown]
# ## 2. Environment Detection

#%%
def is_kaggle() -> bool:
    """Check if running on Kaggle."""
    return Path("/kaggle/input").exists()


def find_competition_data() -> Path:
    """Find competition data directory."""
    if not is_kaggle():
        # Local fallback
        local_path = Path("data")
        if local_path.exists():
            return local_path
        raise FileNotFoundError("Cannot find competition data locally")
    
    # On Kaggle: look for train.csv
    for d in CFG.kaggle_input.iterdir():
        if (d / "train.csv").exists():
            return d
    raise FileNotFoundError("Cannot find competition data in /kaggle/input")


def get_output_dir() -> Path:
    """Get output directory for model checkpoints."""
    if is_kaggle():
        return CFG.kaggle_working / "akkadian_v1"
    return Path("outputs/akkadian_v1")


COMP_DATA_DIR = find_competition_data()
OUTPUT_DIR = get_output_dir()

print(f"📁 Competition data: {COMP_DATA_DIR}")
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"🖥️ Running on Kaggle: {is_kaggle()}")
print(f"🎮 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

#%% [markdown]
# ## 3. Data Preprocessing

#%%
# Subscript conversion map
_SUBSCRIPT_MAP = str.maketrans({
    "\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u2084": "4",
    "\u2085": "5", "\u2086": "6", "\u2087": "7", "\u2088": "8", "\u2089": "9",
    "\u2093": "x",
})


def normalize_transliteration(text: str) -> str:
    """Normalize Akkadian transliteration for model input."""
    if pd.isna(text):
        return ""
    text = str(text)
    
    # Unicode normalization
    text = unicodedata.normalize("NFC", text)
    
    # Normalize special H character
    text = text.replace("\u1E2A", "H").replace("\u1E2B", "h")
    
    # Convert subscripts to numbers
    text = text.translate(_SUBSCRIPT_MAP)
    
    # Handle gaps and damaged portions
    text = text.replace("\u2026", " <gap> ")  # ellipsis
    text = re.sub(r"\.\.\.+", " <gap> ", text)
    text = re.sub(r"\[([^\]]*)\]", " <gap> ", text)  # [damaged text]
    
    # Handle unknown signs
    text = re.sub(r"\bx\b", " <unk> ", text)
    
    # Remove editorial marks
    text = re.sub(r"[!?/]", " ", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def normalize_translation(text: str) -> str:
    """Normalize English translation for model output."""
    if pd.isna(text):
        return ""
    text = str(text)
    
    # Unicode normalization
    text = unicodedata.normalize("NFC", text)
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"['']", "'", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


#%%
def load_and_preprocess_data(data_path: Path) -> pd.DataFrame:
    """Load and preprocess training data."""
    print(f"📖 Loading data from {data_path}")
    
    train_df = pd.read_csv(data_path / CFG.train_file)
    print(f"   Raw samples: {len(train_df)}")
    
    # Check columns
    required_cols = {"transliteration", "translation"}
    if not required_cols.issubset(train_df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(train_df.columns)}")
    
    # Add oare_id if not present (for grouping)
    if "oare_id" not in train_df.columns:
        train_df["oare_id"] = train_df.index.astype(str)
    
    # Normalize texts
    train_df["src"] = train_df["transliteration"].apply(normalize_transliteration)
    train_df["tgt"] = train_df["translation"].apply(normalize_translation)
    
    # Filter empty samples
    mask = (train_df["src"].str.len() > 5) & (train_df["tgt"].str.len() > 5)
    train_df = train_df[mask].reset_index(drop=True)
    print(f"   After filtering: {len(train_df)}")
    
    return train_df


#%%
def group_split(df: pd.DataFrame, group_col: str, val_frac: float, seed: int):
    """Split data by group to prevent data leakage."""
    groups = df[group_col].unique()
    np.random.seed(seed)
    np.random.shuffle(groups)
    
    n_val = max(1, int(len(groups) * val_frac))
    val_groups = set(groups[:n_val])
    
    train_mask = ~df[group_col].isin(val_groups)
    
    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[~train_mask].reset_index(drop=True)
    
    return train_df, val_df


#%% [markdown]
# ## 4. Tokenization & Dataset

#%%
def build_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer):
    """Build HuggingFace datasets with tokenization."""
    
    def tokenize_fn(examples):
        # Tokenize source
        model_inputs = tokenizer(
            examples["src"],
            max_length=CFG.max_source_length,
            truncation=True,
            padding=False,
        )
        
        # Tokenize target
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["tgt"],
                max_length=CFG.max_target_length,
                truncation=True,
                padding=False,
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Create datasets
    train_ds = Dataset.from_pandas(train_df[["src", "tgt"]])
    val_ds = Dataset.from_pandas(val_df[["src", "tgt"]])
    
    # Tokenize
    train_ds = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["src", "tgt"],
        desc="Tokenizing train",
    )
    val_ds = val_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["src", "tgt"],
        desc="Tokenizing val",
    )
    
    return train_ds, val_ds


#%% [markdown]
# ## 5. Metrics

#%%
def build_compute_metrics(tokenizer):
    """Build compute_metrics function for Trainer."""
    bleu = BLEU()
    chrf = CHRF(word_order=2)  # chrF++
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # Decode predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Replace -100 with pad token
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Decode
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean up
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]  # wrapped for sacrebleu
        
        # Calculate metrics
        bleu_score = bleu.corpus_score(decoded_preds, decoded_labels).score
        chrf_score = chrf.corpus_score(decoded_preds, decoded_labels).score
        
        # Geometric mean (competition metric)
        geo_mean = np.sqrt(bleu_score * chrf_score) if bleu_score > 0 and chrf_score > 0 else 0.0
        
        return {
            "bleu": bleu_score,
            "chrf": chrf_score,
            "geo_mean": geo_mean,
        }
    
    return compute_metrics


#%% [markdown]
# ## 6. Training

#%%
def train():
    """Main training function."""
    print("=" * 60)
    print("🚀 Starting Akkadian V1 Training")
    print("=" * 60)
    
    # Set seed
    set_seed(CFG.seed)
    
    # Load data
    df = load_and_preprocess_data(COMP_DATA_DIR)
    train_df, val_df = group_split(df, "oare_id", CFG.val_frac, CFG.seed)
    print(f"📊 Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Load model and tokenizer
    print(f"🤖 Loading model: {CFG.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name)
    
    if CFG.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("   ✅ Gradient checkpointing enabled")
    
    # Build datasets
    train_ds, val_ds = build_datasets(train_df, val_df, tokenizer)
    print(f"   Train tokens: {len(train_ds)}, Val tokens: {len(val_ds)}")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        
        # Training
        num_train_epochs=CFG.epochs,
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size * 2,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        
        # Optimizer
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        warmup_ratio=CFG.warmup_ratio,
        
        # Precision
        fp16=CFG.fp16 and torch.cuda.is_available(),
        bf16=CFG.bf16,
        
        # Evaluation
        eval_strategy=CFG.eval_strategy,
        save_strategy=CFG.save_strategy,
        save_total_limit=CFG.save_total_limit,
        load_best_model_at_end=CFG.load_best_model_at_end,
        metric_for_best_model=CFG.metric_for_best_model,
        greater_is_better=CFG.greater_is_better,
        predict_with_generate=True,
        generation_max_length=CFG.max_target_length,
        
        # Misc
        dataloader_num_workers=CFG.dataloader_num_workers,
        logging_steps=50,
        report_to="none",  # disable wandb etc on Kaggle
        seed=CFG.seed,
        
        # Multi-GPU (Kaggle T4 x2)
        ddp_find_unused_parameters=False,
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )
    
    # Train
    print("\n🏋️ Training...")
    trainer.train()
    
    # Save final model
    final_model_dir = OUTPUT_DIR / "final"
    print(f"\n💾 Saving final model to {final_model_dir}")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    # Evaluate
    print("\n📈 Final evaluation...")
    eval_results = trainer.evaluate()
    print(f"   BLEU: {eval_results.get('eval_bleu', 'N/A'):.2f}")
    print(f"   chrF++: {eval_results.get('eval_chrf', 'N/A'):.2f}")
    print(f"   Geo Mean: {eval_results.get('eval_geo_mean', 'N/A'):.2f}")
    
    print("\n✅ Training complete!")
    print(f"📁 Model saved to: {final_model_dir}")
    
    return trainer


#%% [markdown]
# ## 7. Run Training

#%%
if __name__ == "__main__":
    trainer = train()

#%% [markdown]
# ## 8. Upload to Kaggle Models
#
# After training, upload the model to Kaggle Models:
#
# ```python
# # In Kaggle notebook, after training:
# import kaggle
# 
# # Create model metadata
# model_dir = "/kaggle/working/akkadian_v1/final"
# 
# # Upload via Kaggle API
# # kaggle models create -p {model_dir} --title "akkadian-byt5-v1"
# ```
#
# Or manually:
# 1. Download the `/kaggle/working/akkadian_v1/final` folder
# 2. Go to Kaggle Models → New Model
# 3. Upload the folder

#%%
# Optional: Create a zip for easy download
def create_model_zip():
    """Create a zip file of the trained model for easy download."""
    import shutil
    
    model_dir = OUTPUT_DIR / "final"
    if not model_dir.exists():
        print("❌ No model found to zip")
        return
    
    zip_path = CFG.kaggle_working / "akkadian_v1_model"
    shutil.make_archive(str(zip_path), 'zip', model_dir)
    print(f"📦 Model zipped to: {zip_path}.zip")


# Uncomment to create zip after training:
# create_model_zip()
