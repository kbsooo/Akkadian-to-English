#%% [markdown]
# # Akkadian V4 Training: Full FT + OCR Noise Augmentation
#
# **Key Features:**
# - ByT5 (base or large) with Full Fine-tuning
# - OCR noise augmentation for robustness
# - Same data as V2 (v2_sentence_train.csv)
#
# **Environment**: Google Colab with A100 GPU
#
# **Output**: Saved to Google Drive `/content/drive/MyDrive/akkadian/v4`

#%% [markdown]
# ## 0. Setup: Mount Drive & Download Data

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
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
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
    """Training configuration for V4: Full FT + OCR Augmentation."""
    
    # Model selection: "base" or "large"
    model_size: str = "base"  # Change to "large" for v4-large
    
    # Paths (Colab + Google Drive)
    data_dir: Path = None  # Set after kagglehub download
    output_dir: Path = None  # Set after model_size is determined
    
    # Data files
    train_file: str = "v2_sentence_train.csv"
    val_file: str = "v2_sentence_val.csv"
    
    # OCR Augmentation
    augment_prob: float = 0.3  # Probability of applying noise to each sample
    
    # Training
    seed: int = 42
    max_source_length: int = 512
    max_target_length: int = 512
    epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Hardware
    fp16: bool = False  # ByT5 is unstable with FP16
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    
    # Model-specific settings (set in __post_init__)
    model_name: str = field(init=False)
    batch_size: int = field(init=False)
    gradient_accumulation_steps: int = field(init=False)
    learning_rate: float = field(init=False)
    
    def __post_init__(self):
        if self.model_size == "base":
            self.model_name = "google/byt5-base"
            self.batch_size = 4
            self.gradient_accumulation_steps = 4
            self.learning_rate = 1e-4
            self.output_dir = Path("/content/drive/MyDrive/akkadian/v4-base")
        else:
            self.model_name = "google/byt5-large"
            self.batch_size = 2
            self.gradient_accumulation_steps = 8
            self.learning_rate = 5e-5
            self.output_dir = Path("/content/drive/MyDrive/akkadian/v4-large")


# ============================================
# ⚠️ CHANGE THIS FOR v4-large
# ============================================
CFG = Config(model_size="base")  # "base" or "large"

# Set data directory
CFG.data_dir = Path(kbsooo_akkadian_v2_data_path)
CFG.output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print(f"🚀 Akkadian V4: {CFG.model_size.upper()} + Full FT + OCR Augmentation")
print("=" * 60)
print(f"📁 Data directory: {CFG.data_dir}")
print(f"📁 Output directory: {CFG.output_dir}")
print(f"🤖 Model: {CFG.model_name}")
print(f"🎲 Augment prob: {CFG.augment_prob}")
print(f"🎮 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60)

set_seed(CFG.seed)

#%% [markdown]
# ## 2. OCR Noise Augmentation

#%%
# ==============================================================================
# OCR Noise Augmentation Functions
# ==============================================================================

# Diacritics that can be dropped
_DIACRITICS_MAP = {
    'š': 's', 'Š': 'S',
    'ṣ': 's', 'Ṣ': 'S',
    'ṭ': 't', 'Ṭ': 'T',
    'ḫ': 'h', 'Ḫ': 'H',
    'ā': 'a', 'Ā': 'A',
    'ē': 'e', 'Ē': 'E',
    'ī': 'i', 'Ī': 'I',
    'ū': 'u', 'Ū': 'U',
}

# Quote variations (can go both ways)
_QUOTE_PAIRS = [
    ('"', '„'),
    ('"', '"'),
    ("'", '''),
    ("'", '''),
]

# Subscript variations
_SUBSCRIPT_MAP = {
    '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
    '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
}


def drop_diacritics(text: str, prob: float = 0.5) -> str:
    """Randomly drop diacritics from characters."""
    result = []
    for char in text:
        if char in _DIACRITICS_MAP and random.random() < prob:
            result.append(_DIACRITICS_MAP[char])
        else:
            result.append(char)
    return ''.join(result)


def vary_quotes(text: str) -> str:
    """Randomly swap quote styles."""
    for orig, alt in _QUOTE_PAIRS:
        if random.random() < 0.5:
            text = text.replace(orig, alt)
    return text


def vary_subscripts(text: str, prob: float = 0.5) -> str:
    """Randomly convert subscripts to regular numbers or vice versa."""
    result = []
    for char in text:
        if char in _SUBSCRIPT_MAP and random.random() < prob:
            result.append(_SUBSCRIPT_MAP[char])
        elif char.isdigit() and random.random() < prob * 0.3:
            # Occasionally convert number to subscript
            subscripts = '₀₁₂₃₄₅₆₇₈₉'
            result.append(subscripts[int(char)])
        else:
            result.append(char)
    return ''.join(result)


def drop_hyphens(text: str, prob: float = 0.2) -> str:
    """Randomly drop hyphens between syllables."""
    if random.random() < prob:
        return text.replace('-', '')
    return text


def add_ocr_artifacts(text: str, prob: float = 0.1) -> str:
    """Randomly insert OCR-like artifacts."""
    if random.random() < prob:
        artifacts = ['·', '°', '+', '×']
        pos = random.randint(0, len(text))
        artifact = random.choice(artifacts)
        return text[:pos] + artifact + text[pos:]
    return text


def vary_brackets(text: str) -> str:
    """Vary bracket styles."""
    variations = [
        ('[', '⌈'), (']', '⌉'),
        ('[', '⌊'), (']', '⌋'),
    ]
    for orig, alt in variations:
        if random.random() < 0.3:
            text = text.replace(orig, alt)
    return text


def apply_ocr_noise(text: str, prob: float = 0.3) -> str:
    """
    Apply random OCR-like noise to transliteration.
    
    This teaches the model to handle various input styles.
    Only applied during TRAINING, not inference.
    """
    if random.random() > prob:
        return text  # No augmentation for this sample
    
    # Apply one or more noise types
    noise_funcs = [
        lambda t: drop_diacritics(t, prob=0.3),
        vary_quotes,
        lambda t: vary_subscripts(t, prob=0.3),
        lambda t: drop_hyphens(t, prob=0.15),
        lambda t: add_ocr_artifacts(t, prob=0.1),
        vary_brackets,
    ]
    
    # Apply 1-3 random noise functions
    n_funcs = random.randint(1, 3)
    selected_funcs = random.sample(noise_funcs, n_funcs)
    
    for func in selected_funcs:
        text = func(text)
    
    return text


# Test augmentation
print("\n📝 OCR Augmentation Examples:")
test_text = "šum-ma a-wi-lum ṣa-bi-tam i-na ḫu-bu-ul-li-šu"
for i in range(5):
    augmented = apply_ocr_noise(test_text, prob=1.0)
    print(f"   [{i}] {augmented}")

#%% [markdown]
# ## 3. Load Data

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


print("📖 Loading data...")
train_df, val_df = load_data()

print(f"   Train: {len(train_df):,} samples")
print(f"   Val: {len(val_df):,} samples")

print(f"\n📝 Sample (original):")
print(f"   src: {train_df.iloc[0]['src'][:80]}...")
print(f"   tgt: {train_df.iloc[0]['tgt'][:80]}...")

#%% [markdown]
# ## 4. Model Setup

#%%
print(f"\n🤖 Loading model: {CFG.model_name}")
print("   This may take a few minutes...")

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name)

print(f"   Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

if CFG.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    print("   ✅ Gradient checkpointing enabled")

#%% [markdown]
# ## 5. Tokenization with Augmentation

#%%
def tokenize_with_augment(examples, augment=True):
    """Tokenize with optional OCR augmentation on source."""
    # Apply augmentation to source (training only)
    if augment:
        sources = [apply_ocr_noise(s, prob=CFG.augment_prob) for s in examples["src"]]
    else:
        sources = examples["src"]
    
    model_inputs = tokenizer(
        sources,
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

# Train with augmentation
train_ds = Dataset.from_pandas(train_df[["src", "tgt"]])
train_ds = train_ds.map(
    lambda x: tokenize_with_augment(x, augment=True),
    batched=True,
    remove_columns=["src", "tgt"],
    desc="Tokenizing train (with augmentation)"
)

# Validation without augmentation (fair evaluation)
val_ds = Dataset.from_pandas(val_df[["src", "tgt"]])
val_ds = val_ds.map(
    lambda x: tokenize_with_augment(x, augment=False),
    batched=True,
    remove_columns=["src", "tgt"],
    desc="Tokenizing val (no augmentation)"
)

print(f"   Train: {len(train_ds):,} samples")
print(f"   Val: {len(val_ds):,} samples")

#%% [markdown]
# ## 6. Metrics

#%%
def build_compute_metrics(tokenizer):
    """Build metrics computation function."""
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
# ## 7. Logging Callback

#%%
class TqdmLoggingCallback(TrainerCallback):
    """Enhanced logging with clear metrics display."""
    
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
        if state.best_metric:
            print(f"   Best metric: {state.best_metric:.2f}")
        print(f"{'='*60}")

#%% [markdown]
# ## 8. Training

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
print(f"   Model: {CFG.model_size.upper()}")
print(f"   Epochs: {CFG.epochs}")
print(f"   Batch size: {CFG.batch_size} x {CFG.gradient_accumulation_steps} = {CFG.batch_size * CFG.gradient_accumulation_steps}")
print(f"   Learning rate: {CFG.learning_rate}")
print(f"   Augment prob: {CFG.augment_prob}")
print()

trainer.train()

#%% [markdown]
# ## 9. Save Model

#%%
model_dir = CFG.output_dir / "model"
print(f"\n💾 Saving model to: {model_dir}")
trainer.save_model(str(model_dir))
tokenizer.save_pretrained(str(model_dir))

#%%
# Final evaluation
print("\n📈 Final Evaluation:")
results = trainer.evaluate()
print(f"   BLEU:     {results.get('eval_bleu', 0):.2f}")
print(f"   chrF++:   {results.get('eval_chrf', 0):.2f}")
print(f"   Geo Mean: {results.get('eval_geo_mean', 0):.2f}")

#%% [markdown]
# ## 10. Create Archive

#%%
import shutil

zip_path = CFG.output_dir / f"akkadian_v4_{CFG.model_size}"
shutil.make_archive(str(zip_path), 'zip', model_dir)
print(f"\n📦 Model archived: {zip_path}.zip")

print("\n" + "=" * 60)
print(f"✅ V4-{CFG.model_size.upper()} Training Complete!")
print("=" * 60)
print(f"📁 Model: {model_dir}")
print(f"📦 Archive: {zip_path}.zip")
print("\nNext steps:")
print("1. Download the archive from Google Drive")
print("2. Upload to Kaggle as a dataset for inference")
print("=" * 60)
