#!/usr/bin/env python3
"""
Akkadian V5d Training Script
============================
- Model: ByT5-small
- Device: MPS (Mac M4) or CUDA
- Tokenizer: Original HF tokenizer (NOT saved with model)
- Early Stopping: patience=3
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed,
)

# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class Config:
    # Paths
    data_dir: Path = Path("data/v5d")
    output_dir: Path = Path("outputs/v5d")
    
    # Model - IMPORTANT: Use exact same name in train AND infer
    model_name: str = "google/byt5-small"
    
    # Training
    epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch = 16
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 3
    
    # Sequence lengths
    max_source_length: int = 256
    max_target_length: int = 256
    
    # Glossary
    use_glossary: bool = True
    glossary_max_items: int = 8
    glossary_drop_prob: float = 0.5  # Train only
    
    # Misc
    seed: int = 42
    logging_steps: int = 50
    save_total_limit: int = 3
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


CFG = Config()


# ==============================================================================
# Device Setup
# ==============================================================================

def get_device() -> torch.device:
    """Get best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()

print("=" * 60)
print("🚀 Akkadian V5d Training")
print("=" * 60)
print(f"📁 Data: {CFG.data_dir}")
print(f"📁 Output: {CFG.output_dir}")
print(f"🤖 Model: {CFG.model_name}")
print(f"🎮 Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
elif DEVICE.type == "mps":
    print("   Apple Silicon MPS acceleration")
print("=" * 60)

set_seed(CFG.seed)


# ==============================================================================
# Normalization (same as infer!)
# ==============================================================================

_VOWEL_MAP = {
    "à": "a", "á": "a", "â": "a", "ā": "a", "ä": "a",
    "À": "A", "Á": "A", "Â": "A", "Ā": "A", "Ä": "A",
    "è": "e", "é": "e", "ê": "e", "ē": "e", "ë": "e",
    "È": "E", "É": "E", "Ê": "E", "Ē": "E", "Ë": "E",
    "ì": "i", "í": "i", "î": "i", "ī": "i", "ï": "i",
    "Ì": "I", "Í": "I", "Î": "I", "Ī": "I", "Ï": "I",
    "ò": "o", "ó": "o", "ô": "o", "ō": "o", "ö": "o",
    "Ò": "O", "Ó": "O", "Ô": "O", "Ō": "O", "Ö": "O",
    "ù": "u", "ú": "u", "û": "u", "ū": "u", "ü": "u",
    "Ù": "U", "Ú": "U", "Û": "U", "Ū": "U", "Ü": "U",
}

_CONSONANT_MAP = {
    "š": "s", "Š": "S",
    "ṣ": "s", "Ṣ": "S",
    "ṭ": "t", "Ṭ": "T",
    "ḫ": "h", "Ḫ": "H",
}

_QUOTE_MAP = {
    "„": '"', """: '"', """: '"',
    "'": "'", "'": "'", "‚": "'",
    "ʾ": "'", "ʿ": "'",
}

_SUBSCRIPT_MAP = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "ₓ": "x",
}

# Merge all character maps
_ALL_CHAR_MAP = {**_VOWEL_MAP, **_CONSONANT_MAP, **_QUOTE_MAP, **_SUBSCRIPT_MAP}

# Build translation table safely (filter out any invalid keys)
_TRANS_TABLE = {}
for k, v in _ALL_CHAR_MAP.items():
    if isinstance(k, str) and len(k) == 1:
        _TRANS_TABLE[ord(k)] = v


def normalize_transliteration(text) -> str:
    """Normalize Akkadian transliteration - MUST match infer exactly!"""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)

    # Protect literal gap tokens
    text = text.replace("<gap>", "__LIT_GAP__")
    text = text.replace("<big_gap>", "__LIT_BIG_GAP__")

    # Remove apostrophe line numbers only (1', 1'')
    text = re.sub(r"\b\d+'{1,2}\b", " ", text)

    # Remove <content> blocks first
    text = re.sub(r"<<([^>]+)>>", r"\1", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)

    # Large gaps
    text = re.sub(r"\[\s*…+\s*…*\s*\]", " __BIG_GAP__ ", text)
    text = re.sub(r"\[\s*\.\.\.+\s*\.\.\.+\s*\]", " __BIG_GAP__ ", text)

    # Ellipsis
    text = text.replace("…", " __BIG_GAP__ ")
    text = re.sub(r"\.\.\.+", " __BIG_GAP__ ", text)

    # [x]
    text = re.sub(r"\[\s*x\s*\]", " __GAP__ ", text, flags=re.IGNORECASE)

    # [content] -> content
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)

    # Half brackets
    for char in "‹›⌈⌉⌊⌋˹˺":
        text = text.replace(char, "")

    # Character maps
    text = text.translate(_TRANS_TABLE)

    # Scribal notations / word divider
    text = re.sub(r"[!?/]", " ", text)
    text = re.sub(r"\s*:\s*", " ", text)

    # Standalone x
    text = re.sub(r"\bx\b", " __GAP__ ", text, flags=re.IGNORECASE)

    # Convert placeholders
    text = text.replace("__GAP__", "<gap>")
    text = text.replace("__BIG_GAP__", "<big_gap>")

    # Restore literal tokens
    text = text.replace("__LIT_GAP__", "<gap>")
    text = text.replace("__LIT_BIG_GAP__", "<big_gap>")

    # Cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================================================================
# Glossary
# ==============================================================================

SRC_SPLIT_RE = re.compile(r"[\s\-]+")
TGT_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+")


def tokenize_src(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in SRC_SPLIT_RE.split(str(text)) if t]


def load_glossary(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: list(v) for k, v in data.items()}


def build_glossary_prompt(
    src: str,
    glossary: dict[str, list[str]],
    max_items: int,
    drop_prob: float,
    rng: random.Random,
) -> str:
    """Build glossary-augmented prompt for training."""
    if not glossary:
        return src
    if drop_prob > 0 and rng.random() < drop_prob:
        return src

    items = []
    used = set()
    for tok in tokenize_src(src):
        if tok in used:
            continue
        tgts = glossary.get(tok)
        if not tgts:
            continue
        items.append(f"{tok}={tgts[0]}")
        used.add(tok)
        if len(items) >= max_items:
            break

    if not items:
        return src

    return "GLOSSARY: " + "; ".join(items) + " ||| " + src


# ==============================================================================
# Data Loading
# ==============================================================================

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"src", "tgt"}.issubset(df.columns):
        raise ValueError(f"Missing src/tgt columns: {path}")
    df = df.dropna(subset=["src", "tgt"]).reset_index(drop=True)
    return df


def prepare_data(df: pd.DataFrame, glossary: dict, drop_prob: float, seed: int) -> pd.DataFrame:
    """Normalize and add glossary prompts."""
    rng = random.Random(seed)
    df = df.copy()
    
    # Normalize source
    df["src"] = df["src"].apply(normalize_transliteration)
    
    # Add glossary prompts
    if CFG.use_glossary and glossary:
        df["src"] = [
            build_glossary_prompt(src, glossary, CFG.glossary_max_items, drop_prob, rng)
            for src in df["src"]
        ]
    
    return df


# ==============================================================================
# Metrics
# ==============================================================================

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


# ==============================================================================
# Callbacks
# ==============================================================================

class LogCallback(TrainerCallback):
    def __init__(self):
        self.epoch = 0
        self.losses = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch = int(state.epoch) if state.epoch else 0
        self.losses = []
        print(f"\n{'='*60}")
        print(f"📊 Epoch {self.epoch + 1}/{args.num_train_epochs}")
        print("=" * 60)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.losses:
            print(f"\n📉 Train Loss: {sum(self.losses)/len(self.losses):.4f}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\n{'─'*40}")
            print(f"📈 Validation")
            print(f"{'─'*40}")
            print(f"   BLEU: {metrics.get('eval_bleu', 0):.2f}")
            print(f"   chrF: {metrics.get('eval_chrf', 0):.2f}")
            print(f"   Geo:  {metrics.get('eval_geo_mean', 0):.2f}")
            print("─" * 40)


class SampleOutputCallback(TrainerCallback):
    """Print sample translations during validation."""
    
    def __init__(self, tokenizer, val_samples: list[str], device):
        self.tokenizer = tokenizer
        self.val_samples = val_samples[:3]  # First 3 samples
        self.device = device

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or not self.val_samples:
            return
        
        model.eval()
        print("\n📝 Sample Translations:")
        
        for i, src in enumerate(self.val_samples):
            inputs = self.tokenizer(src, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=128, num_beams=4)
            
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"   [{i}] {src[:50]}...")
            print(f"       → {translation[:80]}...")


# ==============================================================================
# Main Training
# ==============================================================================

def main():
    print("\n📖 Loading data...")
    
    train_df = load_data(CFG.data_dir / "v5d_train.csv")
    val_df = load_data(CFG.data_dir / "v5d_val.csv")
    
    print(f"   Train: {len(train_df):,}")
    print(f"   Val: {len(val_df):,}")
    
    # Load glossary
    glossary = load_glossary(CFG.data_dir / "v5d_glossary.json")
    print(f"   Glossary: {len(glossary):,} entries")
    
    # Prepare data
    train_df = prepare_data(train_df, glossary, CFG.glossary_drop_prob, CFG.seed)
    val_df = prepare_data(val_df, glossary, 0.0, CFG.seed)  # No drop for val
    
    # Sample prompts
    print("\n📝 Sample prompts:")
    for i in range(min(2, len(train_df))):
        print(f"   [{i}] {train_df.iloc[i]['src'][:100]}...")
    
    # Load model and tokenizer
    print(f"\n🤖 Loading model: {CFG.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name)
    
    print(f"   Tokenizer vocab: {len(tokenizer)}")
    print(f"   Model vocab: {model.config.vocab_size}")
    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Tokenization
    def tokenize_fn(examples):
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
    
    def to_dataset(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(df[["src", "tgt"]])
        return ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"])
    
    print("\n🔧 Tokenizing...")
    train_ds = to_dataset(train_df)
    val_ds = to_dataset(val_df)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    
    # Training arguments
    # MPS doesn't support fp16/bf16 well, use fp32
    use_fp16 = DEVICE.type == "cuda"
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(CFG.output_dir / "checkpoints"),
        num_train_epochs=CFG.epochs,
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size * 2,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        warmup_ratio=CFG.warmup_ratio,
        max_grad_norm=CFG.max_grad_norm,
        fp16=use_fp16,
        bf16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=CFG.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_geo_mean",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=CFG.max_target_length,
        dataloader_num_workers=0 if DEVICE.type == "mps" else 2,  # MPS needs 0
        logging_steps=CFG.logging_steps,
        report_to="none",
        seed=CFG.seed,
    )
    
    # Callbacks
    callbacks = [
        LogCallback(),
        SampleOutputCallback(tokenizer, val_df["src"].tolist(), DEVICE),
        EarlyStoppingCallback(early_stopping_patience=CFG.early_stopping_patience),
    ]
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=callbacks,
    )
    
    # Train
    print("\n🏁 Starting training...")
    trainer.train()
    
    # Save model ONLY (not tokenizer!)
    model_dir = CFG.output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    trainer.model.save_pretrained(str(model_dir))
    
    # Save config for reference
    config_info = {
        "model_name": CFG.model_name,
        "tokenizer_name": CFG.model_name,  # Important: use same name in infer!
        "max_source_length": CFG.max_source_length,
        "max_target_length": CFG.max_target_length,
    }
    with (model_dir / "v5d_config.json").open("w") as f:
        json.dump(config_info, f, indent=2)
    
    print(f"\n💾 Model saved: {model_dir}")
    print(f"   ⚠️ Tokenizer NOT saved - use '{CFG.model_name}' in inference!")
    
    # Final evaluation
    results = trainer.evaluate()
    print(f"\n📈 Final: BLEU={results.get('eval_bleu',0):.2f}, chrF={results.get('eval_chrf',0):.2f}, Geo={results.get('eval_geo_mean',0):.2f}")
    
    # Quick sanity check
    print("\n🔍 Sanity check...")
    model.eval()
    test_input = "um-ma"
    inputs = tokenizer(test_input, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    model = model.to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50, num_beams=4)
    
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Input: '{test_input}'")
    print(f"   Output: '{translation}'")
    
    if not translation or translation == "":
        print("   ⚠️ WARNING: Empty output! Model may not have trained properly.")
    else:
        print("   ✅ Model produces non-empty output")
    
    print(f"\n{'='*60}")
    print("✅ V5d Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
