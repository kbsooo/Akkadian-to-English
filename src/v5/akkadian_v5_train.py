#%% [markdown]
# # Akkadian V5 Training (2-stage)
#
# Stage A: Publications English doc-level (optional)
# Stage B: Sentence-level main training
#
# Data input: data/v5 or Kaggle input containing v5_* files

#%% [markdown]
# ## 1. Imports & Configuration

#%%
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

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
    set_seed,
)


#%%
@dataclass
class Config:
    model_size: str = "base"  # "base" or "large"
    data_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    # Stage A (publications)
    use_publications_stage: bool = True
    stage_a_epochs: int = 2
    stage_a_lr: float = 5e-5

    # Stage B (sentence-level)
    stage_b_epochs: int = 8
    stage_b_lr: float = 1e-4

    # Sequence lengths
    max_source_length: int = 256
    max_target_length: int = 256

    # Training
    seed: int = 42
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Hardware
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2

    # Model-specific
    model_name: str = field(init=False)

    def __post_init__(self):
        if self.model_size == "base":
            self.model_name = "google/byt5-base"
            if self.output_dir is None:
                self.output_dir = Path("/content/drive/MyDrive/akkadian/v5-base")
        else:
            self.model_name = "google/byt5-large"
            if self.output_dir is None:
                self.output_dir = Path("/content/drive/MyDrive/akkadian/v5-large")


def resolve_data_dir() -> Path:
    env = os.environ.get("V5_DATA_DIR")
    if env:
        p = Path(env)
        if p.exists():
            return p

    local = Path("data/v5")
    if local.exists():
        return local

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for d in kaggle_input.iterdir():
            if (d / "v5_sentence_train.csv").exists():
                return d

    raise FileNotFoundError("V5 data directory not found. Set V5_DATA_DIR or place data/v5.")


CFG = Config(model_size="base")
CFG.data_dir = resolve_data_dir()
CFG.output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print(f"🚀 Akkadian V5 Training: {CFG.model_size.upper()}")
print("=" * 60)
print(f"📁 Data: {CFG.data_dir}")
print(f"📁 Output: {CFG.output_dir}")
print(f"🤖 Model: {CFG.model_name}")
print(f"🎮 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

set_seed(CFG.seed)


#%% [markdown]
# ## 2. Helpers

#%%
def load_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"src", "tgt"}.issubset(df.columns):
        raise ValueError(f"Missing src/tgt columns: {path}")
    df = df.dropna(subset=["src", "tgt"]).reset_index(drop=True)
    return df


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


class LogCallback(TrainerCallback):
    def __init__(self, label: str):
        self.label = label
        self.epoch = 0
        self.losses = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch = int(state.epoch) if state.epoch else 0
        self.losses = []
        print(f"\n{'='*60}\n📊 {self.label} Epoch {self.epoch + 1}/{args.num_train_epochs}\n{'='*60}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.losses:
            print(f"\n📉 {self.label} Train Loss: {sum(self.losses)/len(self.losses):.4f}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\n{'─'*40}\n📈 Validation ({self.label})\n{'─'*40}")
            print(f"   BLEU: {metrics.get('eval_bleu', 0):.2f}")
            print(f"   chrF: {metrics.get('eval_chrf', 0):.2f}")
            print(f"   Geo:  {metrics.get('eval_geo_mean', 0):.2f}\n{'─'*40}")


#%% [markdown]
# ## 3. Load Data

#%%
print("📖 Loading V5 datasets...")

sentence_train_path = CFG.data_dir / "v5_sentence_train.csv"
sentence_val_path = CFG.data_dir / "v5_sentence_val.csv"

if not sentence_train_path.exists() or not sentence_val_path.exists():
    raise FileNotFoundError("v5_sentence_train/val.csv not found in data dir")

sent_train_df = load_pairs(sentence_train_path)
sent_val_df = load_pairs(sentence_val_path)

pub_pairs_path = CFG.data_dir / "v5_publications_doc_pairs.csv"
pub_df = load_pairs(pub_pairs_path) if pub_pairs_path.exists() else None

print(f"   Sentence train: {len(sent_train_df):,}")
print(f"   Sentence val: {len(sent_val_df):,}")
if pub_df is not None:
    print(f"   Publications doc pairs: {len(pub_df):,}")
else:
    print("   Publications doc pairs: not found")


#%% [markdown]
# ## 4. Model Setup

#%%
print(f"\n🤖 Loading model: {CFG.model_name}")

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name)

print(f"   Tokenizer: {len(tokenizer)}, Model vocab: {model.config.vocab_size}")
print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")

if CFG.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    print("   ✅ Gradient checkpointing enabled")


#%% [markdown]
# ## 5. Tokenization

#%%

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


#%% [markdown]
# ## 6. Stage A: Publications Doc-Level (optional)

#%%
if CFG.use_publications_stage and pub_df is not None and len(pub_df) > 0:
    print("\n🏁 Stage A: Publications doc-level")
    pub_train_ds = to_dataset(pub_df)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    stage_a_args = dict(
        output_dir=str(CFG.output_dir / "stage_a_checkpoints"),
        num_train_epochs=CFG.stage_a_epochs,
        per_device_train_batch_size=CFG.batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        learning_rate=CFG.stage_a_lr,
        weight_decay=CFG.weight_decay,
        warmup_ratio=CFG.warmup_ratio,
        max_grad_norm=CFG.max_grad_norm,
        fp16=CFG.fp16,
        bf16=CFG.bf16,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        predict_with_generate=False,
        dataloader_num_workers=CFG.dataloader_num_workers,
        logging_steps=50,
        report_to="none",
        seed=CFG.seed,
    )

    try:
        training_args = Seq2SeqTrainingArguments(**stage_a_args)
    except TypeError:
        stage_a_args["eval_strategy"] = stage_a_args.pop("evaluation_strategy")
        training_args = Seq2SeqTrainingArguments(**stage_a_args)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=pub_train_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCallback("Stage A")],
    )

    trainer.train()
else:
    print("\n⏭️  Stage A skipped (no publications data or disabled)")


#%% [markdown]
# ## 7. Stage B: Sentence-Level Main Training

#%%
print("\n🏁 Stage B: Sentence-level training")

sent_train_ds = to_dataset(sent_train_df)
sent_val_ds = to_dataset(sent_val_df)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

stage_b_args = dict(
    output_dir=str(CFG.output_dir / "stage_b_checkpoints"),
    num_train_epochs=CFG.stage_b_epochs,
    per_device_train_batch_size=CFG.batch_size,
    per_device_eval_batch_size=CFG.batch_size * 2,
    gradient_accumulation_steps=CFG.gradient_accumulation_steps,
    learning_rate=CFG.stage_b_lr,
    weight_decay=CFG.weight_decay,
    warmup_ratio=CFG.warmup_ratio,
    max_grad_norm=CFG.max_grad_norm,
    fp16=CFG.fp16,
    bf16=CFG.bf16,
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
    report_to="none",
    seed=CFG.seed,
)

try:
    training_args = Seq2SeqTrainingArguments(**stage_b_args)
except TypeError:
    stage_b_args["eval_strategy"] = stage_b_args.pop("evaluation_strategy")
    training_args = Seq2SeqTrainingArguments(**stage_b_args)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=sent_train_ds,
    eval_dataset=sent_val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=build_compute_metrics(tokenizer),
    callbacks=[LogCallback("Stage B")],
)

trainer.train()

#%% [markdown]
# ## 8. Save Model

#%%
model_dir = CFG.output_dir / "model"
trainer.save_model(str(model_dir))
tokenizer.save_pretrained(str(model_dir))
print(f"\n💾 Saved: {model_dir}")

results = trainer.evaluate()
print(f"\n📈 Final: BLEU={results.get('eval_bleu',0):.2f}, chrF={results.get('eval_chrf',0):.2f}, Geo={results.get('eval_geo_mean',0):.2f}")

print(f"\n{'='*60}\n✅ V5 Training Complete!\n{'='*60}")
