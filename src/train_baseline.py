#%% [markdown]
# Baseline Training (Seq2Seq)
#
# Usage (Tier3 default):
#   uv run python src/train_baseline.py --data-dir src/outputs --tier tier3
#
# Multi-GPU (2x T4 on Kaggle):
#   torchrun --nproc_per_node=2 src/train_baseline.py --data-dir src/outputs --tier tier3
#   # or
#   accelerate launch src/train_baseline.py --data-dir src/outputs --tier tier3
#
# Kaggle defaults:
# - data dir: /kaggle/working/outputs if present, else /kaggle/input/*/outputs
# - out dir:  /kaggle/working/baseline
#
# Optional:
#   --model-name google/byt5-base
#   --out-dir src/outputs/baseline_tier3
#   --epochs 5 --batch-size 4 --lr 3e-4
#   --max-source-length 256 --max-target-length 256
#
# Notes:
# - Character-level ByT5 is robust to rare diacritics and OCR noise.
# - Uses grouped split by oare_id to reduce leakage.

#%%
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
import inspect

#%%
# -----------------------------
# Config
# -----------------------------


@dataclass
class TrainConfig:
    model_name: str = "google/byt5-base"
    seed: int = 42
    val_frac: float = 0.1
    max_source_length: int = 256
    max_target_length: int = 256
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    predict_with_generate: bool = True
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 2
    save_total_limit: int = 2


#%%
# -----------------------------
# Data loading
# -----------------------------


def load_tier(data_dir: Path, tier: str) -> pd.DataFrame:
    tier_map = {
        "tier1": "sentence_pairs_valid.csv",
        "tier2": "sentence_pairs_q70.csv",
        "tier3": "sentence_pairs_q70_pattern.csv",
    }
    fname = tier_map.get(tier)
    if not fname:
        raise ValueError(f"Unknown tier: {tier}")
    path = data_dir / fname
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _is_kaggle() -> bool:
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def _default_data_dir() -> Path:
    # Kaggle: prefer /kaggle/working/outputs, else search /kaggle/input/*/outputs
    if _is_kaggle():
        working = Path("/kaggle/working/outputs")
        if working.exists():
            return working
        input_root = Path("/kaggle/input")
        if input_root.exists():
            for p in input_root.iterdir():
                cand = p / "outputs"
                if cand.exists():
                    return cand
        return Path("/kaggle/working")
    return Path("src/outputs")


def _default_out_dir() -> Path:
    return Path("/kaggle/working/baseline") if _is_kaggle() else Path("src/outputs/baseline")


def group_split(df: pd.DataFrame, group_col: str, val_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    groups = df[group_col].unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * val_frac))
    val_groups = set(groups[:n_val])

    train_df = df[~df[group_col].isin(val_groups)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_groups)].reset_index(drop=True)
    return train_df, val_df


#%%
# -----------------------------
# Tokenization
# -----------------------------


def build_datasets(
    df: pd.DataFrame,
    tokenizer,
    cfg: TrainConfig,
    use_tagged: bool,
) -> Tuple[Dataset, Dataset]:
    src_col = "src_tagged" if use_tagged and "src_tagged" in df.columns else "src_norm"
    tgt_col = "tgt_norm"

    # Shape safety: required columns
    assert src_col in df.columns, f"Missing {src_col}"
    assert tgt_col in df.columns, f"Missing {tgt_col}"

    train_df, val_df = group_split(df, "oare_id", cfg.val_frac, cfg.seed)

    train_ds = Dataset.from_pandas(train_df[[src_col, tgt_col]])
    val_ds = Dataset.from_pandas(val_df[[src_col, tgt_col]])

    def tokenize(batch):
        # Insight: keep consistent max lengths for stable batching.
        model_inputs = tokenizer(
            batch[src_col],
            max_length=cfg.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch[tgt_col],
            max_length=cfg.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]

        # Shape safety: enforce maximum lengths
        for ids in model_inputs["input_ids"]:
            assert len(ids) <= cfg.max_source_length
        for ids in model_inputs["labels"]:
            assert len(ids) <= cfg.max_target_length

        return model_inputs

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=[src_col, tgt_col])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=[src_col, tgt_col])

    return train_ds, val_ds


#%%
# -----------------------------
# Metrics
# -----------------------------


def build_metrics(tokenizer):
    bleu = BLEU()
    chrf = CHRF(word_order=2)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds == -100, tokenizer.pad_token_id, preds)
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu_score = bleu.corpus_score(decoded_preds, [decoded_labels]).score
        chrf_score = chrf.corpus_score(decoded_preds, [decoded_labels]).score
        score = math.sqrt(max(0.0, bleu_score) * max(0.0, chrf_score))

        return {
            "bleu": bleu_score,
            "chrf": chrf_score,
            "score": score,
        }

    return compute_metrics


#%%
# -----------------------------
# Training
# -----------------------------


def train_baseline(
    data_dir: Path,
    out_dir: Path,
    tier: str,
    cfg: TrainConfig,
    use_tagged: bool,
    use_fp16: bool,
    use_bf16: bool,
    use_torch_compile: bool,
    gradient_checkpointing: bool,
) -> None:
    set_seed(cfg.seed)

    df = load_tier(data_dir, tier)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    # Insight: torch.compile can speed up training on supported backends.
    if use_torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1 and "LOCAL_RANK" not in os.environ:
            print(f"[Info] Detected {n_gpus} GPUs. For full DDP, run with:")
            print("  torchrun --nproc_per_node=2 src/train_baseline.py ...")
            print("  or accelerate launch src/train_baseline.py ...")

    train_ds, val_ds = build_datasets(df, tokenizer, cfg, use_tagged)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Handle HF arg name changes (evaluation_strategy -> eval_strategy)
    arg_sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    eval_key = "evaluation_strategy" if "evaluation_strategy" in arg_sig.parameters else "eval_strategy"

    # Warmup steps from ratio to avoid deprecated warmup_ratio
    steps_per_epoch = math.ceil(len(train_ds) / max(1, cfg.batch_size))
    steps_per_epoch = math.ceil(steps_per_epoch / max(1, cfg.gradient_accumulation_steps))
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    args_kwargs = dict(
        output_dir=str(out_dir),
        save_strategy=cfg.save_strategy,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        predict_with_generate=cfg.predict_with_generate,
        logging_steps=20,
        save_total_limit=cfg.save_total_limit,
        fp16=use_fp16,
        bf16=use_bf16,
        load_best_model_at_end=True,
        metric_for_best_model="score",
        greater_is_better=True,
        report_to="none",
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=cfg.dataloader_num_workers,
    )
    args_kwargs[eval_key] = cfg.eval_strategy

    # Multi-GPU safe settings when available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if "ddp_find_unused_parameters" in arg_sig.parameters:
            args_kwargs["ddp_find_unused_parameters"] = False

    args = Seq2SeqTrainingArguments(**args_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=build_metrics(tokenizer),
    )

    # Transformers API compatibility: tokenizer arg was removed in newer versions.
    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainer.train()
    metrics = trainer.evaluate()

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Visual proof: training curves
    log_history = trainer.state.log_history
    steps, train_loss = [], []
    eval_steps, eval_loss, eval_bleu, eval_chrf, eval_score = [], [], [], [], []

    for entry in log_history:
        if "loss" in entry and "epoch" in entry and "eval_loss" not in entry:
            steps.append(entry.get("step", len(steps)))
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", len(eval_steps)))
            eval_loss.append(entry.get("eval_loss"))
            eval_bleu.append(entry.get("eval_bleu"))
            eval_chrf.append(entry.get("eval_chrf"))
            eval_score.append(entry.get("eval_score"))

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(steps, train_loss, label="train_loss")
        axes[0].plot(eval_steps, eval_loss, label="eval_loss")
        axes[0].set_title("Loss Curves")
        axes[0].set_xlabel("step")
        axes[0].legend()

        axes[1].plot(eval_steps, eval_bleu, label="BLEU")
        axes[1].plot(eval_steps, eval_chrf, label="chrF")
        axes[1].plot(eval_steps, eval_score, label="GeoMean")
        axes[1].set_title("Eval Metrics")
        axes[1].set_xlabel("step")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(out_dir / "training_curves.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot skipped: {exc}")


#%%
# -----------------------------
# CLI
# -----------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Baseline Seq2Seq training")
    p.add_argument("--data-dir", type=str, default=str(_default_data_dir()))
    p.add_argument("--tier", type=str, default="tier3", choices=["tier1", "tier2", "tier3"])
    p.add_argument("--model-name", type=str, default="google/byt5-base")
    p.add_argument("--out-dir", type=str, default=str(_default_out_dir()))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-source-length", type=int, default=256)
    p.add_argument("--max-target-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--use-tagged", action="store_true")
    p.add_argument("--no-auto-fp16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--torch-compile", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--save-total-limit", type=int, default=2)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    cfg = TrainConfig(
        model_name=args.model_name,
        seed=args.seed,
        val_frac=args.val_frac,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        epochs=args.epochs,
        lr=args.lr,
        dataloader_num_workers=args.num_workers,
        save_total_limit=args.save_total_limit,
    )

    out_dir = Path(args.out_dir) / args.tier

    auto_fp16 = not args.no_auto_fp16
    use_fp16 = args.fp16
    if auto_fp16 and torch.cuda.is_available() and not args.bf16:
        use_fp16 = True

    train_baseline(
        data_dir=Path(args.data_dir),
        out_dir=out_dir,
        tier=args.tier,
        cfg=cfg,
        use_tagged=args.use_tagged,
        use_fp16=use_fp16,
        use_bf16=args.bf16,
        use_torch_compile=args.torch_compile,
        gradient_checkpointing=args.gradient_checkpointing,
    )


if __name__ == "__main__":
    main()
