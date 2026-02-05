#!/usr/bin/env python3
from __future__ import annotations

"""
V5c ByT5-small training (stable):
- Local training on Apple Silicon (MPS) via `uv run python ...`
- Deterministic tokenizer (ByT5Tokenizer(extra_ids=...)) so train/infer can't drift
- Builds inference assets: TM pairs + global glossary

Example:
  uv run python src/v5c/akkadian_v5c_small_train.py --data-dir data/v5 --out-dir models/v5c-small
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sacrebleu.metrics import BLEU, CHRF
from transformers import (
    AutoModelForSeq2SeqLM,
    ByT5Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

SRC_SPLIT_RE = re.compile(r"[\s\-]+")
TGT_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+")


def tokenize_src(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in SRC_SPLIT_RE.split(str(text)) if t]


def tokenize_tgt(text: str) -> list[str]:
    if not text:
        return []
    return TGT_TOKEN_RE.findall(str(text))


def is_bad_output(s: str) -> bool:
    """Guardrail: treat empty / punctuation-only / dots-only outputs as invalid."""
    if s is None:
        return True
    t = str(s).strip()
    if not t:
        return True
    if t in {"...", "…"}:
        return True
    # punctuation/dot-only (allow letters/numbers to pass)
    if re.fullmatch(r"[.\s…\-–—,;:!?\"'()\[\]{}<>/\\]+", t):
        return True
    return False


def build_glossary(
    src_texts: list[str],
    tgt_texts: list[str],
    *,
    min_src_count: int = 5,
    min_pair_count: int = 2,
    min_score: float = 0.15,
    max_targets: int = 2,
    min_src_len: int = 2,
    min_tgt_len: int = 2,
) -> dict[str, list[str]]:
    src_count: Counter[str] = Counter()
    cooc: dict[str, Counter[str]] = defaultdict(Counter)

    for src, tgt in zip(src_texts, tgt_texts):
        s_tokens = {t for t in tokenize_src(src) if len(t) >= min_src_len}
        t_tokens = {t for t in tokenize_tgt(tgt) if len(t) >= min_tgt_len}
        if not s_tokens or not t_tokens:
            continue
        for s in s_tokens:
            src_count[s] += 1
        for s in s_tokens:
            for t in t_tokens:
                cooc[s][t] += 1

    glossary: dict[str, list[str]] = {}
    for s, total in src_count.items():
        if total < min_src_count:
            continue
        cand = []
        for t, c in cooc[s].items():
            if c < min_pair_count:
                continue
            score = c / total
            if score < min_score:
                continue
            cand.append((score, c, t))
        cand.sort(key=lambda x: (-x[0], -x[1], x[2]))
        if cand:
            glossary[s] = [t for _, _, t in cand[:max_targets]]

    return glossary


def build_glossary_prompt(
    src: str,
    glossary: dict[str, list[str]] | None,
    *,
    max_items: int,
    drop_prob: float,
    rng: random.Random,
) -> str:
    if not glossary:
        return src
    if drop_prob > 0 and rng.random() < drop_prob:
        return src

    items: list[str] = []
    used: set[str] = set()
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


def save_tm_pairs(path: Path, df: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            rec = {"src": row.src, "tgt": row.tgt, "oare_id": getattr(row, "oare_id", "")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


@dataclass
class TrainArgs:
    data_dir: Path
    out_dir: Path
    epochs: int
    lr: float
    batch_size: int
    grad_accum: int
    max_source_len: int
    max_target_len: int
    seed: int
    disable_glossary: bool
    glossary_max_items: int
    glossary_drop_train: float
    glossary_drop_eval: float
    warmup_ratio: float
    weight_decay: float
    gradient_checkpointing: bool
    dataloader_num_workers: int
    use_publications: bool
    stage_a_epochs: int
    stage_a_lr: float


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/v5"))
    p.add_argument("--out-dir", type=Path, default=Path("models/v5c-small"))
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-source-len", type=int, default=256)
    p.add_argument("--max-target-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disable-glossary", action="store_true")
    p.add_argument("--glossary-max-items", type=int, default=8)
    p.add_argument("--glossary-drop-train", type=float, default=0.5)
    p.add_argument("--glossary-drop-eval", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--dataloader-num-workers", type=int, default=0)

    # Optional Stage A (publications doc-level)
    p.add_argument("--use-publications", action="store_true", default=False)
    p.add_argument("--stage-a-epochs", type=int, default=1)
    p.add_argument("--stage-a-lr", type=float, default=5e-5)
    a = p.parse_args()
    return TrainArgs(
        data_dir=a.data_dir,
        out_dir=a.out_dir,
        epochs=a.epochs,
        lr=a.lr,
        batch_size=a.batch_size,
        grad_accum=a.grad_accum,
        max_source_len=a.max_source_len,
        max_target_len=a.max_target_len,
        seed=a.seed,
        disable_glossary=a.disable_glossary,
        glossary_max_items=a.glossary_max_items,
        glossary_drop_train=a.glossary_drop_train,
        glossary_drop_eval=a.glossary_drop_eval,
        warmup_ratio=a.warmup_ratio,
        weight_decay=a.weight_decay,
        gradient_checkpointing=a.gradient_checkpointing,
        dataloader_num_workers=a.dataloader_num_workers,
        use_publications=a.use_publications,
        stage_a_epochs=a.stage_a_epochs,
        stage_a_lr=a.stage_a_lr,
    )


def main() -> None:
    # MPS fallback for unsupported ops
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_dir / "model"
    assets_dir = args.out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    device = resolve_device()
    print("=" * 60)
    print("🚀 V5c ByT5-small TRAIN")
    print("=" * 60)
    print(f"📁 Data: {args.data_dir}")
    print(f"📁 Out:  {args.out_dir}")
    print(f"🧠 Device: {device}")
    print(f"🔧 MPS available: {torch.backends.mps.is_available()}")
    print("=" * 60)

    train_path = args.data_dir / "v5_sentence_train.csv"
    val_path = args.data_dir / "v5_sentence_val.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Missing v5_sentence_train/val.csv under data-dir.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    for df in (train_df, val_df):
        if not {"src", "tgt"}.issubset(df.columns):
            raise ValueError("CSV must contain src,tgt columns")
        df.dropna(subset=["src", "tgt"], inplace=True)
        df["src"] = df["src"].astype(str)
        df["tgt"] = df["tgt"].astype(str)

    print(f"📊 Train pairs: {len(train_df):,}, Val pairs: {len(val_df):,}")

    # Assets (always build)
    print("🧱 Building assets: TM pairs + global glossary...")
    save_tm_pairs(assets_dir / "v5c_tm_pairs.jsonl", train_df[["oare_id", "src", "tgt"]] if "oare_id" in train_df.columns else train_df[["src", "tgt"]])

    glossary = build_glossary(train_df["src"].tolist(), train_df["tgt"].tolist())
    save_json(assets_dir / "v5c_glossary.json", glossary)

    save_json(
        assets_dir / "v5c_assets_stats.json",
        {
            "train_pairs": int(len(train_df)),
            "val_pairs": int(len(val_df)),
            "glossary_size": int(len(glossary)),
        },
    )
    print(f"   ✅ glossary_size={len(glossary):,}")

    # Model (download ok on local)
    model_name = "google/byt5-small"
    print(f"🤖 Loading base model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("   ✅ gradient_checkpointing enabled")

    # Deterministic tokenizer (no network)
    extra_ids = int(model.config.vocab_size) - 259
    if extra_ids < 0:
        raise ValueError(f"Unexpected vocab_size={model.config.vocab_size} for ByT5")
    tokenizer = ByT5Tokenizer(extra_ids=extra_ids)
    assert len(tokenizer) == model.config.vocab_size, "Tokenizer/model vocab mismatch!"
    print(f"🔤 Tokenizer vocab: {len(tokenizer)} (extra_ids={extra_ids})")

    # Glossary prompt augmentation
    if not args.disable_glossary and glossary:
        print("🧠 Applying glossary prompts...")
        rng_train = random.Random(args.seed)
        rng_val = random.Random(args.seed)
        train_df["src_aug"] = [
            build_glossary_prompt(
                s,
                glossary,
                max_items=args.glossary_max_items,
                drop_prob=args.glossary_drop_train,
                rng=rng_train,
            )
            for s in train_df["src"].tolist()
        ]
        val_df["src_aug"] = [
            build_glossary_prompt(
                s,
                glossary,
                max_items=args.glossary_max_items,
                drop_prob=args.glossary_drop_eval,
                rng=rng_val,
            )
            for s in val_df["src"].tolist()
        ]
        src_col = "src_aug"
    else:
        print("🧠 Glossary prompts disabled")
        src_col = "src"

    def tokenize_fn(examples):
        model_inputs = tokenizer(
            examples[src_col],
            max_length=args.max_source_len,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            examples["tgt"],
            max_length=args.max_target_len,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    cols = [src_col, "tgt"]
    train_ds = Dataset.from_pandas(train_df[cols]).map(tokenize_fn, batched=True, remove_columns=cols)
    val_ds = Dataset.from_pandas(val_df[cols]).map(tokenize_fn, batched=True, remove_columns=cols)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    # Warmup steps (avoid warmup_ratio deprecation)
    steps_per_epoch = int(np.ceil(len(train_df) / args.batch_size / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    train_kwargs: dict[str, Any] = dict(
        output_dir=str(args.out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_geo_mean",
        greater_is_better=True,
        predict_with_generate=True,
        generation_num_beams=4,
        generation_max_length=args.max_target_len,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=50,
    )

    # Prefer MPS when available (best-effort, depends on transformers version)
    if torch.backends.mps.is_available():
        train_kwargs["use_mps_device"] = True

    try:
        training_args = Seq2SeqTrainingArguments(**train_kwargs)
    except TypeError:
        # Compatibility with transformers arg renames
        if "evaluation_strategy" in train_kwargs:
            train_kwargs["eval_strategy"] = train_kwargs.pop("evaluation_strategy")
        train_kwargs.pop("use_mps_device", None)
        training_args = Seq2SeqTrainingArguments(**train_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    trainer.train()

    # Collapse/empty guard check (sample a few val rows)
    sample_n = min(3, len(val_df))
    if sample_n > 0:
        print("🧪 Sanity check: generating on val samples (non-empty guarantee check)")
        sample_srcs = val_df[src_col].tolist()[:sample_n]
        inputs = tokenizer(sample_srcs, return_tensors="pt", padding=True, truncation=True, max_length=args.max_source_len)
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                num_beams=4,
                max_new_tokens=128,
                min_new_tokens=8,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        bad = [d for d in decoded if is_bad_output(d)]
        for i, d in enumerate(decoded):
            print(f"   [{i}] {d[:160]}")
        if bad:
            raise RuntimeError(
                f"Sanity check failed: {len(bad)}/{len(decoded)} bad outputs detected (empty/dots-only). "
                "Refusing to write final model/ artifacts."
            )
        else:
            print("✅ Sanity check passed: no bad outputs.")

    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    # Save tokenizer optionally, but do not depend on it at inference-time.
    tokenizer.save_pretrained(str(model_dir))
    print(f"💾 Saved model: {model_dir}")

    final_metrics = trainer.evaluate()
    print("📈 Final metrics:", {k: float(v) for k, v in final_metrics.items() if k.startswith("eval_")})
    print("✅ V5c training complete")


if __name__ == "__main__":
    main()
