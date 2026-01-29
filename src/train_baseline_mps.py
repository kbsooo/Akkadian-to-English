#%% [markdown]
# Baseline Training (MPS-optimized)
#
# Usage:
#   uv run python src/train_baseline_mps.py --data-dir src/outputs --tier tier3
#
# This wrapper is tuned for MacBook (M1/M2/M3/M4) with MPS.
# It sets conservative defaults to fit 16GB unified memory.

#%%
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from train_baseline import TrainConfig, train_baseline


#%%

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Baseline Seq2Seq training (MPS)")
    p.add_argument("--data-dir", type=str, default="src/outputs")
    p.add_argument("--tier", type=str, default="tier3", choices=["tier1", "tier2", "tier3"])
    p.add_argument("--model-name", type=str, default="google/byt5-small")
    p.add_argument("--out-dir", type=str, default="src/outputs/baseline_mps")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-source-length", type=int, default=256)
    p.add_argument("--max-target-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--use-tagged", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    return p


def main() -> None:
    # MPS fallback helps if a kernel is missing on Apple Silicon.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine.")

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
    )

    out_dir = Path(args.out_dir) / args.tier

    # MPS: keep fp16/bf16 off for stability.
    train_baseline(
        data_dir=Path(args.data_dir),
        out_dir=out_dir,
        tier=args.tier,
        cfg=cfg,
        use_tagged=args.use_tagged,
        use_fp16=False,
        use_bf16=False,
        use_torch_compile=False,
        gradient_checkpointing=args.gradient_checkpointing,
    )


if __name__ == "__main__":
    main()
