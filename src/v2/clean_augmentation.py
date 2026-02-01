#%% [markdown]
# # V2 Augmentation Cleaner
#
# Removes suspicious augmented pairs based on simple heuristics.
#
# Heuristics (default):
# - length ratio outside [0.25, 4.0]
# - <gap> ratio > 0.2
# - <unk> ratio > 0.2

#%%
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


#%%
def token_len(text: str) -> int:
    return len([t for t in text.strip().split() if t])


def gap_ratio(text: str) -> float:
    toks = [t for t in text.split() if t]
    if not toks:
        return 0.0
    return sum(1 for t in toks if t == "<gap>") / len(toks)


def unk_ratio(text: str) -> float:
    toks = [t for t in text.split() if t]
    if not toks:
        return 0.0
    return sum(1 for t in toks if t == "<unk>") / len(toks)


def is_bad_pair(r: Dict[str, str], min_ratio: float, max_ratio: float, max_gap: float, max_unk: float) -> bool:
    src = r.get("src", "")
    tgt = r.get("tgt", "")
    src_len = token_len(src)
    tgt_len = token_len(tgt)
    if src_len == 0 or tgt_len == 0:
        return True
    ratio = src_len / max(tgt_len, 1)
    if ratio < min_ratio or ratio > max_ratio:
        return True
    if gap_ratio(src) > max_gap:
        return True
    if unk_ratio(src) > max_unk:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Clean augmented dataset")
    parser.add_argument("--input", type=str, default="data/v2/v2_train_augmented.csv")
    parser.add_argument("--output", type=str, default="data/v2/v2_train_augmented_clean.csv")
    parser.add_argument("--min-ratio", type=float, default=0.25)
    parser.add_argument("--max-ratio", type=float, default=4.0)
    parser.add_argument("--max-gap", type=float, default=0.2)
    parser.add_argument("--max-unk", type=float, default=0.2)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input: {input_path}")

    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    kept = []
    removed = 0
    for r in rows:
        if is_bad_pair(r, args.min_ratio, args.max_ratio, args.max_gap, args.max_unk):
            removed += 1
            continue
        kept.append(r)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames or ["oare_id", "src", "tgt"])
        writer.writeheader()
        writer.writerows(kept)

    print(f"✅ Cleaned: {len(kept)} kept, {removed} removed")
    print(f"💾 Saved: {output_path}")


if __name__ == "__main__":
    main()
