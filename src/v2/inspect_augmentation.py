#%% [markdown]
# # V2 Augmentation Audit
#
# Quick heuristics to flag potentially misaligned pairs in v2_train_augmented.csv.

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


def audit_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    flagged = []
    for idx, r in enumerate(rows):
        src = r.get("src", "")
        tgt = r.get("tgt", "")
        src_len = token_len(src)
        tgt_len = token_len(tgt)
        if src_len == 0 or tgt_len == 0:
            continue
        ratio = src_len / max(tgt_len, 1)
        g_ratio = gap_ratio(src)
        u_ratio = unk_ratio(src)

        flags = []
        if ratio < 0.25 or ratio > 4.0:
            flags.append("len_ratio")
        if g_ratio > 0.2:
            flags.append("gap")
        if u_ratio > 0.2:
            flags.append("unk")

        if flags:
            flagged.append(
                {
                    "idx": str(idx),
                    "oare_id": r.get("oare_id", ""),
                    "src": src[:300],
                    "tgt": tgt[:300],
                    "src_len": str(src_len),
                    "tgt_len": str(tgt_len),
                    "len_ratio": f"{ratio:.2f}",
                    "gap_ratio": f"{g_ratio:.2f}",
                    "unk_ratio": f"{u_ratio:.2f}",
                    "flags": ",".join(flags),
                }
            )

    return flagged


#%%
def main():
    parser = argparse.ArgumentParser(description="Audit v2_train_augmented.csv for misalignment")
    parser.add_argument("--input", type=str, default="data/v2/v2_train_augmented.csv")
    parser.add_argument("--output", type=str, default="data/v2/augmentation_audit.csv")
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input: {input_path}")

    flagged = audit_rows(input_path)
    print(f"⚠️ Flagged rows (total): {len(flagged)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "oare_id",
                "src",
                "tgt",
                "src_len",
                "tgt_len",
                "len_ratio",
                "gap_ratio",
                "unk_ratio",
                "flags",
            ],
        )
        writer.writeheader()
        writer.writerows(flagged[: args.max_samples])

    print(f"✅ Audit saved: {output_path} (top {min(len(flagged), args.max_samples)})")


if __name__ == "__main__":
    main()
