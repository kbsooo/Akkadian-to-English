#%% [markdown]
# # V2 Sentence-Level Dataset Builder
#
# Builds train/val CSVs from sentence pairs (tiered outputs).
# Uses the unified V2 normalization to align Train/Test styles.

#%%
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Local normalization
sys.path.insert(0, str(Path(__file__).parent))
from normalize import normalize_transliteration, normalize_translation


#%%
def infer_columns(header: List[str]) -> Tuple[str, str]:
    src_candidates = ["src_norm", "src_tagged", "transliteration", "src"]
    tgt_candidates = ["tgt_norm", "translation", "tgt"]

    src_col = next((c for c in src_candidates if c in header), None)
    tgt_col = next((c for c in tgt_candidates if c in header), None)

    if not src_col or not tgt_col:
        raise ValueError(f"Could not infer src/tgt columns from header: {header}")
    return src_col, tgt_col


def load_rows(path: Path) -> Tuple[List[Dict[str, str]], str, str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        src_col, tgt_col = infer_columns(header)
        id_col = "oare_id" if "oare_id" in header else ("id" if "id" in header else None)
        rows = list(reader)
    return rows, src_col, tgt_col, id_col or ""


def filter_rows(
    rows: Iterable[Dict[str, str]],
    src_col: str,
    tgt_col: str,
    min_src_len: int,
    min_tgt_len: int,
    min_quality: float | None,
) -> List[Dict[str, str]]:
    filtered = []
    for r in rows:
        if min_quality is not None and "quality_score" in r:
            try:
                if float(r["quality_score"]) < min_quality:
                    continue
            except ValueError:
                continue

        src = normalize_transliteration(r.get(src_col, ""))
        tgt = normalize_translation(r.get(tgt_col, ""))
        if len(src) < min_src_len or len(tgt) < min_tgt_len:
            continue

        r["_src"] = src
        r["_tgt"] = tgt
        filtered.append(r)
    return filtered


def split_train_val(rows: List[Dict[str, str]], id_col: str, val_frac: float, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not id_col:
        random.Random(seed).shuffle(rows)
        n_val = max(1, int(len(rows) * val_frac))
        return rows[n_val:], rows[:n_val]

    ids = sorted({r.get(id_col, "") for r in rows})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val])

    train_rows = [r for r in rows if r.get(id_col, "") not in val_ids]
    val_rows = [r for r in rows if r.get(id_col, "") in val_ids]
    return train_rows, val_rows


def write_csv(path: Path, rows: List[Dict[str, str]], id_col: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if id_col:
            writer.writerow(["oare_id", "src", "tgt"])
            for r in rows:
                writer.writerow([r.get(id_col, ""), r["_src"], r["_tgt"]])
        else:
            writer.writerow(["src", "tgt"])
            for r in rows:
                writer.writerow([r["_src"], r["_tgt"]])


#%%
def main():
    parser = argparse.ArgumentParser(description="Build sentence-level dataset for V2")
    parser.add_argument("--input", type=str, default="src/outputs/sentence_pairs_q70_pattern.csv")
    parser.add_argument("--output-dir", type=str, default="data/v2")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-src-len", type=int, default=5)
    parser.add_argument("--min-tgt-len", type=int, default=5)
    parser.add_argument("--min-quality", type=float, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows, src_col, tgt_col, id_col = load_rows(input_path)
    print(f"📖 Loaded: {len(rows)} rows from {input_path}")
    print(f"   src_col={src_col}, tgt_col={tgt_col}, id_col={id_col or 'None'}")

    rows = filter_rows(
        rows,
        src_col=src_col,
        tgt_col=tgt_col,
        min_src_len=args.min_src_len,
        min_tgt_len=args.min_tgt_len,
        min_quality=args.min_quality,
    )
    print(f"✅ After filtering: {len(rows)} rows")

    train_rows, val_rows = split_train_val(rows, id_col, args.val_frac, args.seed)
    print(f"📊 Split: train={len(train_rows)}, val={len(val_rows)}")

    out_dir = Path(args.output_dir)
    train_path = out_dir / "v2_sentence_train.csv"
    val_path = out_dir / "v2_sentence_val.csv"

    write_csv(train_path, train_rows, id_col)
    write_csv(val_path, val_rows, id_col)

    print(f"💾 Saved: {train_path}")
    print(f"💾 Saved: {val_path}")


if __name__ == "__main__":
    main()
