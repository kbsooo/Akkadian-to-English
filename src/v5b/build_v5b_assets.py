from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Iterable

from tm_utils import build_glossary, save_glossary


def load_rows(path: Path, max_rows: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(row)
    return rows


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)


def write_tm_jsonl(path: Path, rows: Iterable[dict], split: str):
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            rec = {
                "src": row.get("src", ""),
                "tgt": row.get("tgt", ""),
                "oare_id": row.get("oare_id", ""),
                "split": split,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build V5b assets (TM + Glossary)")
    parser.add_argument("--data-dir", type=str, default="data", help="Base data dir")
    parser.add_argument("--v5-dir", type=str, default="data/v5", help="V5 data dir")
    parser.add_argument("--out-dir", type=str, default="data/v5b", help="Output dir")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows for glossary and TM")
    parser.add_argument("--tm-include-val", action="store_true", help="Include val pairs in TM")
    parser.add_argument("--min-src-count", type=int, default=5)
    parser.add_argument("--min-pair-count", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.15)
    parser.add_argument("--max-targets", type=int, default=2)
    parser.add_argument("--min-src-len", type=int, default=2)
    parser.add_argument("--min-tgt-len", type=int, default=2)

    args = parser.parse_args()

    v5_dir = Path(args.v5_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy core datasets for convenience
    copy_if_exists(v5_dir / "v5_sentence_train.csv", out_dir / "v5_sentence_train.csv")
    copy_if_exists(v5_dir / "v5_sentence_val.csv", out_dir / "v5_sentence_val.csv")
    copy_if_exists(v5_dir / "v5_publications_doc_pairs.csv", out_dir / "v5_publications_doc_pairs.csv")

    train_path = v5_dir / "v5_sentence_train.csv"
    val_path = v5_dir / "v5_sentence_val.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")

    print(f"📖 Loading: {train_path}")
    train_rows = load_rows(train_path, max_rows=args.max_rows)

    print("🧠 Building glossary...")
    glossary = build_glossary(
        train_rows,
        min_src_count=args.min_src_count,
        min_pair_count=args.min_pair_count,
        min_score=args.min_score,
        max_targets=args.max_targets,
        min_src_len=args.min_src_len,
        min_tgt_len=args.min_tgt_len,
    )
    glossary_path = out_dir / "v5b_glossary.json"
    save_glossary(glossary_path, glossary)

    print("🧠 Building TM pairs...")
    tm_path = out_dir / "v5b_tm_pairs.jsonl"
    if tm_path.exists():
        tm_path.unlink()
    write_tm_jsonl(tm_path, train_rows, split="train")

    if args.tm_include_val and val_path.exists():
        print(f"📖 Loading: {val_path}")
        val_rows = load_rows(val_path, max_rows=args.max_rows)
        write_tm_jsonl(tm_path, val_rows, split="val")
    else:
        val_rows = []

    stats = {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "glossary_size": len(glossary),
        "tm_pairs": len(train_rows) + len(val_rows),
        "glossary_params": {
            "min_src_count": args.min_src_count,
            "min_pair_count": args.min_pair_count,
            "min_score": args.min_score,
            "max_targets": args.max_targets,
            "min_src_len": args.min_src_len,
            "min_tgt_len": args.min_tgt_len,
        },
    }

    stats_path = out_dir / "v5b_assets_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("✅ V5b assets ready")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
