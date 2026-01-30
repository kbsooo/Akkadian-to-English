"""
Akkadian V2: Data Integration Pipeline

Loads and normalizes competition training data.

Note: published_texts.csv has URLs in AICC_translation, not actual translations.
      Sentences_Oare has only first_word, not full sentences.
      So we use only train.csv for now.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# Import normalization functions 
import sys
sys.path.insert(0, str(Path(__file__).parent))
from normalize import normalize_transliteration, normalize_translation


# ==============================================================================
# Data Loading
# ==============================================================================

def load_train_csv(data_dir: Path) -> pd.DataFrame:
    """Load and normalize competition training data."""
    path = data_dir / "train.csv"
    df = pd.read_csv(path)
    print(f"📖 train.csv: {len(df)} rows")
    
    # Normalize
    df["src"] = df["transliteration"].apply(normalize_transliteration)
    df["tgt"] = df["translation"].apply(normalize_translation)
    
    # Add oare_id if missing
    if "oare_id" not in df.columns:
        df["oare_id"] = df.index.astype(str)
    
    return df[["oare_id", "src", "tgt"]]


# ==============================================================================
# Processing
# ==============================================================================

def filter_quality(df: pd.DataFrame, min_src_len: int = 5, min_tgt_len: int = 5) -> pd.DataFrame:
    """Filter out low-quality samples."""
    initial = len(df)
    
    # Remove empty/short
    mask = (df["src"].str.len() > min_src_len) & (df["tgt"].str.len() > min_tgt_len)
    df = df[mask].copy()
    
    print(f"   Quality filter: {initial} → {len(df)} rows")
    return df


def split_train_val(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42) -> tuple:
    """Group-based train/val split to prevent data leakage."""
    groups = df["oare_id"].unique().tolist()
    np.random.seed(seed)
    np.random.shuffle(groups)
    
    n_val = max(1, int(len(groups) * val_frac))
    val_groups = set(groups[:n_val])
    
    train_mask = ~df["oare_id"].isin(val_groups)
    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[~train_mask].reset_index(drop=True)
    
    print(f"   Split: {len(train_df)} train, {len(val_df)} val")
    return train_df, val_df


def analyze_charset(df: pd.DataFrame):
    """Check for remaining non-ASCII characters."""
    all_chars = set()
    for text in df["src"]:
        all_chars.update(set(str(text)))
    
    non_ascii = {c for c in all_chars if ord(c) > 127}
    if non_ascii:
        print(f"   ⚠️ Non-ASCII chars remaining: {len(non_ascii)}")
        for c in sorted(non_ascii, key=ord)[:10]:
            print(f"      U+{ord(c):04X}: {repr(c)}")
    else:
        print("   ✅ All ASCII (normalized successfully)")


# ==============================================================================
# Main Pipeline
# ==============================================================================

def build_dataset(
    data_dir: Path,
    output_dir: Path,
    val_frac: float = 0.1,
    seed: int = 42,
):
    """Build normalized training dataset."""
    print("=" * 60)
    print("🔧 Building V2 Dataset")
    print("=" * 60)
    
    # Load train.csv
    df = load_train_csv(data_dir)
    
    # Quality filter
    df = filter_quality(df)
    
    # Character set analysis  
    print("\n📊 Character set analysis:")
    analyze_charset(df)
    
    # Split
    print("\n📊 Splitting data:")
    train_df, val_df = split_train_val(df, val_frac, seed)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "v2_train.csv"
    val_path = output_dir / "v2_val.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"\n✅ Dataset built!")
    print(f"   📁 Train: {train_path} ({len(train_df)} rows)")
    print(f"   📁 Val: {val_path} ({len(val_df)} rows)")
    
    # Sample
    print("\n📝 Sample (first row):")
    print(f"   src: {train_df.iloc[0]['src'][:80]}...")
    print(f"   tgt: {train_df.iloc[0]['tgt'][:80]}...")
    
    return train_df, val_df


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build V2 training dataset")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="data/v2")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    build_dataset(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        val_frac=args.val_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
