#%% [markdown]
# Deep Past Initiative MT - EDA
#
# Dependencies (suggested latest):
#   python -m pip install -U pandas numpy matplotlib
#
# Notes:
# - Uses #%% cells for Jupytext.
# - Plots are saved under GPT/figs and also shown inline.

#%%
from __future__ import annotations

from pathlib import Path
from collections import Counter
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Insight: new Matplotlib color cycle (3.10+) improves accessibility; fallback if unavailable.
if "petroff10" in plt.style.available:
    plt.style.use("petroff10")

DATA_DIR = Path("data")
FIG_DIR = Path("GPT") / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    """
    Try pyarrow-backed strings if available; fall back to default.
    This reduces memory and gives better string ops on large text columns.
    """
    try:
        return pd.read_csv(path, dtype_backend="pyarrow", keep_default_na=False)
    except Exception:
        return pd.read_csv(path, keep_default_na=False)


train = read_csv(DATA_DIR / "train.csv")
test = read_csv(DATA_DIR / "test.csv")
sample_sub = read_csv(DATA_DIR / "sample_submission.csv")

# Shape safety: ensure the competition schema did not drift.
assert train.shape[1] == 3, f"train columns: {train.columns.tolist()}"
assert test.shape[1] == 5, f"test columns: {test.columns.tolist()}"
assert sample_sub.shape[1] == 2, f"sample_sub columns: {sample_sub.columns.tolist()}"
assert set(train.columns) == {"oare_id", "transliteration", "translation"}
assert set(test.columns) == {"id", "text_id", "line_start", "line_end", "transliteration"}
assert set(sample_sub.columns) == {"id", "translation"}

#%%
# Quick peek
print("train shape", train.shape)
print("test shape", test.shape)
print("sample_sub shape", sample_sub.shape)
print(train.head(3))
print(test.head(3))

#%%
# Basic missingness check
missing = pd.DataFrame(
    {
        "train_missing": train.isna().sum(),
        "test_missing": test.isna().sum(),
    }
)
print(missing)

#%%
# Add basic text stats

def add_text_stats(df: pd.DataFrame, col: str, prefix: str) -> None:
    # Token split is whitespace-only to keep logic stable across unicode.
    df[f"{prefix}char_len"] = df[col].str.len()
    df[f"{prefix}tok_len"] = df[col].str.split().str.len()


add_text_stats(train, "transliteration", "src_")
add_text_stats(train, "translation", "tgt_")
train["tgt_per_src_tok"] = train["tgt_tok_len"] / train["src_tok_len"].replace(0, np.nan)

#%%
# Summary stats helper

def describe_lengths(series: pd.Series) -> pd.Series:
    return series.describe(percentiles=[0.1, 0.5, 0.9])


print("src token length summary")
print(describe_lengths(train["src_tok_len"]))
print("tgt token length summary")
print(describe_lengths(train["tgt_tok_len"]))
print("tgt/src token ratio summary")
print(describe_lengths(train["tgt_per_src_tok"]))

#%%
# Visual proof: length distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(train["src_tok_len"], bins=40)
axes[0].set_title("Src token length")
axes[0].set_xlabel("tokens")

axes[1].hist(train["tgt_tok_len"], bins=40)
axes[1].set_title("Tgt token length")
axes[1].set_xlabel("tokens")

axes[2].hist(train["tgt_per_src_tok"].dropna(), bins=40)
axes[2].set_title("Tgt/src token ratio")
axes[2].set_xlabel("ratio")

fig.tight_layout()
fig.savefig(FIG_DIR / "length_distributions.png", dpi=150)
plt.show()

#%%
# Visual proof: alignment relationship
plt.figure(figsize=(5, 4))
plt.scatter(train["src_tok_len"], train["tgt_tok_len"], s=6, alpha=0.4)
plt.xlabel("src tokens")
plt.ylabel("tgt tokens")
plt.title("Token length alignment")
plt.tight_layout()
plt.savefig(FIG_DIR / "token_alignment.png", dpi=150)
plt.show()

#%%
# Marker and notation diagnostics (heuristics)
MARKERS = {
    "curly_braces": r"[{}]",  # determinatives in transliteration
    "square_brackets": r"[\[\]]",  # broken signs
    "angle_brackets": r"[<>]",  # scribal insertions
    "gap_token": r"<gap>",
    "big_gap_token": r"<big_gap>",
}

marker_counts = {}
for name, pattern in MARKERS.items():
    marker_counts[name] = train["transliteration"].str.contains(pattern, regex=True).sum()

print("marker_counts", marker_counts)

#%%
# Non-ASCII character rate (proxy for diacritics)

def non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    non_ascii = sum(ord(ch) > 127 for ch in text)
    return non_ascii / max(1, len(text))


train["src_non_ascii_ratio"] = train["transliteration"].map(non_ascii_ratio)
train["tgt_non_ascii_ratio"] = train["translation"].map(non_ascii_ratio)

print(train[["src_non_ascii_ratio", "tgt_non_ascii_ratio"]].describe())

plt.figure(figsize=(5, 4))
plt.hist(train["src_non_ascii_ratio"], bins=40)
plt.title("Non-ASCII ratio in transliteration")
plt.xlabel("ratio")
plt.tight_layout()
plt.savefig(FIG_DIR / "src_non_ascii_ratio.png", dpi=150)
plt.show()

#%%
# Token frequency (top-N)

def top_tokens(series: pd.Series, n: int = 20) -> list[tuple[str, int]]:
    counter = Counter()
    for text in series:
        counter.update(text.split())
    return counter.most_common(n)


def plot_top_tokens(items: list[tuple[str, int]], title: str, path: Path) -> None:
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()


plot_top_tokens(top_tokens(train["transliteration"], 20), "Top transliteration tokens", FIG_DIR / "top_translit_tokens.png")
plot_top_tokens(top_tokens(train["translation"], 20), "Top translation tokens", FIG_DIR / "top_translation_tokens.png")

#%%
# Character frequency (top-N)

def top_chars(series: pd.Series, n: int = 30) -> list[tuple[str, int]]:
    counter = Counter()
    for text in series:
        counter.update(text)
    return counter.most_common(n)


plot_top_tokens(top_chars(train["transliteration"], 30), "Top transliteration characters", FIG_DIR / "top_translit_chars.png")
plot_top_tokens(top_chars(train["translation"], 30), "Top translation characters", FIG_DIR / "top_translation_chars.png")

#%%
# Heuristic: logograms are mostly uppercase tokens in transliteration
# This is imperfect for unicode, but gives a quick proxy.

def is_logogram(token: str) -> bool:
    return bool(re.search(r"[A-Z]", token)) and not bool(re.search(r"[a-z]", token))


logogram_flags = []
for text in train["transliteration"]:
    tokens = text.split()
    if not tokens:
        continue
    logogram_flags.append(sum(is_logogram(t) for t in tokens) / len(tokens))

print("logogram_ratio_mean", float(np.mean(logogram_flags)))

plt.figure(figsize=(5, 4))
plt.hist(logogram_flags, bins=40)
plt.title("Estimated logogram token ratio")
plt.xlabel("ratio")
plt.tight_layout()
plt.savefig(FIG_DIR / "logogram_ratio.png", dpi=150)
plt.show()

#%%
# Test set line_start diagnostics (string-based ordering hints)
# Insight: line_start may include apostrophes (1', 1'') indicating broken lines.
line_start = test["line_start"].astype(str)
print("line_start_has_quote", line_start.str.contains("'").sum())
print("line_start_has_double_quote", line_start.str.contains("''").sum())

#%%
# Longest / shortest samples for inspection
print("shortest src")
print(train.sort_values("src_tok_len").head(3)[["oare_id", "transliteration", "translation"]])

print("longest src")
print(train.sort_values("src_tok_len", ascending=False).head(3)[["oare_id", "transliteration", "translation"]])

#%% [markdown]
# Next steps
# - Decide on preprocessing rules for scribal notations (see DATA_INFO.md).
# - Consider sentence-level alignment strategies; test set is sentence-level.
# - After EDA, prototype a tokenizer (char-level vs subword) and build a baseline.
