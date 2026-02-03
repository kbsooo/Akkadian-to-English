from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

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


def build_glossary(
    rows: Iterable[dict],
    min_src_count: int = 5,
    min_pair_count: int = 2,
    min_score: float = 0.15,
    max_targets: int = 2,
    min_src_len: int = 2,
    min_tgt_len: int = 2,
) -> dict[str, list[str]]:
    src_count: Counter[str] = Counter()
    cooc: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        src = row.get("src") or ""
        tgt = row.get("tgt") or ""
        src_tokens = set(t for t in tokenize_src(src) if len(t) >= min_src_len)
        tgt_tokens = set(t for t in tokenize_tgt(tgt) if len(t) >= min_tgt_len)
        if not src_tokens or not tgt_tokens:
            continue
        for s in src_tokens:
            src_count[s] += 1
        for s in src_tokens:
            for t in tgt_tokens:
                cooc[s][t] += 1

    glossary: dict[str, list[str]] = {}
    for s, total in src_count.items():
        if total < min_src_count:
            continue
        candidates = []
        for t, c in cooc[s].items():
            if c < min_pair_count:
                continue
            score = c / total
            if score < min_score:
                continue
            candidates.append((score, c, t))
        candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
        if candidates:
            glossary[s] = [t for _, _, t in candidates[:max_targets]]

    return glossary


def build_glossary_prompt(
    src: str,
    glossary: dict[str, list[str]] | None,
    max_items: int = 8,
    drop_prob: float = 0.0,
    rng: random.Random | None = None,
) -> str:
    if not glossary:
        return src
    rng = rng or random
    if drop_prob > 0 and rng.random() < drop_prob:
        return src

    items: list[str] = []
    used = set()
    for tok in tokenize_src(src):
        if tok in used:
            continue
        tgts = glossary.get(tok)
        if not tgts:
            continue
        tgt = tgts[0]
        items.append(f"{tok}={tgt}")
        used.add(tok)
        if len(items) >= max_items:
            break

    if not items:
        return src

    return "GLOSSARY: " + "; ".join(items) + " ||| " + src


def load_glossary(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # ensure list values
    return {k: list(v) for k, v in data.items()}


def save_glossary(path: Path, glossary: dict[str, list[str]]):
    with path.open("w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)
