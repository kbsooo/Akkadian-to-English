#!/usr/bin/env python3
from __future__ import annotations

"""
V5c ByT5-small inference (Kaggle internet-off safe):
- Never downloads anything from HF at runtime.
- Deterministic tokenizer derived from model.config.vocab_size.
- Retrieval + glossary prompting if assets provided.
- Non-empty guarantee: never outputs empty/`...`/punct-only strings.

Kaggle usage:
  python src/v5c/akkadian_v5c_small_infer.py --model-dir /kaggle/input/<your-model>/pytorch/default/1 --assets-dir /kaggle/input/<your-assets>
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, ByT5Tokenizer


# -----------------------------
# V5 normalization (inline)
# -----------------------------

_VOWEL_MAP = {
    "\u00e0": "a",
    "\u00e1": "a",
    "\u00e2": "a",
    "\u0101": "a",
    "\u00e4": "a",
    "\u00c0": "A",
    "\u00c1": "A",
    "\u00c2": "A",
    "\u0100": "A",
    "\u00c4": "A",
    "\u00e8": "e",
    "\u00e9": "e",
    "\u00ea": "e",
    "\u0113": "e",
    "\u00eb": "e",
    "\u00c8": "E",
    "\u00c9": "E",
    "\u00ca": "E",
    "\u0112": "E",
    "\u00cb": "E",
    "\u00ec": "i",
    "\u00ed": "i",
    "\u00ee": "i",
    "\u012b": "i",
    "\u00ef": "i",
    "\u00cc": "I",
    "\u00cd": "I",
    "\u00ce": "I",
    "\u012a": "I",
    "\u00cf": "I",
    "\u00f2": "o",
    "\u00f3": "o",
    "\u00f4": "o",
    "\u014d": "o",
    "\u00f6": "o",
    "\u00d2": "O",
    "\u00d3": "O",
    "\u00d4": "O",
    "\u014c": "O",
    "\u00d6": "O",
    "\u00f9": "u",
    "\u00fa": "u",
    "\u00fb": "u",
    "\u016b": "u",
    "\u00fc": "u",
    "\u00d9": "U",
    "\u00da": "U",
    "\u00db": "U",
    "\u016a": "U",
    "\u00dc": "U",
}

_CONSONANT_MAP = {
    "\u0161": "s",
    "\u0160": "S",
    "\u1e63": "s",
    "\u1e62": "S",
    "\u1e6d": "t",
    "\u1e6c": "T",
    "\u1e2b": "h",
    "\u1e2a": "H",
}

_QUOTE_MAP = {
    "\u201e": '"',
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u02be": "'",
    "\u02bf": "'",
}

_SUBSCRIPT_MAP = str.maketrans(
    {
        "\u2080": "0",
        "\u2081": "1",
        "\u2082": "2",
        "\u2083": "3",
        "\u2084": "4",
        "\u2085": "5",
        "\u2086": "6",
        "\u2087": "7",
        "\u2088": "8",
        "\u2089": "9",
        "\u2093": "x",
    }
)

_FULL_MAP = str.maketrans({**_VOWEL_MAP, **_CONSONANT_MAP, **_QUOTE_MAP})


def normalize_transliteration(text) -> str:
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)

    # Protect literal tokens
    text = text.replace("<gap>", "__LIT_GAP__").replace("<big_gap>", "__LIT_BIG_GAP__")

    # Remove apostrophe line numbers only (1', 1'')
    text = re.sub(r"\b\d+'{1,2}\b", " ", text)

    # <content> blocks
    text = re.sub(r"<<([^>]+)>>", r"\1", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)

    # large gaps
    text = re.sub(r"\[\s*\u2026+\s*\u2026*\s*\]", " __BIG_GAP__ ", text)
    text = re.sub(r"\[\s*\.\.\.+\s*\.\.\.+\s*\]", " __BIG_GAP__ ", text)
    text = text.replace("\u2026", " __BIG_GAP__ ")
    text = re.sub(r"\.\.\.+", " __BIG_GAP__ ", text)

    # [x]
    text = re.sub(r"\[\s*x\s*\]", " __GAP__ ", text, flags=re.IGNORECASE)

    # [content] -> content
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)

    # Half brackets
    text = text.replace("\u2039", "").replace("\u203A", "")
    text = text.replace("\u2308", "").replace("\u2309", "")
    text = text.replace("\u230A", "").replace("\u230B", "")
    text = text.replace("\u02F9", "").replace("\u02FA", "")

    # Character maps
    text = text.translate(_FULL_MAP).translate(_SUBSCRIPT_MAP)

    # Scribal notations / divider
    text = re.sub(r"[!?/]", " ", text)
    text = re.sub(r"\s*:\s*", " ", text)

    # Standalone x
    text = re.sub(r"\bx\b", " __GAP__ ", text, flags=re.IGNORECASE)

    # Convert placeholders and restore literals
    text = text.replace("__GAP__", "<gap>").replace("__BIG_GAP__", "<big_gap>")
    text = text.replace("__LIT_GAP__", "<gap>").replace("__LIT_BIG_GAP__", "<big_gap>")

    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Prompting + Retrieval
# -----------------------------

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
    if s is None:
        return True
    t = str(s).strip()
    if not t:
        return True
    if t in {"...", "…"}:
        return True
    if re.fullmatch(r"[.\s…\-–—,;:!?\"'()\[\]{}<>/\\]+", t):
        return True
    return False


def char_ngrams(text: str, n: int = 3) -> list[str]:
    text = f" {text} "
    if len(text) < n:
        return [text]
    return [text[i : i + n] for i in range(len(text) - n + 1)]


class JaccardRetriever:
    def __init__(self, texts: list[str], *, n: int = 3, max_candidates: int = 500):
        self.texts = texts
        self.n = n
        self.max_candidates = max_candidates
        self.grams = [set(char_ngrams(t, n)) for t in texts]
        self.inv: dict[str, list[int]] = {}
        for i, gs in enumerate(self.grams):
            for g in gs:
                self.inv.setdefault(g, []).append(i)

    def retrieve(self, query: str, k: int) -> list[int]:
        qg = set(char_ngrams(query, self.n))
        freq: Counter[int] = Counter()
        for g in qg:
            for idx in self.inv.get(g, []):
                freq[idx] += 1
        if not freq:
            # No overlap: return first k (stable)
            return list(range(min(k, len(self.texts))))
        candidates = [idx for idx, _ in freq.most_common(self.max_candidates)]
        scored = []
        for idx in candidates:
            inter = len(qg & self.grams[idx])
            union = len(qg) + len(self.grams[idx]) - inter
            scored.append((inter / union if union else 0.0, idx))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [idx for _, idx in scored[:k]]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tm_pairs(path: Path, *, max_rows: int | None = None) -> list[dict]:
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs


def build_prompt_with_retrieval(
    src: str,
    *,
    tm_pairs: list[dict],
    retriever: JaccardRetriever | None,
    glossary: dict[str, list[str]] | None,
    tm_k: int,
    max_items: int,
    max_prompt_chars: int,
) -> str:
    if not tm_pairs or retriever is None:
        return src

    idxs = retriever.retrieve(src, k=tm_k)
    neighbors = [tm_pairs[i] for i in idxs if 0 <= i < len(tm_pairs)]

    q_tokens = tokenize_src(src)
    local_counts: dict[str, Counter[str]] = {t: Counter() for t in q_tokens}
    for nb in neighbors:
        nb_src_tokens = set(tokenize_src(nb.get("src", "")))
        nb_tgt_tokens = tokenize_tgt(nb.get("tgt", ""))
        if not nb_src_tokens or not nb_tgt_tokens:
            continue
        for tok in q_tokens:
            if tok in nb_src_tokens:
                local_counts[tok].update(nb_tgt_tokens)

    items: list[str] = []
    used: set[str] = set()
    for tok in q_tokens:
        if tok in used:
            continue
        tgt = None
        if local_counts.get(tok) and local_counts[tok]:
            tgt = local_counts[tok].most_common(1)[0][0]
        elif glossary and tok in glossary and glossary[tok]:
            tgt = glossary[tok][0]
        if tgt:
            items.append(f"{tok}={tgt}")
            used.add(tok)
        if len(items) >= max_items:
            break

    if not items:
        return src

    prompt = "GLOSSARY: " + "; ".join(items) + " ||| " + src
    if len(prompt) > max_prompt_chars:
        return src
    return prompt


def tm_fallback_translation(
    src: str,
    *,
    tm_pairs: list[dict],
    retriever: JaccardRetriever | None,
) -> str | None:
    if not tm_pairs or retriever is None:
        return None
    idxs = retriever.retrieve(src, k=1)
    if not idxs:
        return None
    tgt = tm_pairs[idxs[0]].get("tgt", "")
    if tgt and not is_bad_output(tgt):
        return str(tgt).strip()
    return None


@dataclass
class InferArgs:
    model_dir: Path | None
    assets_dir: Path | None
    num_beams: int
    max_source_len: int
    max_new_tokens: int
    min_new_tokens: int
    batch_size: int
    tm_k: int
    glossary_max_items: int
    max_prompt_chars: int
    max_candidates: int
    seed: int


def parse_args() -> InferArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--assets-dir", type=Path, default=None)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--max-source-len", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--min-new-tokens", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--tm-k", type=int, default=5)
    p.add_argument("--glossary-max-items", type=int, default=8)
    p.add_argument("--max-prompt-chars", type=int, default=512)
    p.add_argument("--max-candidates", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    # Notebook-friendly: ignore ipykernel argv like `-f ...`
    a, _unknown = p.parse_known_args()
    return InferArgs(
        model_dir=a.model_dir,
        assets_dir=a.assets_dir,
        num_beams=a.num_beams,
        max_source_len=a.max_source_len,
        max_new_tokens=a.max_new_tokens,
        min_new_tokens=a.min_new_tokens,
        batch_size=a.batch_size,
        tm_k=a.tm_k,
        glossary_max_items=a.glossary_max_items,
        max_prompt_chars=a.max_prompt_chars,
        max_candidates=a.max_candidates,
        seed=a.seed,
    )


def is_kaggle() -> bool:
    return Path("/kaggle/input").exists()


def find_comp_data_dir() -> Path:
    if not is_kaggle():
        if Path("data/test.csv").exists():
            return Path("data")
        raise FileNotFoundError("test.csv not found")
    base = Path("/kaggle/input")
    for d in base.iterdir():
        if (d / "test.csv").exists():
            return d
    raise FileNotFoundError("Competition data dir not found under /kaggle/input")


def find_assets_dir() -> Path | None:
    if not is_kaggle():
        for d in [Path("assets"), Path("data"), Path("models")]:
            if (d / "v5c_tm_pairs.jsonl").exists() or (d / "v5c_glossary.json").exists():
                return d
        return None
    base = Path("/kaggle/input")
    for d in base.iterdir():
        if (d / "v5c_tm_pairs.jsonl").exists() or (d / "v5c_glossary.json").exists():
            return d
    return None


def find_model_dir() -> Path:
    # Allow overriding from env (useful in notebooks).
    env = os.environ.get("V5C_MODEL_DIR")
    if env:
        p = Path(env)
        if (p / "config.json").exists():
            return p

    if not is_kaggle():
        raise FileNotFoundError("Please provide --model-dir when running locally.")
    base = Path("/kaggle/input")
    preferred = base / "akkadian-v5c-small/pytorch/default/1"
    if (preferred / "config.json").exists():
        return preferred
    # common layout for Kaggle Model
    for d in base.iterdir():
        cand = d / "pytorch/default/1"
        if (cand / "config.json").exists():
            return cand
    raise FileNotFoundError("Model dir not found. Pass --model-dir explicitly.")


@torch.no_grad()
def generate_once(
    model,
    tokenizer,
    device: torch.device,
    texts: list[str],
    *,
    num_beams: int,
    max_source_len: int,
    max_new_tokens: int,
    min_new_tokens: int,
) -> list[str]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_len,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=False,
        early_stopping=True,
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def chunked(iterable: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def main() -> None:
    args = parse_args()

    # Make sampling retry reproducible (best-effort).
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    comp_dir = find_comp_data_dir()
    model_dir = args.model_dir or find_model_dir()
    assets_dir = args.assets_dir or find_assets_dir()

    print("=" * 60)
    print("🚀 V5c ByT5-small INFER")
    print("=" * 60)
    print(f"📁 Competition: {comp_dir}")
    print(f"🤖 Model dir:   {model_dir}")
    print(f"🧱 Assets dir:  {assets_dir if assets_dir else 'not found'}")
    print(f"🎮 CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # Load model (local only)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Deterministic tokenizer derived from model vocab size (no network, no files)
    extra_ids = int(model.config.vocab_size) - 259
    if extra_ids < 0:
        raise ValueError(f"Unexpected vocab_size={model.config.vocab_size} for ByT5")
    tokenizer = ByT5Tokenizer(extra_ids=extra_ids)
    assert len(tokenizer) == model.config.vocab_size, "Tokenizer/model vocab mismatch!"
    print(f"🔤 Tokenizer vocab: {len(tokenizer)} (extra_ids={extra_ids})")

    # Load assets (optional)
    tm_pairs: list[dict] = []
    glossary: dict[str, list[str]] | None = None
    retriever: JaccardRetriever | None = None

    if assets_dir:
        tm_path = assets_dir / "v5c_tm_pairs.jsonl"
        gl_path = assets_dir / "v5c_glossary.json"
        if tm_path.exists():
            tm_pairs = load_tm_pairs(tm_path)
            print(f"🧠 TM pairs: {len(tm_pairs):,}")
        if gl_path.exists():
            glossary = load_json(gl_path)
            glossary = {k: list(v) for k, v in glossary.items()}
            print(f"🧠 Glossary size: {len(glossary):,}")
        if tm_pairs:
            retriever = JaccardRetriever([p.get("src", "") for p in tm_pairs], max_candidates=args.max_candidates)

    # Load test
    test_df = pd.read_csv(comp_dir / "test.csv")
    print(f"📄 Test rows: {len(test_df):,}")

    # Normalize
    normalized = [normalize_transliteration(t) for t in tqdm(test_df["transliteration"], desc="Normalizing")]

    # Build prompts
    if tm_pairs and retriever:
        prompts = [
            build_prompt_with_retrieval(
                s,
                tm_pairs=tm_pairs,
                retriever=retriever,
                glossary=glossary,
                tm_k=args.tm_k,
                max_items=args.glossary_max_items,
                max_prompt_chars=args.max_prompt_chars,
            )
            for s in tqdm(normalized, desc="Prompting")
        ]
    else:
        prompts = normalized

    print("📝 Prompt sample:")
    for i in range(min(2, len(prompts))):
        print(f"   [{i}] {prompts[i][:160]}...")

    # Generate with retries (model-only; no output substitution).
    translations: list[str] = [""] * len(prompts)
    remaining = list(range(len(prompts)))

    def run_attempt(
        attempt_name: str,
        *,
        num_beams: int,
        do_sample: bool,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[int]:
        nonlocal translations
        if not remaining:
            return []
        print(f"🚀 Generation attempt: {attempt_name} (remaining={len(remaining)})")
        new_remaining: list[int] = []
        for chunk in tqdm(list(chunked(remaining, args.batch_size)), desc=f"Gen:{attempt_name}", unit="batch"):
            texts = [prompts[i] for i in chunk]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_source_len,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            gen_kwargs: dict[str, Any] = dict(
                num_beams=num_beams,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                do_sample=do_sample,
                early_stopping=True,
            )
            if do_sample:
                gen_kwargs["temperature"] = float(temperature or 0.8)
                gen_kwargs["top_p"] = float(top_p or 0.95)

            out = model.generate(**inputs, **gen_kwargs)
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            for idx, s in zip(chunk, decoded):
                translations[idx] = s
                if is_bad_output(s):
                    new_remaining.append(idx)

        return new_remaining

    # Attempt 1: beam search
    remaining = run_attempt("beam", num_beams=args.num_beams, do_sample=False)
    # Attempt 2: greedy
    if remaining:
        remaining = run_attempt("greedy", num_beams=1, do_sample=False)
    # Attempt 3: sampling (last resort, still model-only)
    if remaining:
        remaining = run_attempt("sample", num_beams=1, do_sample=True, temperature=0.9, top_p=0.95)

    if remaining:
        # Hard-fail instead of substituting outputs: prevents silent garbage submissions.
        print("❌ Failed to produce valid outputs for some rows after retries.")
        for i in remaining[:10]:
            print(f"   [bad idx={i}] src='{normalized[i][:120]}...' out='{translations[i]}'")
        raise RuntimeError(f"Bad outputs remain after retries: {len(remaining)}")

    print("✅ Generation completed without empty/ellipsis-only outputs (after retries).")

    print("📝 Output sample:")
    for i in range(min(3, len(translations))):
        print(f"   [{i}] {translations[i][:160]}...")

    sub = pd.DataFrame({"id": test_df["id"], "translation": translations})
    out_path = Path("/kaggle/working/submission.csv") if is_kaggle() else Path("submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path} ({len(sub):,} rows)")


if __name__ == "__main__":
    main()
