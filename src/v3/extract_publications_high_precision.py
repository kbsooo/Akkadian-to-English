#%% [markdown]
# # Publications High-Precision Extraction (V3)
#
# Goal: Build a **high-precision** doc-level dataset from publications.csv
# using strict heuristics:
# - has_akkadian == True
# - English translation marker ("Translation") in page_text
# - ID alias match between publications page and published_texts aliases/labels
#
# Output: data/v3/v3_publications_high_precision.csv

#%%
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple


#%%
# ID patterns (matches common catalog IDs)
ID_PATTERNS = [
    r"kt\\s*[\\w/]+\\s*\\d+",
    r"ick\\s*\\d+\\s*\\d+",
    r"cct\\s*\\d+\\s*\\d+\\w*",
    r"bin\\s*\\d+\\s*\\d+",
    r"tc\\s*\\d+\\s*\\d+",
    r"akt\\s*\\d+[a-z]?\\s*\\d+",
    r"poat\\s*\\d+",
    r"vs\\s*\\d+\\s*\\d+",
    r"kts\\s*\\d+\\s*\\d+",
    r"ra\\s*\\d+\\s*\\d+",
    r"or\\s*\\d+\\s*\\d+\\w*",
]

COMPILED_IDS = [re.compile(p, re.IGNORECASE) for p in ID_PATTERNS]
TRANS_MARKER = re.compile(r"\\bTranslation\\b", re.IGNORECASE)


def normalize_id(text: str) -> str:
    return re.sub(r"\\s+", "", text.lower())


def build_alias_map(published_texts_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Build alias -> [oare_id] map and oare_id -> transliteration map.
    Only keeps aliases that match known ID patterns.
    """
    alias_map: Dict[str, List[str]] = {}
    translit_map: Dict[str, str] = {}

    with published_texts_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            oare_id = row.get("oare_id", "")
            translit = row.get("transliteration") or ""
            translit_map[oare_id] = translit

            aliases = row.get("aliases") or ""
            label = row.get("label") or ""
            alias_list = [a.strip() for a in aliases.split("|") if a.strip()]
            if label:
                alias_list.append(label.strip())

            for alias in alias_list:
                if not any(p.search(alias) for p in COMPILED_IDS):
                    continue
                norm = normalize_id(alias)
                if not norm:
                    continue
                alias_map.setdefault(norm, []).append(oare_id)

    return alias_map, translit_map


def extract_translation_block(page_text: str) -> str:
    """
    Extract translation block after the 'Translation' marker.
    Conservative: take up to 5 lines after the marker until blank line.
    """
    lines = [ln.strip() for ln in page_text.splitlines()]
    for i, ln in enumerate(lines):
        if TRANS_MARKER.search(ln):
            # take next lines as translation
            block = []
            for nxt in lines[i + 1 : i + 6]:
                if not nxt:
                    break
                block.append(nxt)
            return " ".join(block).strip()
    return ""


def quality_filter(src: str, tgt: str) -> bool:
    if len(src) < 50 or len(tgt) < 40:
        return False
    if len(tgt) > 2000:
        return False
    # simple ratio check
    ratio = len(src.split()) / max(1, len(tgt.split()))
    if ratio < 0.25 or ratio > 5.0:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract high-precision publication pairs")
    parser.add_argument("--publications", type=str, default="data/publications.csv")
    parser.add_argument("--published-texts", type=str, default="data/published_texts.csv")
    parser.add_argument("--train", type=str, default="data/train.csv")
    parser.add_argument("--output", type=str, default="data/v3/v3_publications_high_precision.csv")
    args = parser.parse_args()

    publications_path = Path(args.publications)
    published_texts_path = Path(args.published_texts)
    train_path = Path(args.train)
    output_path = Path(args.output)

    if not publications_path.exists():
        raise FileNotFoundError(f"Missing: {publications_path}")
    if not published_texts_path.exists():
        raise FileNotFoundError(f"Missing: {published_texts_path}")

    # Train IDs (exclude overlap)
    train_ids = set()
    if train_path.exists():
        with train_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                train_ids.add(row.get("oare_id", ""))

    alias_map, translit_map = build_alias_map(published_texts_path)

    results = []
    total_pages = 0
    matched_pages = 0

    with publications_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_pages += 1
            has_akk = str(row.get("has_akkadian") or "").strip().lower() in ("true", "1", "yes")
            if not has_akk:
                continue
            page_text = row.get("page_text") or ""
            if not TRANS_MARKER.search(page_text):
                continue

            # extract candidate IDs from page text
            found_ids = set()
            for pat in COMPILED_IDS:
                for m in pat.findall(page_text):
                    norm = normalize_id(m)
                    if norm in alias_map:
                        found_ids.add(norm)

            if not found_ids:
                continue

            translation = extract_translation_block(page_text)
            if not translation:
                continue

            matched_pages += 1
            for norm in found_ids:
                for oare_id in alias_map[norm]:
                    if oare_id in train_ids:
                        continue
                    src = translit_map.get(oare_id, "")
                    if not src:
                        continue
                    if not quality_filter(src, translation):
                        continue
                    results.append(
                        {
                            "oare_id": oare_id,
                            "alias": norm,
                            "pdf_name": row.get("\ufeffpdf_name") or row.get("pdf_name") or "",
                            "page": row.get("page", ""),
                            "lang": "en",
                            "src": src,
                            "tgt": translation,
                            "source": "publications_translation_marker",
                        }
                    )

    # Deduplicate
    seen = set()
    deduped = []
    for r in results:
        key = (r["oare_id"], r["tgt"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["oare_id", "alias", "pdf_name", "page", "lang", "src", "tgt", "source"],
        )
        writer.writeheader()
        writer.writerows(deduped)

    print(f"Total pages scanned: {total_pages}")
    print(f"Matched pages: {matched_pages}")
    print(f"Raw pairs: {len(results)}")
    print(f"Deduped pairs: {len(deduped)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
