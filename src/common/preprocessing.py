"""
Akkadian Transliteration Preprocessing Module

통합 전처리 함수 - Train과 Test 모두에 동일하게 적용
"""

import re
import unicodedata
from typing import Optional

import pandas as pd

# ============================================================
# 아래첨자 → 숫자 변환 맵
# ============================================================
SUBSCRIPT_MAP = str.maketrans({
    "\u2080": "0",  # ₀
    "\u2081": "1",  # ₁
    "\u2082": "2",  # ₂
    "\u2083": "3",  # ₃
    "\u2084": "4",  # ₄
    "\u2085": "5",  # ₅
    "\u2086": "6",  # ₆
    "\u2087": "7",  # ₇
    "\u2088": "8",  # ₈
    "\u2089": "9",  # ₉
    "\u2093": "x",  # ₓ
})

# ============================================================
# OCR 아티팩트 정리 맵
# ============================================================
OCR_FIXES = {
    # 독일식 따옴표 → 표준 따옴표 (Test에서 발견된 OOV)
    "„": '"',     # U+201E → U+0022
    """: '"',     # U+201C → U+0022
    """: '"',     # U+201D → U+0022
    "'": "'",     # U+2018 → U+0027
    "'": "'",     # U+2019 → U+0027

    # 특수 하이픈 → 표준 하이픈
    "–": "-",     # U+2013 en-dash
    "—": "-",     # U+2014 em-dash
    "‐": "-",     # U+2010 hyphen
}


def normalize_transliteration(
    text: str,
    handle_gaps: bool = True,
    remove_editorial: bool = True,
) -> str:
    """
    아카드어 전사(transliteration) 정규화

    Train과 Test 모두에 동일하게 적용되어야 함!

    Args:
        text: 원본 전사 텍스트
        handle_gaps: 갭/손상 표시 처리 여부
        remove_editorial: 편집 기호(!, ?, /) 제거 여부

    Returns:
        정규화된 텍스트
    """
    if pd.isna(text) or text is None:
        return ""

    text = str(text)

    # 1. Unicode 정규화 (NFC)
    text = unicodedata.normalize("NFC", text)

    # 2. OCR 아티팩트 정리 (OOV 문자 처리)
    for old, new in OCR_FIXES.items():
        text = text.replace(old, new)

    # 3. 특수 H 문자 정규화 (Ḫ → H, ḫ → h)
    text = text.replace("\u1E2A", "H")  # Ḫ
    text = text.replace("\u1E2B", "h")  # ḫ

    # 4. 아래첨자 → 숫자
    text = text.translate(SUBSCRIPT_MAP)

    # 5. 갭/손상 부분 처리
    if handle_gaps:
        # 말줄임표 → <gap>
        text = text.replace("\u2026", " <gap> ")  # …
        text = re.sub(r"\.\.\.+", " <gap> ", text)

        # 대괄호 내용 → <gap> (손상된 텍스트)
        text = re.sub(r"\[([^\]]*)\]", " <gap> ", text)

    # 6. 미확인 기호 처리
    text = re.sub(r"\bx\b", " <unk> ", text)

    # 7. 편집 기호 제거
    if remove_editorial:
        text = re.sub(r"[!?/]", " ", text)

    # 8. 공백 정규화
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_translation(text: str) -> str:
    """
    영어 번역 정규화

    Args:
        text: 원본 영어 번역

    Returns:
        정규화된 텍스트
    """
    if pd.isna(text) or text is None:
        return ""

    text = str(text)

    # Unicode 정규화
    text = unicodedata.normalize("NFC", text)

    # 공백 정규화
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_for_byt5(
    source: str,
    target: Optional[str] = None,
    max_source_length: int = 256,
    max_target_length: int = 256,
) -> dict:
    """
    ByT5 모델용 데이터 준비

    Args:
        source: 전사 텍스트 (정규화됨)
        target: 번역 텍스트 (정규화됨, 추론 시 None)
        max_source_length: 최대 소스 길이
        max_target_length: 최대 타겟 길이

    Returns:
        {'source': str, 'target': str or None}
    """
    # 이미 정규화된 텍스트 가정
    source = source[:max_source_length * 4]  # 바이트 기준 대략적 제한

    result = {"source": source}

    if target is not None:
        target = target[:max_target_length * 4]
        result["target"] = target

    return result


# ============================================================
# 데이터프레임 전처리 헬퍼
# ============================================================

def preprocess_train_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train 데이터프레임 전처리

    Args:
        df: transliteration, translation 컬럼 포함

    Returns:
        전처리된 데이터프레임
    """
    df = df.copy()

    df["source"] = df["transliteration"].apply(normalize_transliteration)
    df["target"] = df["translation"].apply(normalize_translation)

    # 빈 문자열 제거
    df = df[df["source"].str.len() > 0]
    df = df[df["target"].str.len() > 0]

    return df


def preprocess_test_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test 데이터프레임 전처리

    Args:
        df: transliteration 컬럼 포함

    Returns:
        전처리된 데이터프레임
    """
    df = df.copy()

    df["source"] = df["transliteration"].apply(normalize_transliteration)

    return df


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    # 테스트 케이스
    test_cases = [
        # OOV 문자 „ (Test에서 발견)
        'i-na mup-pì-im aa a-lim(ki) ia-tù u„-mì-im',
        # 말줄임표
        'um-ma kà-ru-um… a-na aa-qí-il…',
        # 아래첨자
        'KIŠIB ma-nu-ba-lúm-a-šur DUMU ṣí-lá-(d)IM₄',
        # 특수 H
        'Ḫa-bu-a-lá DUMU Ì-lí-dan',
        # 대괄호 (손상)
        'a-na [damaged text] qí-bi-ma',
    ]

    print("=== 전처리 테스트 ===\n")
    for text in test_cases:
        normalized = normalize_transliteration(text)
        print(f"원본: {text}")
        print(f"결과: {normalized}")
        print()
