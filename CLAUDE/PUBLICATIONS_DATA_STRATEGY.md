# Publications 데이터 활용 전략 (수정본)

**Date**: 2026-02-01
**Version**: 2.0 (대회 가이드라인 기반 전면 재작성)

---

## 1. 데이터 구조 이해

### 1.1 핵심 데이터 파일 관계

```
┌─────────────────────────────────────────────────────────────────────┐
│                         데이터 흐름도                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   published_texts.csv (7,953개)                                     │
│   ┌─────────────────────────────┐                                   │
│   │ • oare_id (고유 식별자)      │                                   │
│   │ • transliteration ✅        │                                   │
│   │ • aliases, label (검색 키)  │                                   │
│   │ • ❌ 번역 없음              │                                   │
│   └──────────────┬──────────────┘                                   │
│                  │                                                   │
│         ┌───────┴───────┐                                           │
│         ▼               ▼                                           │
│   ┌─────────────┐  ┌─────────────────────┐                         │
│   │Sentences_   │  │  publications.csv   │                         │
│   │Oare (1,700) │  │  (216,602 pages)    │                         │
│   ├─────────────┤  ├─────────────────────┤                         │
│   │• text_uuid  │  │• 학술 논문 OCR      │                         │
│   │• translation│  │• 번역 포함 ✅       │                         │
│   │  ✅ (영어)  │  │• 다국어 (영/독/불)  │                         │
│   └──────┬──────┘  └──────────┬──────────┘                         │
│          │                    │                                      │
│          │    JOIN by ID      │                                      │
│          └────────┬───────────┘                                      │
│                   ▼                                                  │
│         ┌─────────────────────┐                                     │
│         │   학습 데이터 생성   │                                     │
│         │  (transliteration,  │                                     │
│         │   translation) 쌍   │                                     │
│         └─────────────────────┘                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 데이터 통계

| 데이터셋 | 행 수 | 역할 |
|---------|-------|------|
| **train.csv** | 1,561 | 기존 학습 데이터 (transliteration + translation) |
| **published_texts.csv** | 7,953 | Transliteration 저장소 (번역 없음) |
| **Sentences_Oare** | 9,782 (1,700 문서) | 문장 단위 번역 (영어) |
| **publications.csv** | 216,602 pages | 학술 논문 OCR (번역 포함, 다국어) |

### 1.3 핵심 수치 분석

```
published_texts (7,953)
    │
    ├── train.csv에 이미 있음: 1,561 (19.6%)
    │
    └── train.csv에 없음: 6,392 (80.4%)  ← 신규 데이터 후보!
            │
            ├── Sentences_Oare에 번역 있음: 1,164 ✅ (즉시 활용 가능)
            │
            └── Sentences_Oare에 없음: 5,228 (publications에서 추출 필요)
```

---

## 2. 3단계 데이터 확장 전략

### 전략 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 1: Sentences_Oare 활용 (즉시, 고품질)                        │
│  ────────────────────────────────────────────────────────────────   │
│  • published_texts + Sentences_Oare JOIN                            │
│  • 예상 수확: +1,164개                                              │
│  • 난이도: ⭐ (쉬움)                                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 2: Publications ID 매칭 (중기, 중품질)                       │
│  ────────────────────────────────────────────────────────────────   │
│  • published_texts.aliases → publications.page_text 검색            │
│  • 번역 추출 + 다국어→영어 변환                                     │
│  • 예상 수확: +2,000~3,000개                                        │
│  • 난이도: ⭐⭐⭐ (중간)                                              │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 3: 문장 단위 정렬 (장기, 최적화)                             │
│  ────────────────────────────────────────────────────────────────   │
│  • Sentences_Oare의 line_number 활용                                │
│  • 문장 단위 정밀 정렬                                              │
│  • 품질 향상 (문서→문장 단위)                                       │
│  • 난이도: ⭐⭐ (중간)                                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 1: Sentences_Oare 활용

### 3.1 개요

**가장 쉽고 확실한 방법**: `published_texts.csv`와 `Sentences_Oare`를 `oare_id` = `text_uuid`로 JOIN

### 3.2 구현

```python
import pandas as pd

def extract_sentences_oare_pairs():
    """
    Sentences_Oare에서 번역이 있는 새 학습 데이터 추출
    """
    # 데이터 로드
    train = pd.read_csv('data/train.csv')
    published_texts = pd.read_csv('data/published_texts.csv')
    sentences = pd.read_csv('data/Sentences_Oare_FirstWord_LinNum.csv')

    # train에 없는 ID 찾기
    train_ids = set(train['oare_id'])

    # published_texts에서 transliteration 가져오기
    new_data = []

    for oare_id in published_texts['oare_id'].unique():
        if oare_id in train_ids:
            continue  # 이미 train에 있음

        # Sentences_Oare에서 번역 찾기
        sent_rows = sentences[sentences['text_uuid'] == oare_id]
        if len(sent_rows) == 0:
            continue  # 번역 없음

        # published_texts에서 transliteration
        pt_row = published_texts[published_texts['oare_id'] == oare_id].iloc[0]
        transliteration = pt_row['transliteration']

        # 문장들의 번역을 결합
        translations = sent_rows['translation'].dropna().tolist()
        if not translations:
            continue

        combined_translation = ' '.join(translations)

        new_data.append({
            'oare_id': oare_id,
            'src': transliteration,
            'tgt': combined_translation,
            'source': 'sentences_oare'
        })

    return pd.DataFrame(new_data)

# 실행
new_pairs = extract_sentences_oare_pairs()
print(f"추출된 새 학습 쌍: {len(new_pairs)}")
# 예상: ~1,164개
```

### 3.3 예상 결과

| 항목 | 수치 |
|------|------|
| 추출 가능 데이터 | ~1,164개 |
| 기존 데이터 | 2,565개 |
| **Phase 1 후 총합** | **~3,729개 (+45%)** |

---

## 4. Phase 2: Publications ID 매칭

### 4.1 개요

`Sentences_Oare`에 없는 5,228개 텍스트의 번역을 `publications.csv`에서 찾아 추출

### 4.2 워크플로우

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: ID 매칭                                                    │
│  ─────────────────                                                  │
│  published_texts.aliases (예: "kt v/k 204", "BIN 4 112")           │
│           ↓                                                         │
│  publications.page_text에서 해당 ID 검색                            │
│           ↓                                                         │
│  해당 페이지 찾기                                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Step 2: 번역 추출                                                  │
│  ─────────────────                                                  │
│  페이지 텍스트에서 번역 부분 파싱                                   │
│  (패턴: 줄번호 범위 "1-3)" 뒤의 텍스트)                            │
├─────────────────────────────────────────────────────────────────────┤
│  Step 3: 다국어 → 영어 변환                                         │
│  ─────────────────────────                                          │
│  독일어/프랑스어 번역 감지                                          │
│           ↓                                                         │
│  번역 API (Google Translate / LLM) 사용                             │
├─────────────────────────────────────────────────────────────────────┤
│  Step 4: 품질 필터링                                                │
│  ─────────────────                                                  │
│  • 길이 검증                                                        │
│  • 언어 일관성 검증                                                 │
│  • 중복 제거                                                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 ID 매칭 구현

```python
import pandas as pd
import re

def build_id_index(publications):
    """
    publications의 모든 페이지에서 텍스트 ID 인덱스 구축
    """
    # 자주 사용되는 ID 패턴
    id_patterns = [
        r'kt\s*[\w/]+\s*\d+',       # kt v/k 204, kt n/k 1340
        r'ICK\s*\d+\s*\d+',         # ICK 1 146
        r'CCT\s*\d+\s*\d+\w*',      # CCT 6 17a
        r'BIN\s*\d+\s*\d+',         # BIN 4 112
        r'TC\s*\d+\s*\d+',          # TC 3 100
        r'AKT\s*\d+[a-z]?\s*\d+',   # AKT 3 112, AKT 7a 15
        r'POAT\s*\d+',              # POAT 28
        r'VS\s*\d+\s*\d+',          # VS 26 71
        r'KTS\s*\d+\s*\d+',         # KTS 1 15
    ]

    id_to_pages = {}  # ID → [(pdf_name, page, page_text), ...]

    akk_pubs = publications[publications['has_akkadian'] == True]

    for idx, row in akk_pubs.iterrows():
        text = str(row['page_text'])

        for pattern in id_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # ID 정규화 (공백 제거, 소문자)
                norm_id = re.sub(r'\s+', '', match.lower())

                if norm_id not in id_to_pages:
                    id_to_pages[norm_id] = []

                id_to_pages[norm_id].append({
                    'pdf_name': row['pdf_name'],
                    'page': row['page'],
                    'page_text': text
                })

    return id_to_pages

def match_published_texts(published_texts, id_index):
    """
    published_texts의 각 항목을 publications와 매칭
    """
    matches = []

    for idx, row in published_texts.iterrows():
        aliases = str(row['aliases']) if pd.notna(row['aliases']) else ''

        for alias in aliases.split('|'):
            alias = alias.strip()
            norm_alias = re.sub(r'\s+', '', alias.lower())

            if norm_alias in id_index:
                matches.append({
                    'oare_id': row['oare_id'],
                    'alias': alias,
                    'transliteration': row['transliteration'],
                    'pages': id_index[norm_alias]
                })
                break  # 첫 매칭만 사용

    return matches
```

### 4.4 번역 추출 구현

```python
def extract_translation_from_page(page_text, text_id):
    """
    학술 논문 페이지에서 특정 텍스트의 번역 추출

    일반적인 패턴:
    - "1-3) Translation text here..."
    - "Translation:" 마커 뒤의 텍스트
    - 따옴표로 둘러싸인 번역
    """
    translations = []

    # 패턴 1: 줄번호 범위 뒤의 번역
    # 예: "1-3) An Lip(i)t-Istar, folgendermaßen Samas-damiq:"
    line_range_pattern = r'(\d+[-–]\d+\))\s*([^0-9\n][^\n]{20,})'
    matches = re.findall(line_range_pattern, page_text)
    for line_range, translation in matches:
        translations.append({
            'type': 'line_range',
            'line_range': line_range,
            'text': translation.strip()
        })

    # 패턴 2: "Translation:" 마커
    trans_marker_pattern = r'(?:Translation|Übersetzung|Traduction)[:\s]+([^\n]{20,})'
    matches = re.findall(trans_marker_pattern, page_text, re.IGNORECASE)
    for match in matches:
        translations.append({
            'type': 'marker',
            'text': match.strip()
        })

    # 패턴 3: 따옴표 안의 번역
    # 영어 번역은 보통 "..." 안에
    quote_pattern = r'"([A-Z][^"]{20,})"'
    matches = re.findall(quote_pattern, page_text)
    for match in matches:
        translations.append({
            'type': 'quoted',
            'text': match.strip()
        })

    return translations
```

### 4.5 다국어 → 영어 변환

```python
from deep_translator import GoogleTranslator
import langdetect

def translate_to_english(text):
    """
    비영어 텍스트를 영어로 번역
    """
    try:
        lang = langdetect.detect(text)
    except:
        return text, 'unknown'

    if lang == 'en':
        return text, 'en'

    # 지원 언어: 독일어(de), 프랑스어(fr), 터키어(tr) 등
    if lang in ['de', 'fr', 'tr', 'it']:
        try:
            translator = GoogleTranslator(source=lang, target='en')
            translated = translator.translate(text)
            return translated, lang
        except Exception as e:
            print(f"번역 실패: {e}")
            return text, lang

    return text, lang

def process_multilingual_translations(matches):
    """
    추출된 번역을 영어로 통일
    """
    processed = []

    for match in matches:
        for trans in match.get('translations', []):
            text = trans['text']
            translated, orig_lang = translate_to_english(text)

            processed.append({
                'oare_id': match['oare_id'],
                'transliteration': match['transliteration'],
                'translation_orig': text,
                'translation_en': translated,
                'orig_lang': orig_lang,
                'source_type': trans['type']
            })

    return processed
```

### 4.6 예상 결과

| 항목 | 수치 |
|------|------|
| ID 매칭 성공 (추정) | ~3,000~4,000개 |
| 번역 추출 성공 (추정) | ~2,000~3,000개 |
| 영어 번역 | ~60% (직접) |
| 다국어→영어 번역 | ~40% |
| **Phase 2 후 총합** | **~5,700~6,700개** |

---

## 5. Phase 3: 문장 단위 정렬

### 5.1 개요

대회 가이드에서 권장하는 최적의 학습 형식: **문장 단위 정렬**

```
문서 단위:
  src: "1 GÍN a-šùr-ma-lik 1 10-e a-na-ku 0.83333 GÍN a-nu-nu..."
  tgt: "Dagānia owes Suen-nādā 37 1/2 shekels refined silver. He will add..."

문장 단위 (더 좋음):
  src: "1 GÍN a-šùr-ma-lik 1 10-e a-na-ku"
  tgt: "Dagānia owes Suen-nādā 37 1/2 shekels refined silver."
```

### 5.2 Sentences_Oare 활용

```python
def create_sentence_aligned_data():
    """
    Sentences_Oare의 line_number를 활용하여 문장 단위 정렬
    """
    sentences = pd.read_csv('data/Sentences_Oare_FirstWord_LinNum.csv')
    published_texts = pd.read_csv('data/published_texts.csv')

    aligned_pairs = []

    for text_uuid in sentences['text_uuid'].unique():
        # 해당 문서의 모든 문장
        doc_sentences = sentences[sentences['text_uuid'] == text_uuid].sort_values('line_number')

        # published_texts에서 transliteration
        pt_row = published_texts[published_texts['oare_id'] == text_uuid]
        if len(pt_row) == 0:
            continue

        full_translit = pt_row.iloc[0]['transliteration']

        # 각 문장에 대해
        for idx, sent_row in doc_sentences.iterrows():
            line_num = sent_row['line_number']
            translation = sent_row['translation']
            first_word = sent_row['first_word_spelling']

            if pd.isna(translation) or pd.isna(line_num):
                continue

            # transliteration에서 해당 줄 추출 시도
            # (line_number와 first_word를 이용한 매칭)
            sentence_translit = extract_sentence_translit(
                full_translit,
                line_num,
                first_word
            )

            if sentence_translit:
                aligned_pairs.append({
                    'text_uuid': text_uuid,
                    'line_number': line_num,
                    'src': sentence_translit,
                    'tgt': translation
                })

    return pd.DataFrame(aligned_pairs)

def extract_sentence_translit(full_translit, line_num, first_word):
    """
    전체 transliteration에서 특정 줄의 텍스트 추출

    패턴: 줄번호가 transliteration 앞에 표시됨
    예: "1 a-na be-lí-ia 2 qí-bí-ma 3 um-ma..."
    """
    # 줄번호 기반 분리 시도
    line_pattern = rf'(?:^|\s){int(line_num)}\s+([^\d]+?)(?=\s+\d+\s|\s*$)'
    match = re.search(line_pattern, full_translit)

    if match:
        return match.group(1).strip()

    # first_word 기반 검색
    if first_word:
        word_pattern = rf'{re.escape(first_word)}[^\d]*?(?=\s+\d+\s|\s*$)'
        match = re.search(word_pattern, full_translit, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    return None
```

### 5.3 예상 효과

| 정렬 방식 | 데이터 수 | 품질 |
|----------|----------|------|
| 문서 단위 | ~6,000개 | 중간 |
| **문장 단위** | ~15,000~20,000개 | **높음** |

---

## 6. 통합 파이프라인

### 6.1 실행 순서

```
┌─────────────────────────────────────────────────────────────────────┐
│  Day 1-2: Phase 1 (Sentences_Oare)                                  │
│  ─────────────────────────────────────                              │
│  • published_texts + Sentences_Oare JOIN                            │
│  • 품질 검증                                                        │
│  • 결과: +1,164개                                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Day 3-5: Phase 2 (Publications ID 매칭)                            │
│  ─────────────────────────────────────                              │
│  • ID 인덱스 구축                                                   │
│  • 번역 추출                                                        │
│  • 다국어 → 영어 변환                                               │
│  • 결과: +2,000~3,000개                                             │
├─────────────────────────────────────────────────────────────────────┤
│  Day 6-7: Phase 3 (문장 단위 정렬)                                  │
│  ─────────────────────────────────                                  │
│  • line_number 기반 정렬                                            │
│  • 품질 필터링                                                      │
│  • 결과: 문장 단위 데이터셋                                         │
├─────────────────────────────────────────────────────────────────────┤
│  Day 8+: 학습                                                       │
│  ────────                                                           │
│  • 통합 데이터셋으로 ByT5-Large + LoRA 학습                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 예상 데이터 규모

| 단계 | 추가 데이터 | 누적 |
|------|------------|------|
| 현재 (V2) | - | 2,565 |
| Phase 1 | +1,164 | 3,729 |
| Phase 2 | +2,500 | 6,229 |
| Phase 3 (문장화) | ×3 | ~18,000 |
| **최종** | | **~18,000개 (7배 증가)** |

---

## 7. 품질 관리

### 7.1 필터링 기준

```python
def quality_filter(pair):
    """학습 데이터 품질 필터링"""
    src, tgt = pair['src'], pair['tgt']

    # 1. 길이 필터
    if len(src) < 10 or len(tgt) < 10:
        return False
    if len(src) > 2000 or len(tgt) > 2000:
        return False

    # 2. 빈 내용 필터
    if src.strip() == '' or tgt.strip() == '':
        return False

    # 3. 언어 검증 (tgt가 영어인지)
    try:
        if langdetect.detect(tgt) != 'en':
            return False
    except:
        pass

    # 4. 특수문자 과다 필터
    special_ratio = len(re.findall(r'[^\w\s\-\.\,\;\:\'\"]', tgt)) / len(tgt)
    if special_ratio > 0.3:
        return False

    # 5. 중복 문장 필터
    if re.search(r'(.{20,})\1', tgt):  # 20자 이상 반복
        return False

    return True
```

### 7.2 중복 제거

```python
def deduplicate(df):
    """중복 제거"""
    # 1. 완전 중복
    df = df.drop_duplicates(subset=['src', 'tgt'])

    # 2. src 기준 중복 (같은 src에 여러 tgt)
    # → 가장 긴 tgt 선택
    df = df.sort_values('tgt', key=lambda x: x.str.len(), ascending=False)
    df = df.drop_duplicates(subset=['src'], keep='first')

    return df
```

---

## 8. 파일 구조

### 8.1 생성할 파일

```
data/v3/
├── v3_phase1_sentences_oare.csv    # Phase 1 결과
├── v3_phase2_publications.csv       # Phase 2 결과
├── v3_sentence_aligned.csv          # Phase 3 결과 (문장 단위)
├── v3_train_combined.csv            # 통합 학습 데이터
├── v3_val.csv                       # 검증 데이터
└── extraction_log.json              # 추출 로그 및 통계
```

### 8.2 구현 파일

```
src/data/
├── extract_sentences_oare.py        # Phase 1
├── extract_publications.py          # Phase 2
├── align_sentences.py               # Phase 3
├── translate_to_english.py          # 다국어 변환
├── quality_filter.py                # 품질 필터링
└── combine_datasets.py              # 통합
```

---

## 9. 예상 점수 영향

| 전략 | 데이터 | 예상 점수 |
|------|--------|----------|
| 현재 (V2) | 2,565 | 11.8 |
| + Phase 1 | 3,729 (+45%) | 14-16 |
| + Phase 2 | 6,229 (+143%) | 18-22 |
| + Phase 3 (문장화) | 18,000 (+600%) | 25-30 |
| + Large/LoRA | - | 28-35 |

**목표 35점 달성 가능성: 높음**

---

## 10. 다음 단계

1. **`src/data/extract_sentences_oare.py`** 구현 (Phase 1)
2. **테스트 및 품질 검증**
3. **`src/data/extract_publications.py`** 구현 (Phase 2)
4. **다국어 번역 파이프라인 구축**
5. **통합 데이터셋 생성**

---

**Document Version**: 2.0
**Status**: 대회 가이드라인 기반 전면 재작성
**Previous Version**: 1.0 (Pseudo-labeling 기반 - 폐기)
