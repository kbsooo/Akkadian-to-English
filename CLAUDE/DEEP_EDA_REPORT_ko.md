# Deep Research EDA 보고서: 아카디안 번역 대회

**작성일**: 2026-02-02  
**목표**: 대회 데이터 전반을 종합적으로 이해하여 최적 번역 전략 수립

---

## Executive Summary

이번 분석에서 **train/test 간 구조적 차이**가 매우 크며, 모델 성능에 직접적인 영향을 줄 것으로 확인됨:

1. **Test는 LINE-SEGMENTED** (한 문서에서 4개 구간), **Train은 FULL DOCUMENTS** (1,561개 전체 문서)
2. **Test는 HOLD-OUT 문서** (`text_id: 332fda50`)로 제공되며, 어떤 제공 데이터셋에도 존재하지 않음
3. **인코딩 차이** 존재 (예: `„` vs `₄`)
4. **길이 분포 차이**: Test 평균 ~21 토큰 vs Train 평균 ~57 토큰

---

## 1. Test 데이터 분석 (가장 중요)

### 1.1 구조 개요

| 필드 | 설명 |
|------|------|
| `id` | 순차 정수 (0-3) |
| `text_id` | 문서 식별자: `332fda50` (UUID 일부로 추정) |
| `line_start` | 원문 태블릿 내 시작 라인 |
| `line_end` | 원문 태블릿 내 종료 라인 |
| `transliteration` | 아카디안 전사 텍스트 |

### 1.2 Test 내용 요약

| id | line_start | line_end | span | tokens | chars |
|----|------------|----------|------|--------|-------|
| 0 | 1 | 7 | 6 lines | 16 | 133 |
| 1 | 7 | 14 | 7 lines | 19 | 146 |
| 2 | 14 | 24 | 10 lines | 34 | 267 |
| 3 | 25 | 30 | 5 lines | 16 | 129 |

**핵심 관찰**:
- 4개 행 모두 **같은 문서(text_id: 332fda50)**에 속함
- **라인 단위로 구간 분할**된 형태
- 라인 구간이 **경계에서 겹침** (7, 14, 25)
- 원문 전체는 1-30라인에 해당

### 1.3 Test 전사 패턴

**Row 0 샘플**:
```
um-ma kà-ru-um kà-ni-ia-ma a-na aa-qí-il... da-tim aí-ip-ri-ni kà-ar kà-ar-ma ú wa-bar-ra-tim qí-bi„-ma mup-pu-um aa a-lim(ki) i-li-kam
```

**Test 특수문자**:
- `„` (U+201E) - subscript 숫자 대신 쓰인 듯 (6회 등장)
- `...` (U+2026) - 결손 표시 (1회)
- `+` - 용도 불명, `me-+e-er`에 등장 (1회)
- `(ki)` - 지명 결정자

**Test 전용 토큰** (train에 없음):
- `aa`, `aa-qí-il...`, `aé-bi„-lá`, `aé-bi„-lá-nim`, `aí-ip-ri-ni`
- `aí-mì-im`, `au-mì`, `au-um-au`, `da-aùr`, `ia-ra-tí-au`
- `ia-tí`, `ia-tù`, `kà-ni-ia`, `kà-ni-ia-ma`, `me-+e-er`
- `mup-pu-um`, `mup-pì-im`, `mup-pì-ni`, `na-aí-ma`, `na-áa-ú`
- `ni-bi„-it`, `qí-bi„-ma`, `ta-áa-me-a-ni`, `u„-mì-im`

**어휘 중복률**: Test 토큰 중 **54.2%만 train에 존재**

### 1.4 핵심 발견: Test text_id는 ORPHAN

`text_id = 332fda50`는 다음 어디에도 존재하지 않음:
- train.csv의 `oare_id`
- published_texts.csv의 `oare_id`
- Sentences_Oare.csv의 `text_uuid`
- 전체 데이터 내 어떤 패턴에서도 미발견

**결론**: Test는 **완전히 hold-out된 문서**로 구성됨.

---

## 2. Train 데이터 분석

### 2.1 구조 개요

| 필드 | 설명 |
|------|------|
| `oare_id` | UUID 형식 |
| `transliteration` | 아카디안 전사 |
| `translation` | 영어 번역 |

### 2.2 통계

| 지표 | 값 |
|------|----|
| 총 행 수 | 1,561 |
| 고유 문서 수 | 1,561 |
| 최소 토큰 | 3 |
| 최대 토큰 | 187 |
| 평균 토큰 | 57.5 |
| 중앙값 | 49 |

### 2.3 길이 분포

```
Percentile | Token Count
-----------|------------
5%         | 12
25%        | 28
50%        | 49
75%        | 84
95%        | 140
```

### 2.4 Train 특수문자

| 패턴 | 빈도 | 의미 |
|------|------|------|
| `...` | 1,687 | 결손 |
| ` x ` | 1,807 | 판독 불가 |
| `[...]` | 210 | 복원 구간 |
| `₄, ₆, ₂` 등 | 2,380+ | subscript 숫자 |

---

## 3. Train vs Test 분포 비교

### 3.1 길이 비교

| 지표 | Train | Test |
|------|-------|------|
| 평균 토큰 | 57.5 | 21.2 |
| 중앙값 | 49 | 17.5 |
| 최소 | 3 | 16 |
| 최대 | 187 | 34 |

**핵심 문제**: Test는 Train 평균의 **약 37% 길이**밖에 되지 않음.

### 3.2 인코딩 차이

| 문자 | Train | Test | 유니코드 |
|------|-------|------|---------|
| `₄` | 많음 | 없음 | U+2084 |
| `„` | 없음 | 있음 | U+201E |
| `+` | 없음 | 있음 | 표준 |

`„`는 subscript 숫자(`₄`)의 대체 표기로 보임:
`qí-bi„-ma` (test) ≈ `qí-bi₄-ma` (train)

### 3.3 어휘 중복

| 지표 | 값 |
|------|----|
| Test 어휘 크기 | 59 |
| Train 어휘 크기 | 11,761 |
| 중복 | 32 |
| Test 전용 | 27 |
| 중복률 | 54.2% |

---

## 4. 보조 데이터 분석

### 4.1 Sentences_Oare_FirstWord_LinNum.csv

**라인 기반 학습 데이터 생성에 핵심**

| 지표 | 값 |
|------|----|
| 총 문장 수 | 9,782 |
| 고유 문서 수 | 1,700 |
| 문서당 문장 수 | 1-201 (평균 5.8) |
| 라인 범위 | 1-113 |

**주요 컬럼**:
- `text_uuid`, `sentence_uuid`
- `translation`
- `line_number`

**Train 연결성**: 1,561개 중 253개만 line-level 연결 가능

**전략적 가치**: Test와 유사한 **라인 세그먼트 학습쌍 생성 가능**

### 4.2 published_texts.csv

| 지표 | 값 |
|------|----|
| 총 텍스트 | 7,953 |
| transliteration 포함 | 7,953 (100%) |
| AICC_translation 포함 | 7,702 (96.8%) |
| <big_gap> 포함 | 3,298 (41.5%) |

**주요 컬럼**:
- `oare_id`
- `transliteration`
- `transliteration_orig`
- `genre_label`

**장르 분포**:
- Unknown: 4,046 (50.9%)
- Letter: 2,261 (28.4%)
- Debt note: 527 (6.6%)
- Note: 218 (2.7%)
- Agreement/Contract: 128 (1.6%)

### 4.3 OA_Lexicon_eBL.csv

| 지표 | 값 |
|------|----|
| 총 항목 | 39,332 |
| 단어 | 25,574 |
| 인명(PN) | 13,424 |
| 지명(GN) | 334 |
| 표기형 | 35,048 |
| 렘마 | 6,353 |

**매핑 가능**: `form` → `norm` → `lexeme`

### 4.4 eBL_Dictionary.csv

| 지표 | 값 |
|------|----|
| 총 항목 | 19,215 |
| 정의 포함 | 19,215 |

### 4.5 publications.csv

| 지표 | 값 |
|------|----|
| 총 페이지 | 216,602 |
| 고유 PDF | 952 |
| Akkadian 포함 페이지 | 31,286 (14.4%) |

---

## 5. 데이터 연결 구조

```
train.csv (oare_id)
    |
    |-- 100% match --> published_texts.csv (oare_id)
    |                        |
    |-- 16.2% match ------> Sentences_Oare.csv (text_uuid)
    |                              |
    |                              |-- line_number 데이터
    |                              |-- translation 데이터
    |
    +-- NO MATCH --> test.csv (text_id: 332fda50)
                         |
                         |-- ORPHAN - Held-out 문서
```

---

## 6. 핵심 인사이트

1. **Test는 라인 세그먼트 + 완전 hold-out 문서**
2. **길이 분포가 다름** → Train 길이 축소 필요
3. **인코딩 정규화 필수** (`„` 등)
4. **Sentences_Oare가 Test 형식에 가장 가까운 브릿지 데이터**
5. **어휘 중복이 낮아 OOV 대응이 필요**

---

## 7. 권장 데이터 전략

### 7.1 전처리 정규화

```python
def normalize_encoding(text):
    text = text.replace('„', '₄')  # 특수 따옴표 → subscript
    return text

def normalize_gaps(text):
    text = text.replace('<big_gap>', '...')
    text = text.replace('{large break}', '...')
    return text
```

### 7.2 학습 데이터 증강

**옵션 A: Sentences_Oare로 라인 세그먼트 생성**
```python
for text_uuid in sentences['text_uuid'].unique():
    # 라인 범위별 세그먼트 생성
    # (라인 1-7, 7-14, ... 형태)
```

**옵션 B: Train 슬라이딩 윈도우**
```python
for doc in train:
    tokens = doc['transliteration'].split()
    for i in range(0, len(tokens), window_size):
        segment = tokens[i:i+window_size]
```

### 7.3 우선순위

1. **HIGH**: Sentences_Oare 기반 세그먼트 학습
2. **MEDIUM**: Train을 짧은 문서로 필터링
3. **LOW**: published_texts 어휘 확장
4. **SUPPORT**: lexicon 기반 OOV 처리

---

## 8. 파일 요약

| 파일 | 행 수 | 핵심 컬럼 | 용도 |
|------|------|-----------|------|
| train.csv | 1,561 | oare_id, transliteration, translation | 주요 학습 |
| test.csv | 4 | text_id, line_start, line_end, transliteration | 추론 대상 |
| sample_submission.csv | 4 | id, translation | 제출 포맷 |
| published_texts.csv | 7,953 | oare_id, transliteration | 어휘 확장 |
| Sentences_Oare.csv | 9,782 | text_uuid, line_number, translation | 라인 세그먼트 |
| OA_Lexicon_eBL.csv | 39,332 | form, norm, lexeme | 표준화 |
| eBL_Dictionary.csv | 19,215 | word, definition | 사전 |
| publications.csv | 216,602 | page_text, has_akkadian | 문헌 OCR |

---

## 9. 샘플 번역(제출 예시)

**Row 0 (lines 1-7)**:
```
Thus Kanesh, say to the -payers, our messenger, every single colony, and the trading stations: A letter of the City has arrived.
```

**Row 1 (lines 7-14)**:
```
In the letter of the City (it is written): From this day on, whoever buys meteoric iron, (the City of) Assur is not part of the profit made, the tithe on it Kanesh will collect.
```

**Row 2 (lines 14-24)**:
```
As soon as you have heard our letter, who(ever) over there has either sold it to a palace, or has offered it to palace officials, or still carries it with him without having yet sold it - all meteoric iron he carries, write the exact amount of every (piece of) meteoric iron, his name and the name of his father in a tablet and send it here with our messenger.
```

**Row 3 (lines 25-30)**:
```
Send a copy of (this) letter of ours to every single colony and to all the trading stations. Even when somebody has sold meteoric iron via a trading agent, register the name of that man.
```

**해석**: 운석 철(meteoric iron) 거래에 관한 **공식 규정 문서**로 보임.

---

## 10. 결론

이 대회는 **도메인 시프트 + 구조 차이** 문제가 핵심이다:
- Test는 line-segment 형태 + hold-out 문서
- Train은 full document 형태
- 어휘 겹침이 낮음

성공을 위해서는:
1. Sentences_Oare 기반 라인 세그먼트 학습
2. 인코딩 정규화
3. OOV 대응 강화
4. 짧은 세그먼트 일반화에 강한 모델 구성

핵심 인사이트는 **Sentences_Oare가 Train ↔ Test 구조 차이를 잇는 유일한 브릿지**라는 점이다.

