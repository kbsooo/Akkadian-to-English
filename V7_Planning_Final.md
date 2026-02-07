# V7 Planning — Final Comprehensive Document

---

## 1. 모델 크기 vs 데이터 크기: ByT5-small이 맞는가?

### 결론: ByT5-small이 현재 데이터 규모에서 올바른 선택이다

논문 근거 (arxiv:2302.14220, TACL 2024):
"Are Character-level Translations Worth the Wait? Comparing ByT5 and mT5 for Machine Translation"에서 162개 모델을 비교한 결과, **400~10K 예제 범위에서 ByT5가 mT5를 chrF++ 기준 +2~5점 능가**한다.

핵심 이유:

**mT5-base의 문제점:** 580M 파라미터 중 약 33%가 250K-vocab 임베딩 레이어에 묶여 있다 (250,112 × 768 × 2 ≈ 384M, 입출력 합산). 실제 트랜스포머 연산에 쓰이는 파라미터는 ~196M. 8,000 예제에서 풀 파인튜닝하면 오버피팅 위험이 높다.

**ByT5-small의 장점:** ~300M 파라미터 중 0.2%만 임베딩에 사용 (384 vocab × 1472 d_model = 565K). 거의 모든 파라미터가 트랜스포머 레이어에 투입됨. 바이트 레벨 토큰화로 아카드어의 다이어크리틱, 특수문자를 자연스럽게 처리.

**그러나 고려할 대안:**

| 옵션 | 총 파라미터 | 유효 트랜스포머 파라미터 | 장점 | 단점 |
|------|-----------|---------------------|------|------|
| ByT5-small (현재) | ~300M | ~299M (99.8%) | 오버피팅 낮음, 바이트 처리 우수 | 12 enc + 4 dec 레이어 |
| ByT5-base | ~580M | ~579M (99.9%) | 2x 용량, 같은 아키텍처 | 학습 시간 2-3x |
| mT5-base | 580M | ~196M (33.8%) | 다국어 사전학습 | 250K 임베딩에 파라미터 낭비 |
| mT5-base + LoRA (r=8) | 580M+~5M | ~5M trainable | 다국어 지식 활용 | 아카드어 토큰화 미지원 |

**V7 권장:** ByT5-small 유지. ~300M 파라미터 중 거의 전부가 트랜스포머에 투입되어, 유효 연산 능력은 mT5-base보다 높음. 데이터가 15K+ 이상으로 늘면 ByT5-base로 전환 검토.

---

## 2. Max Length: 256이면 충분한가?

### 분석 결과

**Hidden test는 sentence-level이다** (대회 포럼 확인). Sentences_Oare (9,772문장)가 hidden test의 best proxy:

```
Hidden test proxy (Sentences_Oare) 길이 분포:
  min=1, max=1,129, mean=73.6, median=61 chars
  p25=38, p75=90, p90=135, p95=175, p99=284
```

**Truncation 비율 비교:**

| Max Length | Train src | Train tgt | Hidden test proxy |
|-----------|-----------|-----------|-------------------|
| 256 chars | 87.4% 커버 | 91.3% 커버 | **98.4% 커버** |
| 384 chars | 93.6% | 96.1% | **99.7%** |
| 512 chars | 96.5% | 98.4% | **99.9%** |

**핵심 발견:** Hidden test의 98.4%가 256 chars 이내에 들어간다. 즉, **현재 max_length=256은 hidden test에는 충분**하다.

**하지만 training에는 부족하다:** Training 데이터의 12.6%가 256에서 잘린다. 이 잘린 데이터로 학습하면 모델이 긴 문장 패턴을 못 배운다.

### V7 권장: 비대칭 설정

```python
max_source_length = 384  # Training: 93.6% 커버 (256에서 6% 추가 회수)
max_target_length = 384  # Training: 96.1% 커버
# ByT5 바이트 토큰화: ASCII 텍스트는 1char=1byte이므로 384 bytes = 384 chars
# Hidden test: 99.7% 커버 (충분)
```

512로 올리면 메모리/속도 대비 추가 이득이 2.9%포인트밖에 안 됨. **384가 최적 균형점**.

---

## 3. 미사용 데이터 완전 분석

### 3.1 Sentences_Oare — 즉시 추출 가능 (+2,360쌍)

**파일:** `Sentences_Oare_FirstWord_LinNum.csv` (9,782문장)

현재 V6에서 4,551개만 추출 성공 (46.6%). **나머지 5,231개 중 2,360개를 추가 추출 가능.**

실패 원인 분석:
- first_word가 transliteration에서 매칭 안 됨 (정규화 차이)
- 해당 text_uuid가 published_texts에 transliteration이 없음
- fuzzy match threshold 0.8이 너무 엄격

**V7 개선안:**
```
1. fuzzy match threshold를 0.7로 낮추기
2. transliteration이 없는 문서 → train.csv에서 가져오기
3. first_word 대신 line_number 기반 분할 시도
4. 매칭 실패 시 전체 문서를 하나의 pair로 사용
```

**예상 추가:** +2,360쌍 (HIGH 품질)

### 3.2 eBL Dictionary — 형태소 사전 (+2,000~3,000)

**파일:** `eBL_Dictionary.csv` (19,215 entries)

구조:
```
word: "-a I"        → definition: "my" (1 sg. pron. suff.)
word: "-am I"       → definition: "to me"
word: "abālu(m) I"  → definition: "to dry (up)"
word: "šarru(m)"    → definition: "king"
```

**활용 방법:**
- 형태소 단위 번역 쌍으로 사용 (word → definition)
- 이건 "문장 번역"이 아니라 "단어/접사 번역"이므로 별도 처리 필요
- V7에서 glossary 재구축의 소스로 사용 가능

**주의:** 훈련 데이터로 직접 넣기보다는, **inference 시 glossary lookup**으로 사용하는 게 더 적절

### 3.3 Published_texts — Transliteration만 있음 (번역 없음)

**파일:** `published_texts.csv` (7,953 documents)

**핵심 발견:** `AICC_translation` 컬럼은 번역 텍스트가 아니라 **OARE API URL**이다!

```
예: "https://oare.byu.edu/api/texts/3e87aad8-.../translations"
```

**따라서 published_texts 자체에서는 번역 쌍을 직접 추출할 수 없다.** 번역은 Sentences_Oare에서만 얻을 수 있으며, published_texts의 역할은 **transliteration 원문 제공** (Sentences_Oare의 text_uuid로 조인 시 first-word anchor로 문장 분할에 사용).

**OARE API 스크래핑 가능성:**
- 이론적으로 7,953개 URL에서 번역을 가져올 수 있음
- 하지만 OARE API 접근성/안정성이 불확실
- 대회 규칙상 "freely & publicly available" 데이터 사용 가능
- **V7에서는 미구현** — Sentences_Oare 조인으로 충분한 데이터 확보 가능

**현재 활용 방식 (V7):**
```
1. published_texts의 transliteration + text_uuid 사용
2. Sentences_Oare와 text_uuid로 조인
3. first_word_spelling으로 문장 위치 앵커링
4. 원문 transliteration에서 문장 범위 추출
```

**V7 데이터 예상:** Sentences_Oare 조인으로 ~7,263쌍 추출 가능 (Path C)

### 3.4 OA_Lexicon_eBL — 전처리 지원

**파일:** `OA_Lexicon_eBL.csv` (39,332 word forms)

번역은 없지만, **form → lemma 매핑** 제공:
```
form: "a-bi4-a"     → lemma: "abum"
form: "KU.BABBAR"   → lemma: "kaspum" (= silver)
```

**활용 방법:**
- eBL Dictionary와 결합: form → lemma → definition
- 이렇게 하면 올바른 glossary 구축 가능:
  ```
  a-bi4-a → abum → "father" (from eBL Dictionary)
  KU.BABBAR → kaspum → "silver"
  ```
- 현재 망가진 glossary의 완벽한 대체

### 3.5 데이터 확장 요약

| 소스 | 방법 | 추가 쌍 | 품질 | 구현 시간 |
|------|------|---------|------|----------|
| Sentences_Oare (미추출분) | minimal-norm 매칭 (V7) | +2,360 | HIGH | 30분 |
| eBL Dict + Lexicon | lemma join → glossary | glossary 3,000+ | MEDIUM | 2시간 |
| OARE API (선택) | URL 스크래핑 | +2,800~5,600 | VARIABLE/불확실 | 1일 |
| **합계 (V7 기본)** | | **+2,360 + glossary** | | **2.5시간** |
| **합계 (API 포함)** | | **+5,160~7,960** | | **1일+** |

**V7 목표 데이터 크기:** ~9,500-10,000쌍 (Path A: 1,561 + Path B: ~1,213 + Path C: ~7,263)

---

## 4. 후처리 (Post-Processing) 완전 가이드

### 4.0 왜 후처리가 중요한가

MT 모델의 raw output에는 체계적인 오류가 있다:
- 이름 철자 불일치 (chrF++에 큰 영향)
- gap 마커 포맷 오류 (평가 시 불일치)
- 숫자 표기 불일치
- 구두점/공백 오류

현재 V6는 **후처리가 전혀 없어서** 이런 오류가 그대로 제출된다.

### 4.1 Phase 1: 규칙 기반 후처리 (구현 시간: 30분, 리스크: 제로)

이건 "반드시 해야 하는" 기본이다. 추가 비용이 거의 없다.

**A. Gap 마커 정규화**

Hidden test의 ground truth는 `<gap>`, `<big_gap>` 포맷을 사용한다 (대회 주최자 확인). 모델이 다른 포맷으로 출력하면 점수가 깎인다.

```python
def normalize_gaps_in_output(translation):
    import re
    # 모델이 "gap", "[gap]", "(gap)" 등으로 출력할 수 있음
    translation = re.sub(r'\b(gap)\b', '<gap>', translation, flags=re.IGNORECASE)
    translation = re.sub(r'\[(gap)\]', '<gap>', translation, flags=re.IGNORECASE)
    translation = re.sub(r'\b(big gap|big_gap|large gap)\b', '<big_gap>',
                         translation, flags=re.IGNORECASE)
    # 연속 gap 병합
    translation = re.sub(r'(<gap>\s*){2,}', '<big_gap> ', translation)
    return translation.strip()
```

**B. 숫자 포맷 정규화**

대회 데이터에서 분수가 소수점으로 표기됨: `0.33333` = ⅓, `0.66666` = ⅔

```python
def normalize_numbers(translation):
    import re
    # 0.33333 → ⅓ 등의 변환은 필요 없음 - 원래 포맷 유지
    # 하지만 모델이 "one-third" 등으로 출력할 수 있으므로:
    translation = re.sub(r'\bone[- ]third\b', '0.33333', translation)
    translation = re.sub(r'\btwo[- ]thirds?\b', '0.66666', translation)
    translation = re.sub(r'\bone[- ]half\b', '0.5', translation)
    return translation
```

**C. 공백/구두점 정리**

```python
def cleanup_whitespace(translation):
    import re
    translation = re.sub(r'\s+([.,;:!?)])', r'\1', translation)  # 구두점 앞 공백 제거
    translation = re.sub(r'([(\[])\s+', r'\1', translation)       # 괄호 뒤 공백 제거
    translation = re.sub(r'\s{2,}', ' ', translation)             # 중복 공백
    translation = re.sub(r'"{2,}', '"', translation)              # 중복 따옴표
    return translation.strip()
```

**예상 점수 향상:** +0.5~1.0 (작지만 무료)

### 4.2 Phase 2: Onomasticon 이름 교정 (구현 시간: 2시간, 예상 +2~5점)

대회 주최자가 명시적으로 "named entities가 가장 큰 오류 원인"이라고 했다.

**Onomasticon이란:** 아카드어 고유명사(인명, 지명, 신명) 정규화 사전. 대회 주최자가 Kaggle 보조 데이터셋으로 공개함.

**작동 방식:**

```python
import difflib

class NameCorrector:
    def __init__(self, onomasticon_path):
        """
        onomasticon: {"Aššur-imittī": "Aššur-imittī", "Kanesh": "Kanesh", ...}
        """
        self.names = self._load(onomasticon_path)
        self.lower_map = {n.lower(): n for n in self.names}

    def correct(self, translation):
        words = translation.split()
        corrected = []
        for word in words:
            if word[0:1].isupper():  # 대문자로 시작 = 고유명사 후보
                fixed = self._fix_name(word)
                corrected.append(fixed)
            else:
                corrected.append(word)
        return ' '.join(corrected)

    def _fix_name(self, word):
        # 1. 정확한 매칭 (대소문자 무시)
        if word.lower() in self.lower_map:
            return self.lower_map[word.lower()]

        # 2. Fuzzy 매칭 (유사도 85% 이상)
        matches = difflib.get_close_matches(
            word.lower(),
            self.lower_map.keys(),
            n=1, cutoff=0.85
        )
        if matches:
            return self.lower_map[matches[0]]

        return word  # 매칭 안 되면 원래대로
```

**Onomasticon 데이터 확보:**
- Kaggle 데이터셋: `deeppast/old-assyrian-grammars-and-other-resources`
- 파일: `Onomasticon.csv` (불완전하지만 유효)
- OA_Lexicon에서 고유명사 추출로 보완 가능

### 4.3 Phase 3: MBR Decoding (구현 시간: 1시간, 예상 +1~3점)

Ensemble보다 간단하고 효과적. **같은 모델에서 여러 번역을 생성**하고, 서로 가장 합의가 높은 것을 선택.

**원리:** N개 번역 후보를 생성 → 각 후보를 다른 모든 후보와 비교 → 평균 유사도가 가장 높은 후보 선택 (= 가장 "합의된" 번역)

```python
from sacrebleu.metrics import CHRF

def mbr_decode(model, tokenizer, source, n_samples=8, device='cuda'):
    """
    MBR decoding: 가장 합의도가 높은 번역 선택
    """
    chrf = CHRF(word_order=2)

    # Step 1: N개 후보 생성 (다양한 방법)
    candidates = []

    # Beam search (top-N)
    outputs = model.generate(
        **tokenizer(source, return_tensors="pt").to(device),
        num_beams=n_samples,
        num_return_sequences=n_samples,
        max_length=384,
    )
    for out in outputs:
        candidates.append(tokenizer.decode(out, skip_special_tokens=True))

    # Step 2: 각 후보의 기대 점수 계산
    scores = []
    for i, cand in enumerate(candidates):
        # 다른 모든 후보를 pseudo-reference로 사용
        refs = [c for j, c in enumerate(candidates) if j != i]
        # chrF++로 유사도 측정 (BLEU보다 안정적)
        avg_score = sum(
            chrf.corpus_score([cand], [[r]]).score for r in refs
        ) / len(refs)
        scores.append(avg_score)

    # Step 3: 가장 높은 점수의 후보 반환
    best_idx = scores.index(max(scores))
    return candidates[best_idx]
```

**시간 비용:** N=8이면 inference 시간 ~2x (beam search가 이미 병렬). Hidden test 1000문장 기준 추가 30분 정도.

**장점:** 추가 모델 불필요, hallucination 감소, 안정적

### 4.4 Phase 4: LLM 후처리 (구현 시간: 3시간, 예상 +3~8점, 리스크 있음)

**적합한 모델:** TinyLlama-1.1B (2.2GB, T4에서 ~50 tok/s)

**9시간 예산 내 가능 여부:**
- 메인 모델 (ByT5-small) 학습: ~3시간 (Colab T4)
- 메인 모델 inference: ~30분
- LLM 로드 + 후처리: ~1시간 (1000문장 × 2초)
- **총: ~4.5시간 → 9시간 내 충분**

**하지만 주의:**
- Kaggle은 **inference 노트북에서만 9시간** (학습은 별도)
- Inference 노트북에서 ByT5 + TinyLlama 동시 로드 시 메모리 문제 가능
- 순차 로드로 해결: ByT5 inference → 메모리 해제 → TinyLlama 로드 → 후처리

```python
# Inference 노트북 흐름
# Step 1: ByT5로 기본 번역
translations = byt5_translate_all(test_data)

# Step 2: ByT5 메모리 해제
del model, tokenizer
torch.cuda.empty_cache()

# Step 3: TinyLlama로 후처리
llm = load_tinyllama()  # FP16, 2.2GB
for i, trans in enumerate(translations):
    prompt = f"Fix this Akkadian-English translation:\n{trans}\nFixed:"
    translations[i] = llm_generate(prompt)

del llm
torch.cuda.empty_cache()
```

**리스크:** LLM이 정확한 번역을 오히려 망칠 수 있음. 반드시 validation set에서 검증 후 사용.

### 4.5 Phase 5: 문서 내 일관성 (구현 시간: 30분, 예상 +0.5~1점)

Test에서 같은 text_id의 세그먼트들은 같은 이름/용어를 일관되게 번역해야 한다.

```python
def enforce_document_consistency(translations, text_ids):
    """같은 문서의 세그먼트 간 이름 일관성"""
    from collections import defaultdict, Counter

    doc_groups = defaultdict(list)
    for i, tid in enumerate(text_ids):
        doc_groups[tid].append(i)

    for tid, indices in doc_groups.items():
        if len(indices) <= 1:
            continue

        # 각 세그먼트에서 고유명사 추출
        name_variants = defaultdict(Counter)
        for idx in indices:
            for word in translations[idx].split():
                if word[0:1].isupper() and len(word) > 2:
                    # 비슷한 이름끼리 그룹핑
                    for existing in name_variants:
                        if difflib.SequenceMatcher(
                            None, word.lower(), existing.lower()
                        ).ratio() > 0.8:
                            name_variants[existing][word] += 1
                            break
                    else:
                        name_variants[word][word] += 1

        # 가장 빈번한 형태로 통일
        replacements = {}
        for canonical, variants in name_variants.items():
            if len(variants) > 1:
                most_common = variants.most_common(1)[0][0]
                for variant in variants:
                    if variant != most_common:
                        replacements[variant] = most_common

        # 적용
        for idx in indices:
            for wrong, right in replacements.items():
                translations[idx] = translations[idx].replace(wrong, right)

    return translations
```

### 4.6 후처리 종합 요약

| Phase | 기법 | 구현시간 | 점수향상 | 리스크 | 우선순위 |
|-------|------|---------|---------|--------|---------|
| 1 | 규칙 기반 (gap, 숫자, 공백) | 30분 | +0.5~1.0 | 없음 | **필수** |
| 2 | Onomasticon 이름 교정 | 2시간 | +2~5 | 낮음 | **필수** |
| 3 | MBR Decoding | 1시간 | +1~3 | 낮음 | 권장 |
| 4 | LLM 후처리 | 3시간 | +3~8 | 높음 | 선택 |
| 5 | 문서 내 일관성 | 30분 | +0.5~1 | 낮음 | 권장 |

**V7 최소 구현:** Phase 1 + Phase 2 = 2.5시간, 예상 +3~6점
**V7 최대 구현:** Phase 1~5 전부 = 7시간, 예상 +7~18점

---

## 5. Glossary 재구축 방안

### 현재 문제 (다시 정리)

현재 `build_glossary()`는 첫 단어끼리 매핑:
```
KU.BABBAR (silver) → The (WRONG)
DUMU (son) → The (WRONG)
```

### 올바른 Glossary 구축

**방법: OA_Lexicon + eBL_Dictionary 결합**

```
OA_Lexicon: form → lemma
  "KU.BABBAR" → "kaspum"
  "DUMU" → "mārum"
  "a-na" → "ana"

eBL_Dictionary: lemma → definition
  "kaspum" → "silver"
  "mārum" → "son; daughter"
  "ana" → "to; for; into"

결합: form → lemma → definition
  "KU.BABBAR" → "silver"
  "DUMU" → "son"
  "a-na" → "to"
```

이렇게 하면 **언어학적으로 올바른 glossary**가 만들어진다.

**구현:**
```python
def build_proper_glossary(lexicon_path, dictionary_path):
    lexicon = pd.read_csv(lexicon_path)
    dictionary = pd.read_csv(dictionary_path)

    # lemma → definition 매핑
    lemma_to_def = {}
    for _, row in dictionary.iterrows():
        word = str(row.get('word', '')).strip()
        definition = str(row.get('definition', '')).strip()
        if word and definition:
            # "abālu(m) I" → "abālum" (정규화)
            clean_lemma = re.sub(r'\(.*?\)', '', word).strip().split()[0]
            lemma_to_def[clean_lemma.lower()] = definition

    # form → lemma → definition
    glossary = {}
    for _, row in lexicon.iterrows():
        form = str(row.get('form', '')).strip()
        lemma = str(row.get('lemma', '')).strip()
        if form and lemma:
            clean_lemma = lemma.lower().strip()
            if clean_lemma in lemma_to_def:
                glossary[form] = lemma_to_def[clean_lemma]

    return glossary
```

**예상 glossary 크기:** 3,000~5,000 entries (현재 1,060의 3~5배, 정확도 대폭 향상)

### Glossary 사용 방법 (V7)

Training 시 glossary prompting은 **제거**하는 것을 권장. 이유:
1. 올바른 glossary라도, 모델이 glossary에 의존하면 일반화 능력이 떨어짐
2. Inference 시 glossary가 없는 단어에서 성능 저하

대신, **inference 시에만 constrained decoding이나 post-processing으로 활용:**
```python
def glossary_postprocess(translation, source, glossary):
    """번역에서 glossary 단어가 누락되었으면 보완"""
    src_tokens = source.split()
    for token in src_tokens:
        if token in glossary:
            expected_word = glossary[token]
            if expected_word.lower() not in translation.lower():
                # 누락된 glossary 단어 로깅 (자동 삽입은 위험)
                pass
    return translation
```

---

## 6. 다이어크리틱 보존 재검토

### 대회 주최자의 명확한 지침

> "Converting diacritics into ASCII sequences before training was done in the WRONG DIRECTION."

하지만 이건 **ASCII 시퀀스 (sz→š 등)** 에 대한 이야기이지, **ByT5에서의 유니코드 처리**와는 다른 맥락.

### ByT5에서의 실제 영향

ByT5는 **바이트 레벨** 토크나이저:
- `š` = 2 bytes (0xC5 0xA1)
- `s` = 1 byte (0x73)

현재 V6는 `š→s`로 변환해서 **2바이트를 1바이트로 줄이고 있다**. 이것은:
- **장점:** 입력 길이 절약, 더 많은 컨텍스트 처리 가능
- **단점:** `š`와 `s`의 구분 정보 소실

### V7 권장: 선택적 보존

```
보존해야 할 것:
  - š, Š (shin - 별개의 자음)
  - ṣ, Ṣ (emphatic s - 별개의 자음)
  - ṭ, Ṭ (emphatic t - 별개의 자음)
  → 이들은 아카드어에서 다른 음소를 나타냄

변환해야 할 것 (대회 지침):
  - Ḫ/ḫ → H/h (test에서 H/h만 사용)

변환해도 되는 것:
  - á→a, é→e, ú→u 등 (모음 액센트)
  → 이들은 transliteration 규약 차이일 뿐 같은 음소
  → 하지만 test에도 이들이 포함되어 있으므로 보존이 안전
```

**결론:** 자음 다이어크리틱 (š, ṣ, ṭ)은 보존, `Ḫ/ḫ→H/h`만 변환, 모음 액센트는 test 포맷에 맞춰 판단.

이건 **실험으로 검증**해야 한다. V7에서 A/B 테스트 권장:
- A: 현재 방식 (모두 strip)
- B: 자음만 보존
- validation geo_mean으로 비교

---

## 7. V7 구현 로드맵

### Week 1: 데이터 파이프라인

```
Day 1-2: Sentences_Oare 추가 추출
  - fuzzy threshold 0.7로 완화
  - line_number 기반 대체 분할 구현
  - 예상: +2,360쌍

Day 3: Glossary 재구축
  - OA_Lexicon + eBL_Dictionary join
  - 정확도 검증 (랜덤 100개 수동 확인)

Day 4: 데이터 정제
  - 78개 중복 제거
  - 385개 비율 이상치 제거
  - 222개 누락 원본 train 복구

Day 5: (선택) Published_texts 스크래핑 시도
  - OARE API에서 번역 가져오기
  - 품질 필터링
```

### Week 2: 모델 학습

```
Day 1: 다이어크리틱 A/B 테스트
  - 정규화 A (현재) vs B (자음 보존)
  - 5 에폭 빠른 비교

Day 2-3: V7 모델 학습
  - ByT5-small, max_len=384
  - V7 확장 데이터 (9,751+쌍)
  - glossary prompting OFF
  - 15 에폭, early stopping patience=3

Day 4: 후처리 파이프라인 구축
  - Phase 1: 규칙 기반
  - Phase 2: Onomasticon
  - Phase 3: MBR decoding

Day 5: Kaggle 제출 + 검증
```

### Week 3: 최적화

```
- LLM 후처리 실험 (Phase 4)
- Beam search 파라미터 튜닝
- 앙상블 (다른 seed로 3개 모델)
- 최종 제출
```

---

## 8. 예상 점수 궤적 (수정)

| 변경사항 | 예상 점수 | 근거 |
|---------|----------|------|
| 현재 V6 | 11.0 | 기준 |
| + 망가진 glossary 제거 | 13-15 | 독성 신호 제거 |
| + 데이터 확장 (9,751쌍) | 16-18 | +32% 학습 데이터 |
| + max_len 384 | 18-20 | truncation 감소 |
| + 다이어크리틱 보존 (검증 필요) | 20-23 | test 포맷 매칭 |
| + 규칙 기반 후처리 | 23-24 | gap/숫자/공백 정리 |
| + Onomasticon 이름 교정 | 26-29 | #1 오류 원인 해결 |
| + MBR decoding | 29-31 | 안정적 번역 선택 |
| + Glossary 재구축 (inference 시) | 31-33 | 어휘 커버리지 향상 |
| + LLM 후처리 (선택) | 33-36 | 문법/일관성 개선 |

**보수적 목표:** 29-31 (Phase 1-3 후처리 + 데이터 확장)
**공격적 목표:** 33-36 (전체 파이프라인)

---

## 9. 핵심 실행 파일 목록

```
만들어야 할 파일:
  src/v7/build_v7_data.py        ← 데이터 파이프라인 (최우선)
  src/v7/build_proper_glossary.py ← OA_Lexicon+eBL Dict glossary
  src/v7/akkadian_v7_train.py    ← 학습 (V6 기반 수정)
  src/v7/akkadian_v7_infer.py    ← 추론 + 후처리 파이프라인
  src/v7/postprocess.py          ← 후처리 모듈
  src/v7/name_corrector.py       ← Onomasticon 이름 교정

다운로드해야 할 데이터:
  Kaggle: deeppast/old-assyrian-grammars-and-other-resources
    → Onomasticon.csv
```
