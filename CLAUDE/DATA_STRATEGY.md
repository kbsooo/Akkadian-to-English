# Akkadian Translation Data Strategy
## Deep Past Initiative - Kaggle Competition

---

## Executive Summary

이 대회의 **핵심 과제는 Train/Test 데이터 형식 불일치**입니다:
- **Train**: 문서 레벨 (1,561개 문서, 평균 426자)
- **Test**: 문장 레벨 (~4,000개 문장, 평균 169자)

데이터 전략이 모델 선택보다 더 중요한 이유:
1. 아무리 좋은 모델도 문장 레벨 데이터 없이는 최적 성능 발휘 불가
2. 보조 데이터(Lexicon, Dictionary, Sentences)를 활용하면 데이터 품질 대폭 향상 가능
3. 7,953개의 미번역 텍스트(published_texts)를 활용한 증강 잠재력 존재

---

## 1. 데이터 현황 분석

### 1.1 Core Training Data (train.csv)

| 항목 | 값 |
|------|-----|
| 문서 수 | 1,561 |
| Transliteration 평균 길이 | 426자 |
| Translation 평균 길이 | 500자 |
| 고유 단어 수 | 11,378 |

**주요 특징:**
- **Sumerogram** 빈도 높음: KÙ.BABBAR (3,395회), DUMU (1,937회)
- **결정자(Determinatives)**: (d) 신명 482회, (ki) 지명 383회
- **분수 표기**: 0.33333, 0.5 등 1,682회
- **불명확 표기 'x'**: 2,695회

**장르 분포:**
- 상업 문서 (은, 직물, 구리 거래): ~60%
- 서신 (um-ma, qí-bi-ma): ~25%
- 법적 문서 (KIŠIB, IGI): ~15%

### 1.2 Sentence-Level Annotations (Sentences_Oare_FirstWord_LinNum.csv)

| 항목 | 값 |
|------|-----|
| 총 문장 수 | 9,782 |
| 고유 문서 수 | 1,700 |
| **Train과 중첩** | **253개 문서만** |
| 문장당 평균 번역 길이 | 74자 |

**핵심 발견:**
- Train 1,561개 중 **253개만** 문장 레벨 annotation 존재
- 나머지 **1,308개 문서는 자체 문장 분리 필요**
- 문장 경계 마커: um-ma (1,311), IGI (1,128), KIŠIB (520)

### 1.3 Supplementary Resources

| 리소스 | 크기 | 활용도 |
|--------|------|--------|
| OA_Lexicon_eBL.csv | 39,332 entries | 정규화, 품사태깅 |
| eBL_Dictionary.csv | 19,215 words | 용어 참조 |
| published_texts.csv | 7,953 texts (gap 없음: 3,836) | Back-translation |
| publications.csv | 880 PDFs OCR | 추가 병렬 데이터 추출 |

### 1.4 Test Data 특성

**현재 더미 데이터 분석:**
- 문장 레벨 (5-10 라인 span)
- 평균 169자
- OCR 오류 패턴: „, …, + 등 특수문자

**OOV 위험:**
- Test에만 있는 문자: 1개 (「)
- → ByT5 같은 character-level 모델 필요성 확인

---

## 2. 데이터 전처리 전략

### 2.1 Phase 1: Cleaning & Normalization

```python
# 1. Sumerogram 정규화
SUMEROGRAM_MAP = {
    'KÙ.BABBAR': '[SILVER]',
    'KÙ.GI': '[GOLD]',
    'URUDU': '[COPPER]',
    'AN.NA': '[TIN]',
    'TÚG': '[TEXTILE]',
    'ANŠE': '[DONKEY]',
    'GÍN': '[SHEKEL]',
    'ITU.KAM': '[MONTH]',
    'DUMU': '[SON]',
    'IGI': '[WITNESS]',
    'KIŠIB': '[SEAL]',
}

# 2. 결정자 표준화
def normalize_determinatives(text):
    text = re.sub(r'\(d\)', '{d}', text)  # 신명
    text = re.sub(r'\(ki\)', '{ki}', text)  # 지명
    text = re.sub(r'\(f\)', '{f}', text)   # 여성명
    return text

# 3. 불명확 표기 처리
def handle_unclear(text):
    text = re.sub(r'\bx\b', '[?]', text)
    text = re.sub(r'<gap>', '[GAP]', text)
    text = re.sub(r'<big_gap>', '[BIG_GAP]', text)
    return text

# 4. 숫자 정규화
def normalize_numbers(text):
    # 0.33333 → 1/3, 0.5 → 1/2, 0.66666 → 2/3
    text = re.sub(r'0\.33+', '⅓', text)
    text = re.sub(r'0\.5', '½', text)
    text = re.sub(r'0\.66+', '⅔', text)
    return text
```

### 2.2 Phase 2: Sentence Segmentation

**전략 1: Rule-based Segmentation (1,308개 문서용)**

```python
SENTENCE_BOUNDARIES = [
    r'um-ma\s+\w+-ma',     # 인용문 시작: "From X:"
    r'qí-bi(?:₄)?-ma',     # 말하기: "say:"
    r'\bIGI\b',            # 증인 목록 시작
    r'\bKIŠIB\b',          # 인장 목록
    r'li-mu-um',           # 연대 표기
    r'ITU\.KAM',           # 월 표기
]

def segment_document(transliteration, translation):
    """
    문서를 문장으로 분리하고 alignment 수행
    """
    # 1. 경계 마커로 분리
    segments = split_by_markers(transliteration, SENTENCE_BOUNDARIES)

    # 2. Translation도 유사하게 분리
    # "From X:", "Witnessed by", "Month:" 등으로 분리

    # 3. Dynamic Time Warping으로 alignment
    aligned_pairs = align_segments(segments, translation_segments)

    return aligned_pairs
```

**전략 2: Sentences_Oare 데이터 활용 (253개 문서용)**

```python
def extract_from_sentences_file(train_doc, sentences_df):
    """
    기존 sentence annotation 활용
    """
    doc_sentences = sentences_df[
        sentences_df['text_uuid'] == train_doc['oare_id']
    ]

    pairs = []
    for _, sent in doc_sentences.iterrows():
        # line_number 기반으로 transliteration 추출
        trans_segment = extract_lines(
            train_doc['transliteration'],
            sent['line_number']
        )
        pairs.append({
            'transliteration': trans_segment,
            'translation': sent['translation'],
            'first_word': sent['first_word_transcription']
        })
    return pairs
```

### 2.3 Phase 3: Lexicon Integration

**OA_Lexicon을 활용한 단어 레벨 정규화:**

```python
# Lexicon 기반 정규화 매핑 생성
lexicon = pd.read_csv('OA_Lexicon_eBL.csv')

# form → norm 매핑 (35,048 → 6,353 정규화)
NORM_MAP = dict(zip(lexicon['form'], lexicon['norm']))

def normalize_with_lexicon(text):
    words = text.split()
    normalized = []
    for word in words:
        if word in NORM_MAP:
            normalized.append(NORM_MAP[word])
        else:
            normalized.append(word)
    return ' '.join(normalized)
```

---

## 3. 데이터 증강 전략

### 3.1 Strategy A: Sentence-Level Data Generation

**목표:** 1,561개 문서 → ~6,000개 문장 쌍

| 소스 | 예상 문장 수 |
|------|-------------|
| Sentences_Oare (253개 문서) | ~1,200 |
| Rule-based 분리 (1,308개 문서) | ~4,500 |
| **총계** | **~5,700** |

### 3.2 Strategy B: Back-Translation Augmentation

**published_texts (3,836개 clean) 활용:**

```python
# Phase 1: 모델 학습 후 published_texts 번역
def back_translation_augment():
    # 1. Train으로 초기 모델 학습
    model = train_initial_model(train_data)

    # 2. published_texts 번역 (pseudo-labeling)
    pseudo_pairs = []
    for text in published_texts:
        if quality_check(text):  # gap 없는 텍스트만
            translation = model.translate(text['transliteration'])
            confidence = model.get_confidence(translation)

            if confidence > 0.7:  # 고신뢰도만 사용
                pseudo_pairs.append({
                    'transliteration': text['transliteration'],
                    'translation': translation,
                    'source': 'pseudo'
                })

    # 3. 재학습
    model = retrain_with_pseudo(train_data + pseudo_pairs)
    return model
```

**예상 증강량:** 고신뢰도 번역 ~1,500-2,500개

### 3.3 Strategy C: Lexicon-based Substitution

```python
def lexicon_augmentation(sentence_pair):
    """
    동의어/변형 대체로 데이터 증강
    """
    trans, eng = sentence_pair

    # Lexicon에서 같은 lexeme를 가진 다른 form 찾기
    augmented = []
    for word in trans.split():
        if word in LEXEME_MAP:
            lexeme = LEXEME_MAP[word]
            variants = get_variants(lexeme)
            for var in variants[:2]:  # 최대 2개 변형
                new_trans = trans.replace(word, var)
                augmented.append((new_trans, eng))

    return augmented
```

### 3.4 Strategy D: Publications Mining (Advanced)

**880개 학술 PDF에서 병렬 데이터 추출:**

```python
# publications.csv에서 병렬 코퍼스 추출
def mine_publications():
    pubs = pd.read_csv('publications.csv')

    # 아카드어 포함 페이지만 필터링
    akkadian_pages = pubs[pubs['has_akkadian'] == True]

    # 패턴 매칭으로 transliteration-translation 쌍 추출
    # 학술 논문 형식: "a-na DUMU-šu qí-bi-ma" (to his son say:)
    patterns = [
        r'"([^"]+)"\s*\(([^)]+)\)',  # "akkadian" (translation)
        r'([a-z\-₀-₉]+(?:\s+[a-z\-₀-₉]+)+)(?:,\s*[""]([^""]+)[""])',
    ]

    # 추출 및 품질 검증
    extracted_pairs = extract_with_patterns(akkadian_pages, patterns)
    return filter_quality(extracted_pairs)
```

---

## 4. Train/Test 불일치 해결 전략

### 4.1 Document-to-Sentence Curriculum

```
Stage 1: 문장 레벨 학습 (Primary)
├── Sentences_Oare 데이터 (1,200 문장)
├── Rule-based 분리 데이터 (4,500 문장)
└── 총 ~5,700 문장 쌍

Stage 2: 문서 레벨 Fine-tuning (Secondary)
├── 전체 문서로 context 이해 강화
└── 긴 문서 → 짧은 문장 생성 능력 향상

Stage 3: Pseudo-labeling (Optional)
├── published_texts 번역
└── 고신뢰도 결과만 추가 학습
```

### 4.2 Multi-Task Learning

```python
# Task 1: Sentence Translation (Primary)
# Task 2: Document Summarization (Secondary)
# Task 3: Word-level Translation (Auxiliary)

class MultiTaskModel:
    def forward(self, input, task='sentence'):
        if task == 'sentence':
            # 문장 번역 (Test와 동일 형식)
            return self.translate_sentence(input)
        elif task == 'document':
            # 문서 번역 (context 학습용)
            return self.translate_document(input)
        elif task == 'word':
            # 단어 번역 (Lexicon 활용)
            return self.translate_word(input)
```

---

## 5. 품질 검증 파이프라인

### 5.1 Data Quality Checks

```python
def validate_pair(trans, eng):
    checks = {
        # 길이 비율 체크 (Akkadian : English ≈ 0.8-1.2)
        'length_ratio': 0.5 < len(trans)/len(eng) < 2.0,

        # 숫자 일관성 (숫자는 보존되어야 함)
        'number_match': extract_numbers(trans) == extract_numbers(eng),

        # 고유명사 일관성 (대문자로 시작하는 이름들)
        'name_overlap': check_name_overlap(trans, eng),

        # Sumerogram 번역 확인
        'sumerogram_translated': check_sumerogram_translation(trans, eng),
    }
    return all(checks.values())
```

### 5.2 Alignment Verification

```python
def verify_alignment(sentence_pairs):
    """
    문장 분리 후 alignment 품질 검증
    """
    verified = []
    for trans, eng in sentence_pairs:
        # Cross-entropy 기반 alignment score
        score = compute_alignment_score(trans, eng)

        if score > THRESHOLD:
            verified.append((trans, eng))
        else:
            # Manual review queue에 추가
            add_to_review(trans, eng, score)

    return verified
```

---

## 6. 구현 우선순위

### Phase 1: 필수 (Week 1)
1. ✅ 기본 전처리 파이프라인 구축
2. ✅ Sentence segmentation 알고리즘 구현
3. ✅ Lexicon 정규화 적용

### Phase 2: 중요 (Week 2)
4. 📋 Sentences_Oare 데이터 활용 문장 추출
5. 📋 Rule-based 문장 분리 및 alignment
6. 📋 품질 검증 파이프라인

### Phase 3: 고급 (Week 3-4)
7. 📋 Back-translation augmentation
8. 📋 Publications mining
9. 📋 Multi-task learning 셋업

---

## 7. 예상 데이터 규모

| 데이터셋 | 크기 | 용도 |
|----------|------|------|
| Original Train | 1,561 docs | 문서 레벨 학습 |
| Sentence-level Train | ~5,700 sentences | **Primary 학습** |
| Pseudo-labeled | ~2,000 sentences | 증강 |
| Lexicon-augmented | ~10,000 variants | 로버스트성 향상 |
| **Total** | **~18,000 samples** | - |

---

## 8. 핵심 권장사항

### DO ✅
1. **문장 레벨 데이터 생성 최우선** - Test 형식과 일치시키기
2. **Lexicon 적극 활용** - 정규화로 OOV 감소
3. **Sumerogram 일관 처리** - 특수 토큰으로 표준화
4. **품질 > 양** - 저품질 데이터는 성능 저하 유발

### DON'T ❌
1. ~~문서 레벨만으로 학습~~ - Test는 문장 레벨
2. ~~Gap 있는 published_texts 사용~~ - 노이즈 유발
3. ~~OCR 오류 무시~~ - Test에 특수문자 존재

---

## 결론

**데이터 전략의 핵심:**

1. **Train/Test 불일치 해결**이 가장 중요
   - 1,561개 문서 → ~5,700개 문장 변환 필수

2. **보조 리소스 적극 활용**
   - Lexicon: 정규화, 증강
   - Sentences: 문장 레벨 annotation
   - Dictionary: 용어 참조

3. **단계적 증강**
   - 먼저 고품질 sentence pairs 확보
   - 이후 pseudo-labeling으로 확장

이 전략을 따르면 모델 성능을 **30-50% 향상**시킬 수 있을 것으로 예상됩니다.
