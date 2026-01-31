# Akkadian V2 마스터 플랜

## 현재 문제점 요약

### 1. Train/Test 데이터 레벨 불일치 (가장 큰 문제)
| 구분 | Train.csv | Test.csv |
|------|-----------|----------|
| **레벨** | 문서 (Document) | 문장 (Sentence) |
| **평균 길이** | ~426자 | ~169자 |
| **전사 스타일** | 다양 | 표준화됨 |

### 2. 전사(Transliteration) 스타일 차이
- Train: 다양한 출처, 비표준 기호, OCR 아티팩트
- Test: 표준화된 아카드어 전사

### 3. 모델/토크나이저 불일치
- 학습 모델: ByT5-Small
- 업로드 모델: ByT5-Base (또는 토크나이저 설정 오류)

---

## V2 목표

1. **Train/Test 분포 일치**: 문장 레벨 데이터로 학습
2. **전처리 통일**: Train과 Test에 동일한 전처리 적용
3. **데이터 증강**: publications.csv 활용
4. **일관된 파이프라인**: 학습-추론 완전 일치

---

## Phase 1: 데이터 준비

### 1.1 Train 데이터 문장 분리

**목표**: train.csv의 문서 레벨 데이터를 문장 레벨로 분리

```python
# 접근법
1. 문서 → 문장 분리 (줄바꿈, 마침표 기준)
2. transliteration-translation 정렬 유지
3. 품질 필터링 (너무 짧거나 긴 문장 제거)
```

**기존 Tier3 데이터 활용**:
- `sentence_pairs_q70_pattern.csv`: 2,350개 문장 쌍 (이미 준비됨)
- 품질 점수 >= 0.7, 패턴 필터 적용됨

### 1.2 전처리 함수 표준화

**통합 전처리 모듈**: `preprocessing.py`

```python
def normalize_transliteration(text: str) -> str:
    """Train과 Test 모두에 적용되는 표준 전처리"""

    # 1. Unicode 정규화
    text = unicodedata.normalize("NFC", text)

    # 2. 특수 H 문자 정규화
    text = text.replace("\u1E2A", "H").replace("\u1E2B", "h")

    # 3. 아래첨자 → 숫자
    text = text.translate(SUBSCRIPT_MAP)

    # 4. 손상/갭 처리
    text = text.replace("…", " <gap> ")
    text = re.sub(r"\.\.\.+", " <gap> ", text)
    text = re.sub(r"\[([^\]]*)\]", " <gap> ", text)

    # 5. OCR 아티팩트 정리 (NEW - Test에서 발견된 문자)
    text = text.replace("„", '"')   # U+201E → 표준 따옴표
    text = text.replace("u„", "ù")  # 일반적인 OCR 오류 패턴

    # 6. 미확인 기호
    text = re.sub(r"\bx\b", " <unk> ", text)

    # 7. 편집 기호 제거
    text = re.sub(r"[!?/]", " ", text)

    # 8. 공백 정규화
    text = re.sub(r"\s+", " ", text).strip()

    return text
```

### 1.3 Publications 데이터 증강

**publications.csv 분석 결과**:
- 216,602 페이지
- 31,286 페이지에 아카드어 플래그
- OCR 품질: 불균일 (직접 병렬 추출 어려움)

**증강 전략: Back-Translation**

```
1단계: V2 기본 모델 학습 (Tier3 데이터)
    ↓
2단계: published_texts.csv에서 깨끗한 전사 추출 (3,836개)
    ↓
3단계: 학습된 모델로 전사 → 영어 번역 (pseudo-label)
    ↓
4단계: 품질 필터링 (confidence threshold)
    ↓
5단계: 증강 데이터로 Fine-tuning
```

**published_texts.csv 활용**:
- 7,953 텍스트 중 3,836개 깨끗한 전사 (갭 없음)
- 이들을 pseudo-label로 영어 번역 생성

---

## Phase 2: 모델 학습

### 2.1 기본 설정

```python
MODEL_NAME = "google/byt5-small"  # 일관되게 small 사용
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 256
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
EPOCHS = 10
```

### 2.2 데이터 분할

```python
# GroupKFold로 text_id 기준 분할 (데이터 누수 방지)
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(X, y, groups=text_ids))
```

### 2.3 학습 파이프라인

```python
training_args = Seq2SeqTrainingArguments(
    output_dir="./outputs/v2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=True,  # GPU 환경
    load_best_model_at_end=True,
    metric_for_best_model="geometric_mean",
    greater_is_better=True,
)
```

### 2.4 평가 지표

```python
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Decode
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # BLEU
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score

    # chrF++
    chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels]).score

    # Geometric mean (대회 메트릭)
    geometric_mean = (bleu * chrf) ** 0.5

    return {
        "bleu": bleu,
        "chrf": chrf,
        "geometric_mean": geometric_mean,
    }
```

---

## Phase 3: 추론 & 제출

### 3.1 추론 코드

```python
# 토크나이저: 학습과 동일하게 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# 모델 로드
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

# 전처리: 학습과 동일한 함수 사용
from preprocessing import normalize_transliteration
```

### 3.2 체크리스트

- [ ] `config.json` 확인: d_model=1472 (ByT5-Small)
- [ ] `tokenizer_config.json` 확인: extra_ids=125
- [ ] 전처리 함수 동일한지 확인
- [ ] vocab_size 일치 확인 (384)

---

## Phase 4: 개선 이터레이션

### 4.1 앙상블

```python
# 다양한 설정으로 학습한 모델들 앙상블
models = [
    "v2_small_tier3",      # 기본
    "v2_small_augmented",  # 증강 데이터
    "v2_base_tier3",       # ByT5-Base
]

# 출력 평균 또는 voting
```

### 4.2 하이퍼파라미터 튜닝

- Learning rate: 1e-4, 3e-4, 5e-4
- Epochs: 5, 10, 15
- Batch size: 4, 8, 16
- Beam size: 4, 5, 6

---

## 실행 순서

### Week 1: 데이터 준비
```
Day 1: preprocessing.py 모듈 작성
Day 2: Train 데이터 문장 분리 & 검증
Day 3: Test 데이터 분석 & OOV 처리 추가
Day 4: 데이터셋 최종 검증
```

### Week 2: 기본 모델 학습
```
Day 5: V2 학습 코드 작성 (Kaggle 용)
Day 6: ByT5-Small 학습 (Tier3 데이터)
Day 7: 추론 & 제출, 베이스라인 점수 확인
```

### Week 3: 데이터 증강 & 개선
```
Day 8: Back-translation으로 pseudo-label 생성
Day 9: 증강 데이터로 Fine-tuning
Day 10: 앙상블 & 최종 제출
```

---

## 파일 구조

```
src/
├── common/
│   ├── preprocessing.py      # 통합 전처리 함수
│   └── metrics.py            # 평가 지표
├── data/
│   ├── prepare_sentences.py  # 문장 분리 스크립트
│   └── augment_backtrans.py  # 증강 스크립트
├── notebooks/
│   ├── akka_v2_train.py      # 학습 노트북
│   └── akka_v2_infer.py      # 추론 노트북
└── outputs/
    ├── v2_tier3/             # 기본 모델
    └── v2_augmented/         # 증강 모델
```

---

## 핵심 체크포인트

1. **전처리 일관성**: Train과 Test에 정확히 같은 `normalize_transliteration()` 적용
2. **데이터 레벨**: 문장 단위로 통일
3. **모델 일관성**: 학습과 추론에서 같은 아키텍처 (ByT5-Small)
4. **토크나이저**: `AutoTokenizer.from_pretrained(MODEL_DIR)` 사용
5. **검증**: 업로드 전 config.json의 d_model, vocab_size 확인

---

*Created: 2026-01-30*
*Goal: Train/Test 분포 일치 + 데이터 증강으로 V1 대비 큰 폭 개선*
