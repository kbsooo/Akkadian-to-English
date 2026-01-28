# ByT5-Large + MADLAD-400 앙상블 전략 상세 가이드

**작성일:** 2026-01-27
**목표:** Deep Past Challenge 1등

---

## 1. 전략 개요

### 1.1 왜 이 조합인가?

```
┌─────────────────────────────────────────────────────────────┐
│                    앙상블 시너지 효과                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ByT5-Large                    MADLAD-400-3B               │
│   ┌─────────────┐              ┌─────────────┐              │
│   │ 문자 수준   │              │ 450+ 언어   │              │
│   │ 처리        │              │ 커버리지    │              │
│   │             │              │             │              │
│   │ • OOV 없음  │              │ • 저자원    │              │
│   │ • 특수문자  │              │   특화      │              │
│   │ • 형태소    │              │ • 다국어    │              │
│   │   패턴      │              │   전이학습  │              │
│   └──────┬──────┘              └──────┬──────┘              │
│          │                            │                     │
│          └────────────┬───────────────┘                     │
│                       ▼                                     │
│              ┌─────────────────┐                            │
│              │   앙상블 결합   │                            │
│              │                 │                            │
│              │  서로 다른 오류 │                            │
│              │  패턴 → 상호   │                            │
│              │  보완           │                            │
│              └─────────────────┘                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

| 모델 | 강점 | 약점 | 보완 관계 |
|------|------|------|----------|
| **ByT5-Large** | 특수문자, 희귀어휘, 형태론 | 사전학습 언어 수 적음 | MADLAD가 보완 |
| **MADLAD-400-3B** | 다국어 전이, 저자원 특화 | 토큰 기반 OOV 가능 | ByT5가 보완 |

### 1.2 모델 스펙

| 항목 | ByT5-Large | MADLAD-400-3B |
|------|------------|---------------|
| 파라미터 | 1.2B | 3B |
| 아키텍처 | Encoder-Decoder | Encoder-Decoder (T5 기반) |
| 토큰화 | 바이트 수준 (UTF-8) | SentencePiece |
| 사전학습 데이터 | mC4 (101개 언어) | CommonCrawl (450+ 언어) |
| HuggingFace | `google/byt5-large` | `google/madlad400-3b-mt` |
| 라이선스 | Apache 2.0 | Apache 2.0 |

---

## 2. 데이터 준비

### 2.1 문장 정렬 (핵심!)

Train은 문서 수준, Test는 문장 수준이므로 **문장 정렬이 필수**입니다.

```python
import pandas as pd

# 데이터 로드
train = pd.read_csv('data/train.csv')
sentences_helper = pd.read_csv('data/Sentences_Oare_FirstWord_LinNum.csv')

def align_sentences(train_df, helper_df):
    """
    문서 수준 데이터를 문장 수준으로 정렬
    """
    aligned_data = []

    for idx, row in train_df.iterrows():
        oare_id = row['oare_id']
        transliteration = row['transliteration']
        translation = row['translation']

        # helper에서 해당 문서의 문장 정보 찾기
        doc_sentences = helper_df[helper_df['text_uuid'] == oare_id]

        if len(doc_sentences) > 0:
            # 라인 번호 기반으로 분리
            for _, sent in doc_sentences.iterrows():
                line_num = sent['line_number']
                first_word = sent['first_word_transcription']

                # transliteration에서 해당 부분 추출
                # (실제로는 더 정교한 로직 필요)
                sent_trans = extract_sentence(transliteration, line_num)
                sent_transl = extract_translation(translation, line_num)

                aligned_data.append({
                    'akkadian': sent_trans,
                    'english': sent_transl,
                    'oare_id': oare_id,
                    'line_num': line_num
                })
        else:
            # helper에 없는 경우 문서 전체 사용
            aligned_data.append({
                'akkadian': transliteration,
                'english': translation,
                'oare_id': oare_id,
                'line_num': 'full'
            })

    return pd.DataFrame(aligned_data)

# 문장 정렬 실행
aligned_train = align_sentences(train, sentences_helper)
print(f"원본: {len(train)} 문서 → 정렬 후: {len(aligned_train)} 문장")
```

### 2.2 데이터 증강

```python
# 1. 특수 토큰 정규화
def normalize_special_tokens(text):
    """특수 토큰 일관성 있게 처리"""
    text = text.replace('<gap>', ' [GAP] ')
    text = text.replace('<big_gap>', ' [BIG_GAP] ')
    # x (판독불가)는 그대로 유지 (ByT5가 잘 처리)
    return text.strip()

# 2. Back-translation (선택적)
def back_translate(english_texts, model, tokenizer):
    """
    영어 → Akkadian 방향으로 번역하여 가상 데이터 생성
    published_texts.csv의 단일언어 데이터 활용
    """
    synthetic_pairs = []
    for text in english_texts:
        # 역번역 수행
        akkadian_synthetic = model.generate(text)
        synthetic_pairs.append({
            'akkadian': akkadian_synthetic,
            'english': text
        })
    return synthetic_pairs

# 3. 데이터 포맷팅
def format_for_training(row, model_type='byt5'):
    """모델별 입력 형식 변환"""
    if model_type == 'byt5':
        # ByT5: prefix 사용
        source = f"translate Akkadian to English: {row['akkadian']}"
        target = row['english']
    elif model_type == 'madlad':
        # MADLAD: 언어 태그 사용
        source = f"<2en> {row['akkadian']}"
        target = row['english']

    return {'source': source, 'target': target}
```

### 2.3 데이터셋 구성

```python
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Train/Validation 분리 (90/10)
train_data, val_data = train_test_split(
    aligned_train,
    test_size=0.1,
    random_state=42
)

# HuggingFace Dataset 생성
def create_dataset(df, model_type):
    formatted = [format_for_training(row, model_type) for _, row in df.iterrows()]
    return Dataset.from_list(formatted)

# ByT5용 데이터셋
byt5_dataset = DatasetDict({
    'train': create_dataset(train_data, 'byt5'),
    'validation': create_dataset(val_data, 'byt5')
})

# MADLAD용 데이터셋
madlad_dataset = DatasetDict({
    'train': create_dataset(train_data, 'madlad'),
    'validation': create_dataset(val_data, 'madlad')
})
```

---

## 3. ByT5-Large Fine-tuning

### 3.1 모델 로드 및 LoRA 설정

```python
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

# 모델 및 토크나이저 로드
model_name = "google/byt5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 메모리 절약
    device_map="auto"
)

# LoRA 설정
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,                          # rank: 16이 균형점
    lora_alpha=32,                 # alpha = 2 * r (일반적 권장)
    lora_dropout=0.05,             # 과적합 방지
    target_modules=[               # T5 계열 타겟 모듈
        "q", "v", "k", "o",        # Attention
        "wi_0", "wi_1", "wo"       # FFN
    ],
    bias="none"
)

# LoRA 적용
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 출력 예: trainable params: 14,417,920 || all params: 1,245,253,632 || trainable%: 1.158%
```

### 3.2 전처리 함수

```python
def preprocess_byt5(examples):
    """ByT5용 전처리"""
    # ByT5는 바이트 수준이므로 max_length 더 길게
    max_source_length = 1024  # 문자 수준이므로 더 길게
    max_target_length = 512

    # 토큰화
    model_inputs = tokenizer(
        examples['source'],
        max_length=max_source_length,
        truncation=True,
        padding='max_length'
    )

    # 타겟 토큰화
    labels = tokenizer(
        examples['target'],
        max_length=max_target_length,
        truncation=True,
        padding='max_length'
    )

    model_inputs['labels'] = labels['input_ids']

    # padding 토큰을 -100으로 (loss 계산에서 제외)
    model_inputs['labels'] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in model_inputs['labels']
    ]

    return model_inputs

# 데이터셋 전처리
tokenized_byt5 = byt5_dataset.map(
    preprocess_byt5,
    batched=True,
    remove_columns=['source', 'target']
)
```

### 3.3 학습 설정

```python
# 학습 인자
training_args = Seq2SeqTrainingArguments(
    output_dir="./byt5-akkadian",

    # 학습 설정
    num_train_epochs=10,
    per_device_train_batch_size=4,      # GPU 메모리에 따라 조정
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,       # effective batch = 16

    # 최적화
    learning_rate=3e-4,                  # LoRA 권장 학습률
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,

    # 평가 및 저장
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,

    # 생성 설정
    predict_with_generate=True,
    generation_max_length=512,

    # 메모리 최적화
    fp16=True,
    gradient_checkpointing=True,

    # 로깅
    logging_steps=50,
    report_to="none",
)

# 평가 메트릭
import evaluate
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # 디코딩
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # -100을 pad_token_id로 변환
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # BLEU
    bleu_result = bleu.compute(
        predictions=decoded_preds,
        references=[[l] for l in decoded_labels]
    )

    # chrF++
    chrf_result = chrf.compute(
        predictions=decoded_preds,
        references=[[l] for l in decoded_labels],
        word_order=2  # chrF++
    )

    # Geometric Mean (대회 평가 지표)
    import math
    geo_mean = math.sqrt(bleu_result['score'] * chrf_result['score'])

    return {
        'bleu': bleu_result['score'],
        'chrf': chrf_result['score'],
        'geo_mean': geo_mean
    }

# 데이터 콜레이터
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Trainer 생성
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_byt5['train'],
    eval_dataset=tokenized_byt5['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 학습 실행
trainer.train()

# 최종 모델 저장
trainer.save_model("./byt5-akkadian-final")
```

---

## 4. MADLAD-400-3B Fine-tuning

### 4.1 모델 로드 및 설정

```python
# MADLAD 모델 로드
madlad_model_name = "google/madlad400-3b-mt"
madlad_tokenizer = AutoTokenizer.from_pretrained(madlad_model_name)
madlad_model = AutoModelForSeq2SeqLM.from_pretrained(
    madlad_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 설정 (MADLAD용 - 더 작은 rank)
madlad_lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,                           # 더 작은 rank (모델이 더 크므로)
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"],
    bias="none"
)

madlad_model = get_peft_model(madlad_model, madlad_lora_config)
```

### 4.2 MADLAD 전처리

```python
def preprocess_madlad(examples):
    """MADLAD용 전처리 - 언어 태그 사용"""
    max_source_length = 512   # 토큰 기반이므로 ByT5보다 짧음
    max_target_length = 256

    # 소스: <2en> 태그로 영어 번역 지시
    # MADLAD는 타겟 언어 태그를 prefix로 사용
    sources = [f"<2en> {s}" for s in examples['source']]

    model_inputs = madlad_tokenizer(
        sources,
        max_length=max_source_length,
        truncation=True,
        padding='max_length'
    )

    labels = madlad_tokenizer(
        examples['target'],
        max_length=max_target_length,
        truncation=True,
        padding='max_length'
    )

    model_inputs['labels'] = labels['input_ids']
    model_inputs['labels'] = [
        [(l if l != madlad_tokenizer.pad_token_id else -100) for l in label]
        for label in model_inputs['labels']
    ]

    return model_inputs

# MADLAD 데이터셋 전처리
tokenized_madlad = madlad_dataset.map(
    preprocess_madlad,
    batched=True,
    remove_columns=['source', 'target']
)
```

### 4.3 MADLAD 학습

```python
# MADLAD 학습 설정 (유사하지만 배치 크기 조정)
madlad_training_args = Seq2SeqTrainingArguments(
    output_dir="./madlad-akkadian",
    num_train_epochs=10,
    per_device_train_batch_size=2,      # 모델이 더 크므로 작게
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,       # effective batch = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=256,
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=50,
    report_to="none",
)

# MADLAD Trainer
madlad_trainer = Seq2SeqTrainer(
    model=madlad_model,
    args=madlad_training_args,
    train_dataset=tokenized_madlad['train'],
    eval_dataset=tokenized_madlad['validation'],
    tokenizer=madlad_tokenizer,
    data_collator=DataCollatorForSeq2Seq(madlad_tokenizer, model=madlad_model),
    compute_metrics=compute_metrics
)

madlad_trainer.train()
madlad_trainer.save_model("./madlad-akkadian-final")
```

---

## 5. 앙상블 추론

### 5.1 앙상블 전략 옵션

```
┌─────────────────────────────────────────────────────────────┐
│                    앙상블 전략 비교                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 확률 평균 (Probability Averaging)                       │
│     - 토큰 생성 시 로짓 평균                                │
│     - 가장 이론적으로 타당                                  │
│     - 구현 복잡도 높음                                      │
│                                                             │
│  2. 후보 선택 (Candidate Selection)                        │
│     - 각 모델의 최종 출력 중 선택                           │
│     - QE 점수 또는 BLEU 유사도 기반                         │
│     - 구현 간단, 효과적                                     │
│                                                             │
│  3. MBR 디코딩 (Minimum Bayes Risk)                        │
│     - 여러 후보 간 상호 유사도 계산                         │
│     - 가장 "합의된" 번역 선택                               │
│     - 높은 품질, 느린 속도                                  │
│                                                             │
│  ★ 권장: 후보 선택 + MBR 혼합                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 추론 코드

```python
class AkkadianEnsemble:
    """ByT5 + MADLAD 앙상블 추론 클래스"""

    def __init__(self, byt5_path, madlad_path, device='cuda'):
        self.device = device

        # ByT5 로드
        self.byt5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-large")
        self.byt5_model = AutoModelForSeq2SeqLM.from_pretrained(
            byt5_path,
            torch_dtype=torch.float16
        ).to(device)
        self.byt5_model.eval()

        # MADLAD 로드
        self.madlad_tokenizer = AutoTokenizer.from_pretrained("google/madlad400-3b-mt")
        self.madlad_model = AutoModelForSeq2SeqLM.from_pretrained(
            madlad_path,
            torch_dtype=torch.float16
        ).to(device)
        self.madlad_model.eval()

        # 앙상블 가중치 (검증 점수 기반으로 조정)
        self.byt5_weight = 0.55
        self.madlad_weight = 0.45

    def generate_byt5(self, text, num_beams=5, num_return=3):
        """ByT5로 번역 후보 생성"""
        input_text = f"translate Akkadian to English: {text}"
        inputs = self.byt5_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.byt5_model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=num_beams,
                num_return_sequences=num_return,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0
            )

        return [
            self.byt5_tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]

    def generate_madlad(self, text, num_beams=5, num_return=3):
        """MADLAD로 번역 후보 생성"""
        input_text = f"<2en> {text}"
        inputs = self.madlad_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.madlad_model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=num_beams,
                num_return_sequences=num_return,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0
            )

        return [
            self.madlad_tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]

    def mbr_select(self, candidates):
        """MBR 디코딩으로 최적 후보 선택"""
        from sacrebleu.metrics import BLEU
        bleu = BLEU(effective_order=True)

        scores = []
        for i, cand in enumerate(candidates):
            # 다른 모든 후보와의 BLEU 유사도 계산
            total_score = 0
            for j, other in enumerate(candidates):
                if i != j:
                    score = bleu.sentence_score(cand, [other]).score
                    total_score += score
            avg_score = total_score / (len(candidates) - 1)
            scores.append(avg_score)

        # 가장 높은 평균 유사도를 가진 후보 선택
        best_idx = scores.index(max(scores))
        return candidates[best_idx]

    def weighted_select(self, byt5_candidates, madlad_candidates):
        """가중치 기반 후보 선택"""
        # 각 모델의 1순위 후보에 가중치 적용
        # 길이, 특수 토큰 처리 등 휴리스틱 추가 가능

        byt5_best = byt5_candidates[0]
        madlad_best = madlad_candidates[0]

        # 간단한 휴리스틱: 더 긴 번역 선호 (보통 더 상세)
        # 실제로는 QE 모델 사용 권장
        if len(byt5_best.split()) >= len(madlad_best.split()):
            return byt5_best
        else:
            return madlad_best

    def translate(self, text, method='mbr'):
        """
        앙상블 번역 수행

        Args:
            text: Akkadian transliteration
            method: 'mbr', 'weighted', 'byt5_only', 'madlad_only'
        """
        if method == 'byt5_only':
            return self.generate_byt5(text, num_return=1)[0]

        if method == 'madlad_only':
            return self.generate_madlad(text, num_return=1)[0]

        # 두 모델에서 후보 생성
        byt5_candidates = self.generate_byt5(text, num_return=3)
        madlad_candidates = self.generate_madlad(text, num_return=3)

        if method == 'weighted':
            return self.weighted_select(byt5_candidates, madlad_candidates)

        if method == 'mbr':
            # 모든 후보를 MBR로 선택
            all_candidates = byt5_candidates + madlad_candidates
            return self.mbr_select(all_candidates)

        raise ValueError(f"Unknown method: {method}")

    def batch_translate(self, texts, method='mbr', batch_size=16):
        """배치 번역"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                result = self.translate(text, method)
                results.append(result)

            # 진행상황 출력
            print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)}")

        return results
```

### 5.3 추론 실행

```python
# 앙상블 모델 초기화
ensemble = AkkadianEnsemble(
    byt5_path="./byt5-akkadian-final",
    madlad_path="./madlad-akkadian-final"
)

# 테스트 데이터 로드
test = pd.read_csv('data/test.csv')

# 번역 수행
translations = ensemble.batch_translate(
    test['transliteration'].tolist(),
    method='mbr',
    batch_size=8
)

# 제출 파일 생성
submission = pd.DataFrame({
    'id': test['id'],
    'translation': translations
})
submission.to_csv('submission.csv', index=False)
```

---

## 6. Kaggle 노트북 최적화

### 6.1 메모리 최적화

```python
# 8-bit 양자화로 메모리 절약
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 6.2 시간 최적화

```python
# 추론 시간 추정
import time

def estimate_inference_time(model, tokenizer, sample_texts, n_samples=100):
    """추론 시간 추정"""
    times = []

    for text in sample_texts[:n_samples]:
        start = time.time()
        # 실제 추론
        inputs = tokenizer(text, return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)
        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    total_estimated = avg_time * 4000  # 테스트 4000문장

    print(f"평균 추론 시간: {avg_time:.2f}초/문장")
    print(f"총 예상 시간: {total_estimated/3600:.2f}시간")

    return avg_time

# 시간 체크
# ByT5: ~0.8초/문장 → 4000문장 = ~0.9시간
# MADLAD: ~1.2초/문장 → 4000문장 = ~1.3시간
# 앙상블 (MBR): ~3초/문장 → 4000문장 = ~3.3시간
# 총합: ~5.5시간 (9시간 제한 내)
```

### 6.3 완전한 Kaggle 노트북 구조

```python
# ====== CELL 1: 설정 ======
import os
os.environ['TRANSFORMERS_CACHE'] = '/kaggle/working/cache'

!pip install -q transformers peft accelerate bitsandbytes sacrebleu

# ====== CELL 2: 모델 로드 ======
# 미리 업로드한 모델 가중치 사용
BYT5_PATH = "/kaggle/input/byt5-akkadian-weights"
MADLAD_PATH = "/kaggle/input/madlad-akkadian-weights"

# ====== CELL 3: 앙상블 클래스 정의 ======
# (위의 AkkadianEnsemble 클래스)

# ====== CELL 4: 데이터 로드 및 추론 ======
test = pd.read_csv('/kaggle/input/deep-past-initiative-machine-translation/test.csv')

ensemble = AkkadianEnsemble(BYT5_PATH, MADLAD_PATH)
translations = ensemble.batch_translate(
    test['transliteration'].tolist(),
    method='mbr'
)

# ====== CELL 5: 제출 파일 생성 ======
submission = pd.DataFrame({
    'id': test['id'],
    'translation': translations
})
submission.to_csv('submission.csv', index=False)

print("Done!")
print(submission.head())
```

---

## 7. 성능 예상 및 튜닝 가이드

### 7.1 예상 성능

| 구성 | BLEU | chrF++ | Geo Mean |
|------|------|--------|----------|
| ByT5-Large 단일 | 32-38 | 55-62 | 0.42-0.48 |
| MADLAD-3B 단일 | 30-35 | 52-58 | 0.40-0.45 |
| **앙상블 (MBR)** | **36-42** | **58-65** | **0.46-0.52** |

### 7.2 하이퍼파라미터 튜닝

```python
# 그리드 서치 예시
param_grid = {
    'lora_r': [8, 16, 32],
    'lora_alpha': [16, 32, 64],
    'learning_rate': [1e-4, 2e-4, 3e-4],
    'num_epochs': [5, 10, 15]
}

# 최적 설정 (경험적)
best_params = {
    'byt5': {
        'lora_r': 16,
        'lora_alpha': 32,
        'learning_rate': 3e-4,
        'num_epochs': 10
    },
    'madlad': {
        'lora_r': 8,
        'lora_alpha': 16,
        'learning_rate': 2e-4,
        'num_epochs': 10
    }
}
```

### 7.3 앙상블 가중치 최적화

```python
def optimize_ensemble_weights(val_data, byt5_model, madlad_model):
    """검증 데이터로 앙상블 가중치 최적화"""
    from scipy.optimize import minimize

    # 각 모델의 예측 생성
    byt5_preds = [byt5_model.generate(x) for x in val_data['akkadian']]
    madlad_preds = [madlad_model.generate(x) for x in val_data['akkadian']]
    references = val_data['english'].tolist()

    def objective(weights):
        # 가중 선택으로 최종 예측
        # (실제로는 문자열 결합이 아닌 선택 로직)
        final_preds = []
        for b, m in zip(byt5_preds, madlad_preds):
            # 간단히 가중치가 높은 모델 선택
            if weights[0] > weights[1]:
                final_preds.append(b)
            else:
                final_preds.append(m)

        # BLEU 계산
        bleu_score = bleu.compute(
            predictions=final_preds,
            references=[[r] for r in references]
        )['score']

        return -bleu_score  # 최소화 → 음수

    # 최적화
    result = minimize(
        objective,
        x0=[0.5, 0.5],
        bounds=[(0.3, 0.7), (0.3, 0.7)],
        constraints={'type': 'eq', 'fun': lambda x: sum(x) - 1}
    )

    return result.x  # [byt5_weight, madlad_weight]
```

---

## 8. 체크리스트

### 학습 전
- [ ] 문장 정렬 완료
- [ ] 데이터 증강 (선택)
- [ ] GPU 메모리 확인 (16GB+ 권장)
- [ ] 모델 가중치 다운로드

### 학습 중
- [ ] Loss 감소 확인
- [ ] Validation BLEU 모니터링
- [ ] 과적합 체크 (early stopping)

### 추론 전
- [ ] 모델 저장 완료
- [ ] Kaggle Dataset으로 업로드
- [ ] 추론 시간 추정 (<9시간)

### 추론 후
- [ ] submission.csv 형식 확인
- [ ] 샘플 번역 품질 확인
- [ ] 제출

---

## 9. 참고 자료

### 공식 문서
- [HuggingFace ByT5](https://huggingface.co/docs/transformers/model_doc/byt5)
- [HuggingFace MADLAD-400](https://huggingface.co/docs/transformers/model_doc/madlad-400)
- [PEFT LoRA](https://huggingface.co/docs/peft/main/en/task_guides/seq2seq-lora)
- [HuggingFace Translation Tutorial](https://huggingface.co/docs/transformers/tasks/translation)

### 연구 논문
- [ByT5 Paper (TACL 2022)](https://aclanthology.org/2022.tacl-1.17/)
- [MADLAD-400 Paper (NeurIPS 2023)](https://arxiv.org/abs/2309.04662)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [MBR Decoding](https://arxiv.org/abs/2108.04718)

### 실용 가이드
- [LoRA Hyperparameters Guide](https://www.entrypointai.com/blog/lora-fine-tuning/)
- [Ensemble Methods in NMT](https://arxiv.org/html/2502.01182v1)

---

*이 가이드는 Deep Past Challenge 1등을 목표로 작성되었습니다.*
