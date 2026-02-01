# V2 분석 및 개선 방안

**Date**: 2026-02-01
**Current Score**: 11.8 (Public)
**Target Score**: ~35+ (3배 향상)

---

## 1. 현재 데이터 전처리 분석

### 1.1 데이터 출처

| 데이터셋 | 행 수 | 설명 |
|---------|-------|------|
| `train.csv` (원본) | 1,561 | Kaggle 제공 학습 데이터 |
| `published_texts.csv` | 7,953 | 점토판 전문 transliteration |
| `Sentences_Oare` | 9,782 | 문장 단위 번역 (9,772개에 번역 있음) |
| `publications.csv` | 216,602 | PDF OCR 텍스트 (31,286 페이지에 아카드어 포함) |

### 1.2 V2 데이터 구성

```
v2_train.csv:           1,405 rows  ← train.csv에서 정제
v2_train_augmented.csv: 2,565 rows  ← v2_train + 1,160 새 데이터
```

**증강 방법** (`augment_data.py`):
1. `Sentences_Oare`의 문장별 번역을 문서(text_uuid) 단위로 합침
2. `published_texts.csv`의 transliteration과 조인
3. `train.csv`에 없는 문서만 추출 → 1,160개 새 학습 쌍 생성

### 1.3 전처리 파이프라인 (`normalize.py`)

**Transliteration (Source) 정규화:**
```python
# 적용 순서
1. Unicode NFC 정규화
2. 발음 기호 제거: š→s, ṣ→s, ḫ→h, ṭ→t, ā→a, etc.
3. OCR 아티팩트 정리: „→", '→', etc.
4. 아래첨자 숫자화: ₄→4, ₂→2, etc.
5. 손상/갭 마커: [...] → <gap>, ... → <gap>
6. 미확인 기호: x → <unk>
7. 편집 기호 제거: !?/
8. 공백 정규화
```

**Translation (Target) 정규화:**
```python
1. Unicode NFC 정규화
2. 따옴표 정규화
3. 공백 정규화
```

### 1.4 ⚠️ Publications 데이터 미활용

**현재 상황:**
- `publications.csv` (216,602 페이지, 31,286개 아카드어 포함)는 **아직 활용되지 않음**
- 이 데이터는 PDF OCR 결과로, 구조화된 transliteration-translation 쌍이 아님
- 학술 논문의 아카드어 인용/분석 텍스트가 포함되어 있으나 직접 학습에 사용하기 어려움

**활용 가능성:**
- Pseudo-labeling: 모델로 번역 생성 후 필터링하여 학습 데이터로 사용
- Language modeling: ByT5 사전학습 또는 도메인 적응
- 현실적으로 복잡한 파싱이 필요하여 **즉시 활용은 어려움**

---

## 2. ByT5-Large 사용 시 예상 효과

### 2.1 모델 크기 비교

| 모델 | 파라미터 | Encoder Layers | Hidden Size | VRAM 필요 (FP32) |
|------|----------|----------------|-------------|------------------|
| ByT5-Small | 300M | 12 | 1472 | ~4GB |
| ByT5-Base | 580M | 12 | 1536 | ~8GB |
| **ByT5-Large** | **1.2B** | 24 | 1536 | **~16-20GB** |
| ByT5-XL | 3.7B | 24 | 2048 | ~40GB+ |

### 2.2 예상 성능 향상

일반적으로 모델 크기 2배 증가 시 NMT 품질 10-20% 향상:

```
예상 시나리오:
- Base (11.8) → Large: 15-18 점 (약 30-50% 향상)
- 단독으로 3배(35+)는 어려움
```

### 2.3 ByT5-Large 학습 요구사항

| 설정 | A100 40GB | A100 80GB | T4 16GB |
|------|-----------|-----------|---------|
| Full Fine-tuning (FP32) | ❌ | ✅ | ❌ |
| Full Fine-tuning (BF16) | ⚠️ (작은 배치) | ✅ | ❌ |
| **LoRA (BF16)** | ✅ | ✅ | ⚠️ |
| QLoRA (4-bit) | ✅ | ✅ | ✅ |

---

## 3. Full Fine-tuning vs LoRA

### 3.1 Full Fine-tuning (현재 방식)

```python
# 모든 파라미터 업데이트
for param in model.parameters():
    param.requires_grad = True

# 학습 시 저장되는 것:
# - 전체 모델 가중치 (Base: 580M params ≈ 2.3GB)
```

**장점:**
- 최대 성능 잠재력
- 구현이 단순함

**단점:**
- 메모리 사용량 큼 (모델 + Optimizer states + Gradients)
- Base 580M → 학습 시 ~16GB VRAM 필요
- Large 1.2B → 학습 시 ~40GB+ VRAM 필요 (FP32)

### 3.2 LoRA (Low-Rank Adaptation)

```python
# 기존 가중치 동결
for param in model.parameters():
    param.requires_grad = False

# 저랭크 어댑터만 학습
# W_new = W_frozen + A @ B  (A: d×r, B: r×d, r << d)
lora_config = LoraConfig(
    r=16,              # 랭크 (작을수록 효율적, 보통 8-64)
    lora_alpha=32,     # 스케일링 팩터
    target_modules=["q", "v"],  # 적용할 레이어
    lora_dropout=0.1,
)

# 학습 파라미터: 원본의 0.1-1% 수준
# Large 1.2B → LoRA 파라미터 ~1-10M (0.1-1%)
```

**장점:**
- 메모리 효율적: Large 모델도 A100 40GB에서 학습 가능
- 빠른 학습 속도
- 여러 태스크별 어댑터 저장 가능 (작은 용량)
- Catastrophic forgetting 방지

**단점:**
- Full fine-tuning 대비 약간 낮은 성능 (보통 90-98% 수준)
- 하이퍼파라미터 튜닝 필요 (r, alpha, target_modules)

### 3.3 QLoRA (Quantized LoRA)

```python
# 4-bit 양자화 + LoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

# Large 1.2B 모델:
# - 원본: 4.8GB (FP32)
# - 4-bit: ~1.2GB
# - + LoRA overhead: ~2-3GB total
```

**T4 16GB에서도 ByT5-Large 학습 가능!**

### 3.4 추천 전략

| 상황 | 추천 방법 |
|------|----------|
| A100 80GB | Full Fine-tuning (BF16) |
| A100 40GB | LoRA (BF16) 또는 QLoRA |
| T4/V100 16GB | **QLoRA (4-bit) - ByT5-Large 사용 가능** |
| Colab Free (T4) | QLoRA (4-bit) |

---

## 4. 점수 3배 향상을 위한 종합 전략

현재 11.8 → 목표 35+ 달성을 위한 다각적 접근:

### 4.1 모델 측면

| 전략 | 예상 향상 | 난이도 |
|------|----------|--------|
| ByT5-Base → Large (LoRA) | +3-5점 | ⭐⭐ |
| Ensemble (여러 체크포인트) | +2-3점 | ⭐⭐ |
| Beam Search 튜닝 (beam=5-10) | +1-2점 | ⭐ |

### 4.2 데이터 측면

| 전략 | 예상 향상 | 난이도 |
|------|----------|--------|
| Back-translation (역번역 증강) | +3-5점 | ⭐⭐ |
| Denoising (노이즈 추가 학습) | +2-3점 | ⭐⭐ |
| Curriculum Learning (쉬운→어려운) | +1-2점 | ⭐⭐⭐ |
| 외부 사전 활용 (OA_Lexicon_eBL) | +2-3점 | ⭐⭐⭐ |

### 4.3 추론 측면

| 전략 | 예상 향상 | 난이도 |
|------|----------|--------|
| Length penalty 조정 | +0.5-1점 | ⭐ |
| Repetition penalty | +0.5-1점 | ⭐ |
| Post-processing (고유명사 보정) | +1-2점 | ⭐⭐ |

### 4.4 권장 우선순위

```
1단계 (즉시 적용):
├── ByT5-Large + QLoRA 전환
├── Beam search (num_beams=5)
└── Length penalty 튜닝

2단계 (데이터 증강):
├── Back-translation 구현
├── Lexicon 기반 data augmentation
└── 더 긴 epoch 학습 (10 → 20)

3단계 (앙상블):
├── 다양한 체크포인트 앙상블
└── 다양한 seed로 학습한 모델 앙상블
```

---

## 5. ByT5-Large + LoRA 구현 예시

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
import torch

# 4-bit 양자화 설정 (T4에서도 사용 가능)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# 모델 로드
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/byt5-large",
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA 설정
lora_config = LoraConfig(
    r=16,                    # 랭크 (8-64 사이 실험)
    lora_alpha=32,           # 스케일링
    target_modules=["q", "k", "v", "o", "wi", "wo"],  # T5 attention + FFN
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

# LoRA 적용
model = get_peft_model(model, lora_config)

# 학습 가능한 파라미터 확인
model.print_trainable_parameters()
# 출력 예: trainable params: 9,437,184 || all params: 1,229,837,312 || trainable%: 0.77%
```

---

## 6. 결론

### 현재 상태 요약
- **Publications 데이터**: 직접 활용 안됨 (OCR 텍스트로 파싱 필요)
- **전처리**: 발음기호/특수문자 ASCII 정규화, 갭/미확인 마커 처리
- **증강**: Sentences_Oare + published_texts 조인으로 +1,160개 추가

### 3배 향상을 위한 핵심 액션
1. **ByT5-Large + QLoRA**: 가장 즉각적인 성능 향상 (T4에서도 가능)
2. **Back-translation**: 데이터 양 2배 증가 효과
3. **추론 최적화**: Beam search, length penalty
4. **앙상블**: 최종 제출 시 여러 모델 결합

### 예상 점수 로드맵
```
현재:        11.8
+ Large/LoRA: 15-18
+ 데이터증강:  20-25
+ 앙상블:     28-35
```

---

**다음 단계**: ByT5-Large + QLoRA 학습 코드 구현
