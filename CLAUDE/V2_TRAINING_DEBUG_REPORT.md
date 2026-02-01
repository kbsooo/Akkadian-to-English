# V2 Training Debug Report

**Date**: 2025-01-31
**Issue**: Training loss explosion (7,129,515) and NaN validation loss
**Status**: ✅ Root cause identified

---

## 1. 문제 현상

### 1.1 증상
```
Epoch 1: Training Loss: 7129515.3131, Validation Loss: nan, BLEU: 0.00, chrF++: 0.00
Epoch 2: Training Loss: 0.0000, Validation Loss: nan, BLEU: 0.00, chrF++: 0.00
```

- 1 에포크: Loss가 **7백만 이상**으로 폭발
- 2 에포크: Loss가 **0.0**으로 붕괴 (gradient vanishing)
- 모든 에포크: Validation Loss = **NaN**
- 평가 메트릭: BLEU = 0, chrF++ = 0

### 1.2 이전 시도 (실패)
첫 번째 학습 시도에서도 유사한 패턴:
```
Epoch 1: Training Loss: 341142, Validation Loss: nan, BLEU: 0.00, chrF++: 0.00
Epoch 2: Training Loss: 0.0000, Validation Loss: nan, BLEU: 0.00, chrF++: 0.00
```

---

## 2. 디버깅 과정

### 2.1 데이터 검증 ✅
```python
# 검증 결과
- 총 학습 데이터: 2,565행
- NaN 값: 0개
- 빈 문자열: 0개
- Train/Val 중복 ID: 0개
```

**결론**: 데이터 품질 문제 아님

### 2.2 비정상 문자 검사

Target(영어 번역)에서 비ASCII 문자 발견:
```
전체 행: 2565
비ASCII 타겟 행: 2377 (92.7%)
아카드어 특수문자 포함 행: 2372 (92.5%)
```

발견된 문자들:
- `š`, `Š` (shin)
- `ā`, `ē`, `ī`, `ū` (장모음)
- `ṣ`, `Ṣ`, `ṭ` (emphatic consonants)

**분석 결과**: 이 문자들은 **고유명사의 학술적 전사**에 사용됨
```
예시: "Seal of Mannum-balum-Aššur son of Ṣilli-Adad"
      "From Šu-Tammuzī, Elaya, Ennam-Aššur..."
```

**결론**: 정상적인 학술 번역 데이터, 문제 아님

### 2.3 코드 비교 분석 🔍

두 파일 간 설정 차이 발견:

| 설정 | `akka_v2_train.py` | `akka_v2_train.ipynb` |
|------|--------------------|-----------------------|
| **fp16** | `False` ✅ | `True` ❌ |
| max_source_length | 256 | 512 |
| learning_rate | 1e-4 | 3e-4 |
| warmup_ratio | 0.1 | 0.05 |

---

## 3. 근본 원인

### 🚨 FP16 (Half Precision) + ByT5 = Numerical Instability

**ByT5는 FP16과 호환되지 않습니다.**

#### 이유:
1. **Byte-level processing**: ByT5는 문자가 아닌 바이트 단위로 처리하여 시퀀스가 매우 길어짐
2. **긴 시퀀스 + FP16**: Attention score 계산 시 수치 overflow 발생
3. **Gradient explosion**: Loss가 수백만까지 폭발
4. **NaN propagation**: 한번 NaN이 발생하면 전체 gradient에 전파

#### 기술적 설명:
```
FP16 범위: ±65,504 (최대값)
FP32 범위: ±3.4 × 10^38

ByT5 attention 계산:
- 시퀀스 길이 512 bytes
- Attention scores: softmax(QK^T / √d)
- 긴 시퀀스에서 QK^T 값이 FP16 범위 초과 가능
→ Overflow → Inf → NaN
```

#### 관련 이슈:
- [Hugging Face Issue #12039](https://github.com/huggingface/transformers/issues/12039)
- T5/ByT5 계열 모델의 알려진 FP16 불안정성

---

## 4. 해결 방법

### 4.1 필수 수정 (Config 클래스)

```python
@dataclass
class Config:
    # ... 기존 설정 ...

    # ⚠️ 핵심 수정: FP16 비활성화
    fp16: bool = False      # True → False
    bf16: bool = False      # A100에서는 True 가능하나 안전하게 False

    # 추가 안정화 설정
    max_source_length: int = 256    # 512 → 256 (overflow 방지)
    max_target_length: int = 256    # 512 → 256
    learning_rate: float = 1e-4     # 3e-4 → 1e-4 (안정적)
    warmup_ratio: float = 0.1       # 0.05 → 0.1 (점진적 학습)
```

### 4.2 대안: BF16 사용 (A100/H100만)

A100 GPU는 BF16 (Brain Float 16)을 지원합니다:
```python
fp16: bool = False
bf16: bool = True  # A100에서만!
```

BF16은 FP16보다 더 넓은 지수 범위를 가져 overflow에 강합니다.

**주의**: T4, V100 등은 BF16을 지원하지 않음

### 4.3 메모리 고려사항

FP32는 FP16의 2배 메모리를 사용합니다:

| 설정 | VRAM 사용량 (추정) |
|------|-------------------|
| FP16, batch=2, seq=512 | ~12GB |
| FP32, batch=2, seq=512 | ~24GB |
| FP32, batch=2, seq=256 | ~12GB |

A100 (40GB/80GB)에서는 충분하지만, 시퀀스 길이를 256으로 줄이면 더 안전합니다.

---

## 5. 수정된 설정 (권장)

```python
@dataclass
class Config:
    """Training configuration - FP16 disabled for ByT5 stability."""

    # Model
    model_name: str = "google/byt5-base"

    # Paths
    data_dir: Path = None
    output_dir: Path = Path("/content/drive/MyDrive/akkadian/v2")

    # Sequence lengths (reduced for stability)
    max_source_length: int = 256
    max_target_length: int = 256

    # Training hyperparameters
    seed: int = 42
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch = 16
    epochs: int = 10
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Hardware - CRITICAL: FP16 must be False for ByT5!
    fp16: bool = False
    bf16: bool = False  # Set True only on A100/H100
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
```

---

## 6. 예상 결과

수정 후 예상되는 정상적인 학습 로그:
```
Epoch 1: Training Loss: 2.5-4.0, Validation Loss: 2.0-3.5, BLEU: 0-5, chrF++: 5-15
Epoch 2: Training Loss: 1.5-2.5, Validation Loss: 1.5-2.5, BLEU: 5-15, chrF++: 15-25
...
Epoch 10: Training Loss: 0.3-0.8, Validation Loss: 0.5-1.0, BLEU: 20-40, chrF++: 40-60
```

---

## 7. 체크리스트

수정 전 확인사항:

- [ ] `fp16: bool = False` 설정 확인
- [ ] `bf16: bool = False` 또는 A100이면 `True`
- [ ] `max_source_length` ≤ 256 (권장)
- [ ] `learning_rate` = 1e-4 (권장)
- [ ] `warmup_ratio` ≥ 0.1 (권장)
- [ ] Colab에서 **ipynb 파일**의 Config가 수정되었는지 확인
- [ ] GPU 메모리 여유 확인 (FP32는 2배 메모리 필요)

---

## 8. 참고 자료

- [ByT5 Paper](https://arxiv.org/abs/2105.13626)
- [Transformers FP16 Training Guide](https://huggingface.co/docs/transformers/perf_train_gpu_one#fp16-training)
- [Mixed Precision Training Best Practices](https://pytorch.org/docs/stable/notes/amp_examples.html)
- [BF16 vs FP16](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)

---

**Report generated by Claude**
