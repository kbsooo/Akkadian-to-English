# Deep Past Challenge - 1등을 위한 최적 모델 전략

**작성일:** 2026-01-27
**목표:** Kaggle 대회 1등

---

## 1. Executive Summary

### 🏆 최종 권장 전략

| 순위 | 전략 | 예상 성능 | 복잡도 |
|------|------|----------|--------|
| **1위** | **ByT5-Large + MADLAD-400 앙상블** | ⭐⭐⭐⭐⭐ | 높음 |
| 2위 | ByT5-Base + NLLB-200 앙상블 | ⭐⭐⭐⭐ | 중간 |
| 3위 | Fine-tuned ByT5-Large 단일 모델 | ⭐⭐⭐⭐ | 낮음 |

### 핵심 근거
1. **ByT5가 대세인 이유가 있음** - 문자 수준 처리로 희귀 문자/OOV 문제 해결
2. **Encoder-Decoder가 Translation에 더 적합** - Decoder-only보다 양방향 이해 우수
3. **앙상블이 필수** - 단일 모델로는 상위권 진입 어려움

---

## 2. 후보 모델 종합 비교

### 2.1 전용 번역 모델 (Encoder-Decoder)

| 모델 | 파라미터 | 저자원 성능 | Kaggle 적합성 | 라이선스 |
|------|----------|-------------|---------------|----------|
| **ByT5-Large** | 1.2B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Apache 2.0 |
| **ByT5-Base** | 580M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Apache 2.0 |
| **MADLAD-400** | 10.7B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | CC BY 4.0 |
| **NLLB-200** | 54.5B/3.3B/1.3B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | CC-BY-NC 4.0 |
| **mBART-50** | 611M | ⭐⭐⭐ | ⭐⭐⭐⭐ | MIT |
| **mT5-Large** | 1.2B | ⭐⭐⭐ | ⭐⭐⭐⭐ | Apache 2.0 |

### 2.2 범용 LLM (Decoder-Only)

| 모델 | 파라미터 | 번역 성능 | Kaggle 적합성 | 비고 |
|------|----------|----------|---------------|------|
| **Gemma 2** | 9B/27B | ⭐⭐⭐⭐ | ⭐⭐⭐ | 9시간 제한 주의 |
| **Llama 3.1** | 8B/70B | ⭐⭐⭐⭐ | ⭐⭐⭐ | 영어 중심 |
| **Qwen 2.5** | 7B/14B | ⭐⭐⭐⭐ | ⭐⭐⭐ | 다국어 강점 |
| **Aya 23** | 8B/35B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 저자원 특화 |

### 2.3 성능 vs 속도 트레이드오프

```
성능 (BLEU)
    ^
    |     * MADLAD-400 10.7B
    |   * NLLB-54.5B
    |  * ByT5-Large ← 최적 균형점
    | * NLLB-3.3B
    |* ByT5-Base
    |* mT5-Base
    +-------------------------> 추론 속도
       빠름              느림
```

---

## 3. ByT5가 최적인 이유

### 3.1 Akkadian 특성과의 완벽한 매칭

| Akkadian 특성 | ByT5 장점 |
|---------------|----------|
| 특수 문자 (ṣ, ṭ, š, ḫ) | 바이트 수준 처리로 모든 유니코드 지원 |
| 희귀 어휘 | 토크나이저 없이 OOV 문제 없음 |
| 형태론적 복잡성 | 문자 수준에서 형태소 패턴 학습 |
| 적은 학습 데이터 | 저자원 환경에서 mT5 대비 우수 |
| 결정사 {d}, {ki} | 특수 마커 그대로 처리 가능 |

### 3.2 연구 결과 근거

> "ByT5 provides strong gains in translation under low-data conditions, substantially outperforming mT5 at 400–10,000 examples."
> — [ACL 2024](https://aclanthology.org/2024.tacl-1.22/)

> "Character-level models excel at learning morphology and rare word translation."
> — [MIT TACL](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00651)

### 3.3 ByT5 vs 다른 모델

| 비교 | ByT5 장점 | 단점 |
|------|----------|------|
| vs mT5 | 저자원에서 +3-5 BLEU | 추론 2-7x 느림 |
| vs NLLB | 특수 문자 처리 우수 | 사전학습 언어 수 적음 |
| vs Decoder-only | 양방향 이해 | 생성 속도 |

---

## 4. 왜 From Scratch가 아닌가?

### 4.1 From Scratch의 한계

| 요소 | 현실 |
|------|------|
| 학습 데이터 | 1,561개 문서 (~1M 토큰) |
| 필요 데이터 | 최소 10M+ 토큰 |
| Transformer 학습 | 수일~수주 소요 |
| 경쟁 모델 | 수십억 토큰으로 사전학습됨 |

### 4.2 결론

> ❌ **From Scratch는 비추천**
>
> 저자원 언어에서는 **사전학습된 다국어 모델의 Transfer Learning**이 압도적으로 유리함

---

## 5. 최적 모델 조합 전략

### 5.1 🥇 1등 전략: ByT5-Large + MADLAD-400 앙상블

```
┌─────────────────────────────────────────────────┐
│              Training Pipeline                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────┐      ┌─────────────────┐      │
│  │  Train Data │──────│ Sentence Align  │      │
│  └─────────────┘      └────────┬────────┘      │
│                                │               │
│           ┌────────────────────┴───────┐       │
│           ▼                            ▼       │
│  ┌─────────────────┐      ┌─────────────────┐  │
│  │   ByT5-Large    │      │  MADLAD-400-3B  │  │
│  │   Fine-tuning   │      │   Fine-tuning   │  │
│  │   (LoRA r=16)   │      │   (LoRA r=8)    │  │
│  └────────┬────────┘      └────────┬────────┘  │
│           │                        │           │
│           └────────────┬───────────┘           │
│                        ▼                       │
│              ┌─────────────────┐               │
│              │  Weighted Avg   │               │
│              │  Ensemble       │               │
│              │  (0.6 : 0.4)    │               │
│              └────────┬────────┘               │
│                       ▼                        │
│              ┌─────────────────┐               │
│              │  submission.csv │               │
│              └─────────────────┘               │
└─────────────────────────────────────────────────┘
```

**구성:**
- **ByT5-Large (1.2B):** 문자 수준 처리, 희귀 어휘 강점
- **MADLAD-400-3B:** 450+ 언어 커버리지, 저자원 특화
- **앙상블 비율:** ByT5 60% + MADLAD 40%

**예상 리소스:**
- GPU 메모리: ~24GB (T4 x2 또는 P100)
- 추론 시간: ~4-6시간 (4,000 문장)
- 9시간 제한 내 충분히 가능

### 5.2 🥈 2등 전략: ByT5-Base + NLLB-1.3B 앙상블

**구성:**
- **ByT5-Base (580M):** 빠른 추론, 충분한 성능
- **NLLB-1.3B:** 저자원 언어 최적화
- **앙상블 비율:** ByT5 55% + NLLB 45%

**장점:**
- 단일 T4 GPU로 실행 가능
- 더 많은 앙상블 후보 추가 가능

### 5.3 🥉 3등 전략: ByT5-Large 단일 모델

**구성:**
- ByT5-Large + LoRA (r=16, alpha=32)
- 데이터 증강 (Back-translation)
- Beam search (beam=5)

**장점:**
- 구현 단순
- 디버깅 용이
- 안정적 성능

---

## 6. 세부 구현 권장사항

### 6.1 데이터 전처리

```python
# 권장 특수 토큰 처리
SPECIAL_TOKENS = {
    '<gap>': '<gap>',           # 그대로 유지
    '<big_gap>': '<big_gap>',   # 그대로 유지
    'x': '<unk_char>',          # 특수 토큰
}

# Sumerogram 처리 (선택적)
# KÙ.BABBAR → silver 또는 그대로 유지
```

### 6.2 Fine-tuning 설정

```python
# ByT5-Large LoRA 설정
lora_config = {
    'r': 16,
    'lora_alpha': 32,
    'target_modules': ['q', 'v', 'k', 'o'],
    'lora_dropout': 0.05,
}

# 학습 하이퍼파라미터
training_args = {
    'learning_rate': 3e-4,
    'batch_size': 8,
    'gradient_accumulation': 4,
    'epochs': 10,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
}
```

### 6.3 추론 최적화

```python
# 추론 설정
generation_config = {
    'max_new_tokens': 512,
    'num_beams': 5,
    'length_penalty': 1.0,
    'early_stopping': True,
    'no_repeat_ngram_size': 3,
}
```

### 6.4 앙상블 전략

```python
# 단순 가중 평균 (문자열)
def ensemble_predictions(pred1, pred2, weight1=0.6):
    # BLEU 점수 기반 선택 또는
    # 길이/신뢰도 기반 선택
    pass

# 또는 MBR (Minimum Bayes Risk) 디코딩
def mbr_ensemble(candidates):
    # 후보들 간 BLEU 유사도 계산
    # 가장 높은 평균 유사도 선택
    pass
```

---

## 7. Kaggle 제약 조건 대응

### 7.1 9시간 런타임 제한

| 단계 | 예상 시간 | 최적화 방법 |
|------|----------|-------------|
| 모델 로딩 | 5-10분 | 양자화 모델 사용 |
| 추론 (4K 문장) | 4-6시간 | 배치 처리, Mixed Precision |
| 앙상블 | 10-20분 | 단순 로직 |
| **총합** | **~6시간** | ✅ 안전 마진 확보 |

### 7.2 GPU 메모리 최적화

```python
# 8-bit 양자화
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/byt5-large",
    load_in_8bit=True,
    device_map="auto"
)

# Gradient checkpointing (학습 시)
model.gradient_checkpointing_enable()
```

### 7.3 인터넷 비활성화 대응

- 모델 가중치를 Kaggle Dataset으로 미리 업로드
- 모든 의존성 오프라인 설치

---

## 8. 추가 데이터 활용 전략

### 8.1 publications.csv (580MB)

```python
# OCR 텍스트에서 번역 쌍 추출
# 1. 학술 논문에서 transliteration-translation 쌍 찾기
# 2. 언어 감지 후 영어 번역만 필터링
# 3. 품질 필터링 (길이 비율, 특수문자 등)
```

**예상 추가 데이터:** 500-2,000 문장 쌍

### 8.2 Back-translation

```python
# 영어 → Akkadian 모델 학습
# 단일 언어 Akkadian 텍스트로 가상 병렬 데이터 생성
# published_texts.csv (8,000개) 활용
```

**예상 효과:** +2-4 BLEU

### 8.3 Lexicon 활용

```python
# OA_Lexicon_eBL.csv (39,332 어휘)
# 1. 어휘 임베딩 초기화
# 2. 복사 메커니즘 강화
# 3. 고유명사 정규화
```

---

## 9. 예상 성능 및 리스크

### 9.1 예상 점수 범위

| 전략 | 예상 Score | 리더보드 순위 (추정) |
|------|-----------|---------------------|
| 1등 전략 (앙상블) | 0.45-0.55 | Top 10 |
| 2등 전략 (경량 앙상블) | 0.40-0.48 | Top 30 |
| 3등 전략 (단일 모델) | 0.35-0.42 | Top 50 |

### 9.2 리스크 요소

| 리스크 | 영향 | 대응 |
|--------|------|------|
| 9시간 초과 | 제출 실패 | 배치 크기 조정, 양자화 |
| OOM | 제출 실패 | 8-bit 양자화, 작은 모델 |
| 과적합 | 낮은 점수 | Early stopping, Dropout |
| Test 분포 차이 | 낮은 점수 | 다양한 증강 |

---

## 10. 실행 로드맵

### Phase 1: 베이스라인 (1주)
```
□ ByT5-Base fine-tuning
□ 첫 제출 및 점수 확인
□ 문장 정렬 전처리 완성
```

### Phase 2: 성능 개선 (2주)
```
□ ByT5-Large로 업그레이드
□ MADLAD-400 실험
□ 데이터 증강 적용
□ 앙상블 구축
```

### Phase 3: 최적화 (2주)
```
□ publications.csv 추가 데이터 추출
□ 하이퍼파라미터 튜닝
□ 추론 속도 최적화
□ 최종 앙상블 비율 조정
```

### Phase 4: 마무리 (1주)
```
□ 런타임 테스트 (9시간 이내 확인)
□ 코드 정리 및 문서화
□ 최종 제출
```

---

## 11. 결론

### 🎯 최종 권장: ByT5-Large + MADLAD-400 앙상블

**이유:**
1. ByT5의 문자 수준 처리 → Akkadian 특수 문자에 최적
2. MADLAD의 저자원 언어 강점 → 보완적 역할
3. Encoder-Decoder 구조 → 번역 태스크에 근본적으로 유리
4. Kaggle 제약 내 실행 가능 → 9시간, 단일/듀얼 GPU

### ❌ 추천하지 않는 것
- **From Scratch:** 데이터 부족으로 불가능
- **Decoder-only LLM (Llama, Gemma):** 번역에 비효율적, 느림
- **거대 모델 (NLLB-54B):** Kaggle 리소스 초과

### ✅ 성공 핵심 요소
1. **문장 정렬 전처리** - Train/Test 불일치 해결
2. **데이터 증강** - 1,561개는 너무 적음
3. **앙상블** - 단일 모델로는 한계
4. **특수 토큰 처리** - gap, determinatives 일관성

---

## 참고 자료

### 모델 관련
- [ByT5 Paper (TACL 2022)](https://aclanthology.org/2022.tacl-1.17/)
- [MADLAD-400 Paper (NeurIPS 2023)](https://arxiv.org/abs/2309.04662)
- [NLLB Paper (Nature 2024)](https://www.nature.com/articles/s41586-024-07335-x)

### 저자원 번역
- [Low-Resource NMT Survey (ACM 2023)](https://dl.acm.org/doi/10.1145/3567592)
- [Two-Step Fine-Tuning (NAACL 2024)](https://aclanthology.org/2024.findings-naacl.203/)

### 실용 가이드
- [ByT5 vs mT5 Comparison (TACL 2024)](https://aclanthology.org/2024.tacl-1.22/)
- [Best Open Source Translation Models 2026](https://www.siliconflow.com/articles/en/best-open-source-models-for-translation)
- [Aya Model Paper](https://arxiv.org/abs/2402.07827)

---

*이 리포트는 웹 검색 및 연구 논문 분석을 바탕으로 작성되었습니다.*
