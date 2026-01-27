# Akkadian 언어 특성 및 기계 번역 전략 분석

**작성일:** 2026-01-27

---

## 1. Akkadian 언어의 핵심 특징

### 1.1 언어 계통 및 역사

| 항목 | 내용 |
|------|------|
| **계통** | 동부 셈어(East Semitic) 계열 |
| **관련 언어** | 히브리어, 아랍어와 친족 관계 |
| **사용 기간** | BC 2500년 ~ AD 100년 (약 2,600년) |
| **문자 체계** | 설형문자(Cuneiform) |
| **방언** | Old Akkadian, Old Assyrian, Babylonian |

### 1.2 문자 체계의 복잡성

Akkadian은 **반표의-반음절** (semanto-phonetic) 문자 체계를 사용합니다:

| 유형 | 설명 | 표기 규칙 | 예시 |
|------|------|----------|------|
| **음절문자 (Syllabograms)** | 음절을 표현 | 소문자 이탤릭 | `a-na`, `ša` |
| **표의문자 (Logograms/Sumerograms)** | 단어 전체를 표현 | 대문자 | `KÙ.BABBAR` (은) |
| **결정사 (Determinatives)** | 의미 범주 표시 | 위첨자 | `{d}` (신), `{ki}` (장소) |

> ⚠️ **핵심 도전:** 하나의 설형문자 기호가 여러 의미(다가성, polyvalence)를 가질 수 있음

### 1.3 음운 체계

- **자음:** 20개 (셈어 특유의 인후음/후두음 소실)
- **모음:** 8개 (a, i, e, u의 장단 구분)
- **특수 자음:** ĝ, š, ḫ, q, ṣ, ṭ (영어에 없는 음)

### 1.4 형태론 (Morphology)

**명사:**
| 특성 | 구분 |
|------|------|
| 성(Gender) | 남성 / 여성 (-t, -at 접미사) |
| 수(Number) | 단수 / 쌍수 / 복수 |
| 격(Case) | 주격 / 속격 / 대격 |

**동사:**
- **3자음 어근 체계:** 셈어 공통 특징
- **13개 파생형(Stems):** G, D, Š, N 등
- **시제:** 과거 / 현재-미래

### 1.5 통사론 (Syntax)

| 특성 | 내용 |
|------|------|
| **어순** | SOV (주어-목적어-동사) - 수메르어 영향 |
| **전치사** | `ina` (~에서), `ana` (~에게) - 셈어 중 유일 |

---

## 2. 기계 번역의 주요 도전 과제

### 2.1 언어적 도전

| 도전 | 상세 | 심각도 |
|------|------|--------|
| **다가성 (Polyvalence)** | 하나의 기호가 여러 읽기 가능 | 🔴 높음 |
| **Sumerogram 처리** | 수메르어 표의문자의 Akkadian 읽기 | 🔴 높음 |
| **형태론적 복잡성** | 굴절 접사로 인한 어형 변화 | 🟠 중간 |
| **시대/방언 차이** | 3,000년간의 언어 변화 | 🟠 중간 |

### 2.2 데이터적 도전

| 도전 | 상세 | 심각도 |
|------|------|--------|
| **저자원 언어** | 병렬 코퍼스 약 1M 토큰 | 🔴 높음 |
| **텍스트 훼손** | 점토판 물리적 손상 | 🔴 높음 |
| **단어 분절** | 구두점 없음 | 🟠 중간 |
| **장르별 차이** | 점술 문헌은 표의문자 다수 | 🟠 중간 |

### 2.3 기존 연구의 BLEU 성능

| 연구 | 태스크 | BLEU Score |
|------|--------|------------|
| Gutherz et al. (2023) | Cuneiform → English | 36.52 |
| Gutherz et al. (2023) | Transliteration → English | **37.47** |
| Gordin et al. (2021) | Gap Filling (hit@5) | 89% |

> 💡 **인사이트:** Transliteration에서 번역하는 것이 직접 설형문자에서 번역하는 것보다 성능이 높음

---

## 3. 권장 모델 및 접근법

### 3.1 모델 선택 가이드

| 모델 | 장점 | 단점 | 권장도 |
|------|------|------|--------|
| **NLLB-200** | 202개 언어, 저자원 특화 | Akkadian 미지원 | ⭐⭐⭐⭐ |
| **mBART-50** | 다국어 사전학습, 접근성 좋음 | 고대 언어 미지원 | ⭐⭐⭐⭐ |
| **mT5** | 범용성, 텍스트 생성 강점 | 계산 비용 높음 | ⭐⭐⭐ |
| **Custom Transformer** | 완전 맞춤화 | 데이터 부족으로 한계 | ⭐⭐ |

### 3.2 권장 파이프라인

```
[1단계: 사전학습]
┌─────────────────────────────────────────┐
│  NLLB-200 또는 mBART-50 기반            │
│  - 관련 셈어(히브리어, 아랍어) 활용      │
│  - Domain Adaptive Pre-Training (DAPT)  │
└─────────────────────────────────────────┘
                    ↓
[2단계: 데이터 증강]
┌─────────────────────────────────────────┐
│  - Back-translation                      │
│  - publications.csv에서 추가 쌍 추출     │
│  - 저빈도 단어 타겟팅 증강               │
└─────────────────────────────────────────┘
                    ↓
[3단계: Fine-tuning]
┌─────────────────────────────────────────┐
│  - LoRA/Adapter 기반 효율적 학습        │
│  - 문장 수준 정렬된 데이터로 학습        │
└─────────────────────────────────────────┘
                    ↓
[4단계: 후처리]
┌─────────────────────────────────────────┐
│  - Hallucination 검출/수정              │
│  - Lexicon 기반 고유명사 정규화          │
└─────────────────────────────────────────┘
```

### 3.3 데이터 증강 전략

| 기법 | 설명 | 기대 효과 |
|------|------|----------|
| **Back-translation** | 단일 언어 코퍼스로 가상 병렬 데이터 생성 | +10 BLEU 가능 |
| **POS-tagging 기반 치환** | 품사별 유사 단어 교체 | +1.2~2.4 BLEU |
| **저빈도 단어 증강** | 희귀 단어 문맥 다양화 | +2.9 BLEU |
| **고자원 언어 피벗팅** | 히브리어/아랍어 경유 | +1.5~8 BLEU |

---

## 4. Akkadian 특화 전처리 권장사항

### 4.1 특수 토큰 처리

```python
# 권장 특수 토큰 매핑
SPECIAL_TOKENS = {
    '<gap>': '[GAP]',           # 작은 결손
    '<big_gap>': '[BIG_GAP]',   # 큰 결손
    'x': '[UNKNOWN]',           # 판독 불가
    '...': '[ELLIPSIS]',        # 생략
}
```

### 4.2 결정사 (Determinatives) 처리

| 결정사 | 의미 | 처리 방안 |
|--------|------|----------|
| `{d}` | 신의 이름 | `[DEITY]` 토큰 + 원문 유지 |
| `{ki}` | 장소명 | `[PLACE]` 토큰 + 원문 유지 |
| `{m}` / `{f}` | 남성/여성 인명 | `[PERSON_M]` / `[PERSON_F]` |
| `{lú}` | 직업 | `[PROFESSION]` |

### 4.3 Sumerogram 처리 옵션

| 옵션 | 설명 | 장단점 |
|------|------|--------|
| **그대로 유지** | KÙ.BABBAR → KÙ.BABBAR | 정보 손실 없음, 어휘 증가 |
| **Akkadian 변환** | KÙ.BABBAR → kaspum | 일관성, 추가 전처리 필요 |
| **특수 토큰화** | KÙ.BABBAR → [SILVER] | 의미 명확, 맥락 손실 가능 |

**권장:** Lexicon 활용하여 Akkadian 읽기로 변환 후 원본도 함께 보존

### 4.4 문장 정렬 전략

Train 데이터가 문서 수준이므로:

1. `Sentences_Oare_FirstWord_LinNum.csv` 활용
2. `line_number` 기준으로 문장 분리
3. 영어 번역도 문장 단위로 분리 (마침표 기준 + 수동 검증)

---

## 5. 실험 계획 제안

### 5.1 베이스라인 실험

| 실험 | 모델 | 데이터 | 목표 BLEU |
|------|------|--------|----------|
| Baseline 1 | NLLB-200-distilled-600M | train.csv (문서) | 20-25 |
| Baseline 2 | mBART-50 | train.csv (문장 정렬) | 25-30 |
| Baseline 3 | NLLB + LoRA | train + 증강 데이터 | 30-35 |

### 5.2 개선 실험

| 실험 | 추가 요소 | 기대 효과 |
|------|----------|----------|
| Exp 1 | publications.csv 추가 학습 | +3-5 BLEU |
| Exp 2 | Back-translation | +2-4 BLEU |
| Exp 3 | Lexicon 임베딩 | +1-2 BLEU |
| Exp 4 | 앙상블 (NLLB + mBART) | +1-3 BLEU |

### 5.3 최종 목표

| 지표 | 현재 SOTA | 목표 |
|------|----------|------|
| BLEU | 37.47 | 40+ |
| chrF++ | - | 최적화 필요 |
| Geometric Mean | - | 상위 10% |

---

## 6. 핵심 인사이트 요약

### ✅ Akkadian 특성에서 배운 것

1. **SOV 어순** → Attention 메커니즘이 이를 잘 학습할 수 있음
2. **3자음 어근 체계** → 서브워드 토크나이저가 효과적
3. **Sumerogram 혼용** → 멀티태스크 학습 고려 가능
4. **결정사 시스템** → 명시적 의미 정보로 활용 가능

### ⚠️ 주의해야 할 점

1. **Hallucination** → 긴 문장에서 빈번, 후처리 필수
2. **시대별 변이** → 데이터셋 시대 분포 확인 필요
3. **장르 차이** → 점술/행정/서신 문서별 성능 차이 가능
4. **평가 지표** → BLEU와 chrF++ 모두 최적화 필요

### 🚀 승리 전략

1. **문장 정렬 철저히** → Test가 문장 수준이므로 핵심
2. **publications.csv 활용** → 580MB의 추가 학습 자원
3. **Lexicon 적극 활용** → 어휘 커버리지 확대
4. **앙상블** → 여러 모델 조합으로 안정성 확보
5. **특수 토큰 일관성** → gap, determinatives 처리 통일

---

## 참고 자료

### 학술 논문
- [Translating Akkadian to English with neural machine translation (PNAS Nexus, 2023)](https://academic.oup.com/pnasnexus/article/2/5/pgad096/7147349)
- [Filling the Gaps in Ancient Akkadian Texts (arXiv, 2021)](https://arxiv.org/abs/2109.04513)
- [Reading Akkadian cuneiform using natural language processing (PLOS One)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0240511)

### 언어 자료
- [Akkadian Language - Wikipedia](https://en.wikipedia.org/wiki/Akkadian_language)
- [ORACC Akkadian Stylesheet](https://oracc.museum.upenn.edu/doc/help/languages/akkadian/akkadianstylesheet/index.html)
- [Akkadian Language - Britannica](https://www.britannica.com/topic/Akkadian-language)

### 기계 번역 기법
- [Data Augmentation for Low-Resource NMT (ACL)](https://aclanthology.org/P17-2090/)
- [How to fine-tune NLLB-200 (Medium)](https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865)
- [Fine-tuning mBART to unseen languages](https://medium.com/@pablo_rf/fine-tuning-mbart-to-unseen-languages-c2fd55388ac5)

---

*이 리포트는 Claude에 의해 자동 생성되었습니다.*
