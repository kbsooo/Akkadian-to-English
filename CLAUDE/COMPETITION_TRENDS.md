# Deep Past Challenge - 대회 동향 분석

**작성일:** 2026-01-27
**대회 현황:** 진행 중 (마감: 2026-03-23)

---

## 1. 대회 현황 요약

| 항목 | 내용 |
|------|------|
| **참가자 수** | 1,419명 활동 (6,916명 등록) |
| **제출 수** | 14,959건 |
| **팀 수** | 1,338팀 |
| **남은 기간** | 약 2개월 |

---

## 2. 참가자들이 사용하는 주요 모델

웹 검색을 통해 확인한 인기 모델/접근법:

### 2.1 ByT5 (가장 인기)

| 노트북 | 작성자 | 특징 |
|--------|--------|------|
| [byt5-base Training](https://www.kaggle.com/code/xbar19/deep-past-challenge-byt5-base-training) | xbar19 | 기본 ByT5 학습 |
| [byt5-base Training v2](https://www.kaggle.com/code/sayedathar11/deep-past-challenge-byt5-base-training-v2) | sayedathar11 | 개선된 버전 |
| [byt5-akkadian-combined v1.0.6](https://www.kaggle.com/code/manwithacat/byt5-akkadian-combined-v1-0-6) | manwithacat | 통합 버전 |
| [Akkadian ByT5 v2 Ensemble](https://www.kaggle.com/code/manwithacat/akkadian-byt5-v2-ensemble) | manwithacat | 앙상블 접근 |

**ByT5가 인기인 이유:**
- 문자(character) 수준 처리로 희귀 문자/토큰 처리에 강점
- Akkadian의 특수 기호(ṣ, ṭ, š 등) 처리에 유리
- 서브워드 토크나이저 없이도 작동

### 2.2 NLLB (No Language Left Behind)

| 노트북 | 작성자 | 특징 |
|--------|--------|------|
| [NLLB Akkadian Inference](https://www.kaggle.com/code/manwithacat/nllb-akkadian-inference) | manwithacat | NLLB 기반 추론 |

**NLLB 장점:**
- 200+ 언어 지원, 저자원 언어 특화
- Meta의 사전학습된 다국어 모델

### 2.3 T5 변형

| 노트북 | 작성자 | 특징 |
|--------|--------|------|
| [T5 Akkadian Translation Model](https://www.kaggle.com/code/likithagedipudi/t5-akkadian-translation-model) | likithagedipudi | T5 기반 |
| [Akkadian T5 Best Inference](https://www.kaggle.com/code/manwithacat/akkadian-t5-best-inference) | manwithacat | T5 최적화 |

---

## 3. 공유된 주요 리소스

### 3.1 스타터 노트북

| 노트북 | 설명 |
|--------|------|
| [Starter Notebook](https://www.kaggle.com/code/nihilisticneuralnet/deep-past-challenge-starter-notebook) | 입문자용 기본 코드 |
| [Baseline Model](https://www.kaggle.com/code/leiwong/deep-past-challenge-baseline-model) | 베이스라인 모델 |

### 3.2 EDA 노트북

| 노트북 | 작성자 | 설명 |
|--------|--------|------|
| [Comprehensive EDA](https://www.kaggle.com/code/leiwong/deep-past-challenge-comprehensive-eda) | leiwong | 종합 EDA |
| [EDA + Extended Dataset](https://www.kaggle.com/code/leiwong/deep-past-challenge-eda-extended-dataset) | leiwong | 확장 데이터셋 분석 |

### 3.3 핵심 기여자

**manwithacat** - 가장 활발한 기여자
- ByT5, NLLB, T5 등 다양한 모델 실험
- 앙상블 접근법 공유
- 여러 버전의 모델 공개

**xbar19** - ByT5 기반 학습/추론 파이프라인

**leiwong** - EDA 및 베이스라인 모델

---

## 4. 평가 지표 분석

### 4.1 Geometric Mean of BLEU & chrF++

대회는 **BLEU와 chrF++의 기하평균**으로 평가:

```
Score = √(BLEU × chrF++)
```

**BLEU (Bilingual Evaluation Understudy):**
- n-gram 정밀도 기반
- 단어/구문 수준 일치도 측정
- 짧은 번역에 페널티

**chrF++ (Character F-score):**
- 문자 수준 유사도
- 단어 경계 오류에 덜 민감
- 형태론적 변이에 강건

### 4.2 전략적 시사점

| 지표 | 최적화 방향 |
|------|------------|
| BLEU ↑ | 정확한 단어 선택, n-gram 일치 |
| chrF++ ↑ | 문자 수준 유사성, 철자 정확도 |

> 💡 **팁:** 두 지표를 균형있게 최적화해야 함. 한쪽만 높이면 기하평균이 낮아질 수 있음

---

## 5. 참가자들의 주요 접근법 (추정)

웹 검색 결과를 바탕으로 추정한 주요 전략:

### 5.1 데이터 전처리

| 전략 | 설명 |
|------|------|
| **문장 정렬** | Train(문서)을 Test(문장)와 맞추기 위한 분리 |
| **특수 문자 처리** | gap, determinatives 등 일관된 토큰화 |
| **정규화** | Lexicon 활용 어휘 통일 |

### 5.2 모델 선택 트렌드

```
인기도 순위 (추정):
1. ByT5-base     ████████████ (가장 인기)
2. NLLB-200      ████████
3. T5 variants   ██████
4. mBART         ████
5. Custom        ██
```

### 5.3 앙상블 전략

- 여러 모델의 출력 결합
- ByT5 + T5 조합이 인기
- Voting/Averaging 방식

---

## 6. 예상 리더보드 동향

### 6.1 현재 상황 (2026년 1월)

- 대회 시작 후 약 1개월 경과
- 활발한 노트북 공유 진행 중
- 베이스라인 및 스타터 코드 확립

### 6.2 예상 점수 범위 (추정)

| 수준 | 예상 Score | 접근법 |
|------|-----------|--------|
| 입문 | 0.15-0.25 | 기본 seq2seq |
| 중급 | 0.25-0.35 | Fine-tuned ByT5/NLLB |
| 상위 | 0.35-0.45 | 앙상블 + 데이터 증강 |
| 최상위 | 0.45+ | 고급 전처리 + 앙상블 + 추가 데이터 |

> ⚠️ 실제 점수는 Kaggle 리더보드에서 확인 필요

---

## 7. 권장 전략

### 7.1 즉시 시작할 것

1. **스타터 노트북 분석**
   - [Starter Notebook](https://www.kaggle.com/code/nihilisticneuralnet/deep-past-challenge-starter-notebook) 실행
   - 기본 파이프라인 이해

2. **ByT5-base 실험**
   - 현재 가장 인기 있는 접근법
   - [xbar19의 Training 노트북](https://www.kaggle.com/code/xbar19/deep-past-challenge-byt5-base-training) 참고

3. **EDA 노트북 검토**
   - [leiwong의 Comprehensive EDA](https://www.kaggle.com/code/leiwong/deep-past-challenge-comprehensive-eda)
   - 데이터 특성 파악

### 7.2 차별화 전략

| 전략 | 난이도 | 기대 효과 |
|------|--------|----------|
| publications.csv 활용 | 🔴 높음 | +3-5점 |
| 문장 정렬 최적화 | 🟠 중간 | +2-3점 |
| ByT5 + NLLB 앙상블 | 🟠 중간 | +1-2점 |
| Lexicon 임베딩 | 🟡 낮음 | +0.5-1점 |

### 7.3 주의사항

- **Code Competition:** 노트북으로만 제출
- **런타임 제한:** CPU/GPU 9시간
- **인터넷 비활성화:** 모델/데이터 미리 준비
- **Test 데이터:** 현재는 더미, 실제는 ~4,000문장

---

## 8. 유용한 링크

### 대회 페이지
- [Competition Home](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)
- [Discussion Forum](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion)
- [Models](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/models)
- [Code/Notebooks](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/code)

### 주요 노트북
- [ByT5 Training (xbar19)](https://www.kaggle.com/code/xbar19/deep-past-challenge-byt5-base-training)
- [ByT5 Ensemble (manwithacat)](https://www.kaggle.com/code/manwithacat/akkadian-byt5-v2-ensemble)
- [NLLB Inference (manwithacat)](https://www.kaggle.com/code/manwithacat/nllb-akkadian-inference)
- [Baseline Model (leiwong)](https://www.kaggle.com/code/leiwong/deep-past-challenge-baseline-model)

### 관련 자료
- [평가 지표 설명 (Medium)](https://mhr007.medium.com/how-the-deep-past-challenge-scores-your-translations-0050c2c55d59)
- [Deep Past Initiative 공식](https://www.deeppast.org/)
- [Kaggle 공식 발표 (X/Twitter)](https://x.com/kaggle/status/2001291743099007034)

---

## 9. 요약

### 핵심 인사이트

1. **ByT5가 대세:** 문자 수준 처리로 Akkadian 특수 문자에 적합
2. **앙상블 활용:** 여러 모델 조합이 상위권 전략
3. **데이터 전처리 중요:** 문서→문장 정렬이 핵심
4. **평가 지표 균형:** BLEU와 chrF++ 모두 최적화 필요
5. **publications.csv 미개척:** 580MB 추가 데이터 활용 여지

### 추천 로드맵

```
Week 1-2: 베이스라인 구축
├── 스타터 노트북 실행
├── ByT5-base fine-tuning
└── 첫 제출 및 점수 확인

Week 3-4: 성능 개선
├── 데이터 전처리 최적화
├── NLLB 실험
└── 앙상블 시도

Week 5-6: 고급 전략
├── publications.csv 활용
├── 하이퍼파라미터 튜닝
└── 최종 앙상블

Week 7-8: 마무리
├── 코드 정리
├── 런타임 최적화 (9시간 제한)
└── 최종 제출
```

---

*이 리포트는 웹 검색 결과를 바탕으로 작성되었습니다. 최신 정보는 Kaggle 대회 페이지에서 직접 확인하세요.*
