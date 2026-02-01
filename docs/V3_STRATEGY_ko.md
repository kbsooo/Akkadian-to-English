# V3 전략 문서: Train 데이터 + ByT5‑Large + LoRA

이 문서는 **publication 데이터 없이**, 현재 보유한 **train 기반 데이터만으로**  
ByT5‑Large + LoRA(PEFT) 학습을 진행하기 위한 **상세 실행 지침**이다.  
예상 점수는 다루지 않는다.

---

## 1. 목표
- 문장 레벨 데이터로 안정적으로 학습
- 모델 용량(large) 확장 효과와 LoRA의 효율성 검증
- V2 대비 **학습 안정성 + 일반화 개선**을 목표로 함

---

## 2. V2 데이터 구조 상세

현재 `data/v2/`에 존재하는 데이터는 다음과 같다.

### 2.1 문장 레벨 데이터 (권장, 메인 학습용)

**`data/v2/v2_sentence_train.csv`**  
- 생성 경로: `src/v2/build_sentence_dataset.py`  
- 소스: `src/outputs/sentence_pairs_q70_pattern.csv`  
- 형태: 문장 단위 (sentence‑level)  
- 컬럼: `oare_id`, `src`, `tgt`

**`data/v2/v2_sentence_val.csv`**  
- 동일 파이프라인에서 split된 검증 세트  
- 컬럼: `oare_id`, `src`, `tgt`

> 이 두 파일이 V3의 기본 학습/검증 세트로 사용된다.

---

### 2.2 문서 레벨 데이터 (보조/비교용)

**`data/v2/v2_train.csv`**  
- 생성 경로: `src/v2/build_dataset.py`  
- 소스: `data/train.csv`  
- 형태: 문서 단위 (document‑level)  
- 컬럼: `oare_id`, `src`, `tgt`

**`data/v2/v2_val.csv`**  
- `v2_train.csv`에서 분리된 검증 세트  
- 컬럼: `oare_id`, `src`, `tgt`

> 문서 레벨 데이터는 테스트(문장 단위)와 분포가 달라서  
> 메인 학습에는 적합하지 않다.

---

### 2.3 증강 데이터 (문서 레벨, 정제 버전)

**`data/v2/v2_train_augmented.csv`**  
- 생성 경로: `src/v2/augment_data.py`  
- `Sentences_Oare`의 번역을 문서 단위로 합쳐서  
  `published_texts`의 transliteration과 매칭  
- 정합성 리스크가 있어 기본 학습에 권장되지 않음

**`data/v2/v2_train_augmented_clean.csv`**  
- 생성 경로: `src/v2/clean_augmentation.py`  
- 길이 비율 / `<gap>` / `<unk>` 과다 케이스 제거  
- 컬럼: `oare_id`, `src`, `tgt`

**`data/v2/augmentation_audit.csv`**  
- 생성 경로: `src/v2/inspect_augmentation.py`  
- 정렬 이상 후보 샘플 200개 기록

---

### 2.4 정규화 방식 (공통)

`src/v2/normalize.py`  
핵심 요약:
- diacritics 제거 (š/ṣ/ḫ → s/s/h)
- OCR 특수문자 정리 („ … 등)
- `<gap>`, `<unk>` 통일
- whitespace 정규화

이 정규화는 **train/test 모두 동일하게** 적용되는 것을 전제로 한다.

---

## 3. 모델 전략 (ByT5‑Large + LoRA)

### 3.1 선택 이유
- ByT5‑Large는 모델 용량이 커서 표현력이 좋음
- 하지만 데이터 규모가 작아 **full fine‑tuning은 과적합 위험 큼**
- LoRA는 학습 파라미터를 제한하여 **안정성과 효율성 확보**

### 3.2 기본 방향
1) 문장 레벨 데이터만 사용  
2) LoRA 적용  
3) 학습 안정성 우선

---

## 4. 학습 설정 (권장)

### 4.1 공통 설정
- max_source_length: **256**
- max_target_length: **256**
- fp16: **OFF** (수치 안정성)
- gradient checkpointing: **ON**
- warmup_ratio: **0.1**
- max_grad_norm: **1.0**

### 4.2 LoRA 설정 (초기값)
- r: 8
- alpha: 16
- dropout: 0.05
- target modules: attention의 q/v projection

### 4.3 학습 데이터
기본: `v2_sentence_train.csv`, `v2_sentence_val.csv`

> 문서 레벨/증강 데이터는 V3 기본에서 제외한다.  
> 필요한 경우 “추가 실험”으로 분리 운영한다.

---

## 5. 평가 및 검증 절차

1) **Train sanity check**  
   - 학습 시작 후 몇 스텝에서 출력이 정상인지 확인
2) **Validation 지표**  
   - BLEU / chrF / geo_mean
3) **추론 샘플 확인**  
   - train/val 샘플 몇 개를 직접 번역 확인

---

## 6. 리스크 및 대응

| 리스크 | 설명 | 대응 |
|-------|------|------|
| 과적합 | 데이터가 작음 | LoRA 사용, 길이 256, early stop |
| 수치 불안정 | 긴 시퀀스 + fp16 위험 | fp16 OFF, grad clip |
| 분포 불일치 | 문서/문장 mismatch | 문장 레벨만 사용 |

---

## 7. 실행 흐름 요약

1) `data/v2/`에 문장 레벨 파일 확인  
2) LoRA 학습 스크립트 실행  
3) 검증/추론으로 정상 출력 확인  
4) Kaggle 제출 평가

---

## 8. 참고 파일

- 학습 스크립트: `src/v2/akka_v2_train.py` (문장 레벨 기본)
- 문장 데이터 생성: `src/v2/build_sentence_dataset.py`
- 증강 정제: `src/v2/clean_augmentation.py`
- 증강 검사: `src/v2/inspect_augmentation.py`
- 정규화: `src/v2/normalize.py`

---

이 문서는 **현재 데이터 기준 V3 실행을 위한 기술 기록**이며,  
publication 데이터를 포함한 확장 전략은 별도 문서에서 다룬다.
