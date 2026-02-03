# V3 Robust 전략 문서: OCR 노이즈 증강 + DAPT

이 문서는 **Private test를 볼 수 없는 상황**에서,  
**train + published_texts**를 최대한 활용해 **일반화 성능을 끌어올리는 전략**을 정리한 기록이다.  

핵심은 두 가지:
1) **OCR/표기 변형 노이즈 증강 (Translation finetune 단계)**  
2) **DAPT (Domain‑Adaptive Pretraining, published_texts 기반)**

---

## 1) OCR/표기 변형 노이즈 증강 (Translation finetune)

### 목표
- Test 데이터에 존재할 수 있는 **새로운 표기/기호/오류**에 대해 모델의 **복원력(robustness)**을 확보한다.
- **입력(Akkadian)은 깨뜨리고**, **출력(영어)은 그대로** 유지하는 방식.

### 핵심 아이디어
- 정규화는 **train/infer 일치성 확보**에 유리하지만,  
  **미지의 테스트 스타일**에는 대응이 약할 수 있다.
- 따라서 **입력 쪽에만 노이즈를 주어**,  
  모델이 “깨진 입력 → 정상 출력”을 학습하도록 만든다.

### 적용 위치
- **Translation finetune 단계에서만 사용**  
  (train 데이터 `transliteration`에만 적용, `translation`은 보존)

### 대표 노이즈 유형 (예시)
- **Diacritics drop**: `š → s`, `ṭ → t`, `ḫ → h`, `ā → a`
- **OCR 따옴표 변형**: `"` ↔ `„`, `'` ↔ `’`
- **ellipsis/괴호 변형**: `...` ↔ `[…]`
- **하이픈 변형**: `qa-ti` ↔ `qati`
- **숫자 변형**: `₄` ↔ `4`
- **잡음 문자 삽입**: `+`, `·`, `°` 등 OCR 잔재

### 구현 전략 (권장)
- **확률 기반 적용** (예: 샘플마다 0.3~0.5 확률로 노이즈)
- **한 번에 한 종류만** 적용 → 과도한 파괴 방지
- **노이즈 강도는 점진적 상승**  
  (학습 초기: 약하게, 후반: 강하게)

### 기대 효과
- OCR 스타일/비정형 표기에 대한 **일반화 능력 향상**
- **test 스타일이 다를 때도 성능 하락 완화**

### 리스크
- 노이즈가 과하면 **원래 의미를 훼손** → 성능 하락
- 적정 확률/강도 튜닝이 필요

---

## 2) DAPT (Domain‑Adaptive Pretraining) on published_texts

### 목표
- Akkadian transliteration 자체에 익숙해지도록  
  **번역 전 단계에서 모델을 적응**시키는 과정

### 핵심 아이디어
- published_texts는 **번역 쌍이 없더라도**  
  “문자 패턴, 하이픈 구조, 표기 관습”을 학습하는 데 유용하다.
- **Self‑reconstruction 학습**으로 “Akkadian 형태/문법 패턴”을 내재화한다.

### 방식 (T5 스타일)
1) 원문 transliteration에 **span corruption** 또는 **masking** 적용  
2) 모델은 **원문을 복원하도록 학습**

### 적용 위치
- **Translation finetune 전에 1~2 epoch만 가볍게**
- 비용을 크게 늘리지 않고 모델 적응 효과 확보

### 기대 효과
- Akkadian 특유의 **형태적 패턴 학습**
- 이후 번역 학습에서 **수렴 속도 개선**
- test 스타일이 달라져도 **문자 단위 일반화 개선**

### 리스크
- DAPT 단계가 길어지면 **translation 능력에 역효과** 가능
- 반드시 **짧게(1~2 epoch)**, 그리고 translation finetune으로 재정렬

---

## 권장 실행 순서 (저비용 → 고비용)

1) **OCR 노이즈 증강만 적용**  
   - 추가 학습 비용 거의 없음  
   - 성능 향상 가능성이 가장 높음

2) **published_texts DAPT 1~2 epoch**  
   - 추가 비용 발생하지만 적정 수준  
   - 이후 translation fine‑tune 성능 개선 기대

3) **성능이 올라간 후** seed 또는 n‑fold 시도  
   - compute 비용이 크므로 마지막 단계에서만

---

## 간단한 성공 기준 (측정 지표)

- **V2 baseline 대비 BLEU/chrF 상승 여부**
- sample decode에서 **의미 없는 출력(빈 문자열/반복) 감소**
- training loss 정상 감소 + validation 안정적 상승

---

## 요약

- 정규화는 **일치성 확보**, 노이즈 증강은 **일반화 확보**
- published_texts는 **번역 쌍이 없어도 DAPT로 활용 가능**
- 낮은 compute로 효과를 보려면  
  **OCR 노이즈 증강 → 짧은 DAPT → translation finetune** 순서가 가장 현실적

