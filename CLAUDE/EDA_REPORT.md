# Deep Past Challenge - Akkadian to English Translation
## EDA 분석 리포트

**작성일:** 2026-01-27
**대회:** [Deep Past Challenge - Translate Akkadian to English](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)

---

## 1. 대회 개요

### 1.1 배경
- **목표:** 4,000년 된 고대 아시리아 상업 기록(점토판)을 AI로 해독
- **언어:** Old Assyrian (Akkadian의 초기 형태) → English
- **의의:** 박물관에 100년 넘게 번역되지 않은 수천 개의 점토판 해독에 기여

### 1.2 주요 정보
| 항목 | 내용 |
|------|------|
| **주최** | Deep Past Initiative |
| **총 상금** | $50,000 |
| **마감일** | 2026년 3월 23일 |
| **참가자** | 1,419명 (6,916 등록) |
| **평가 지표** | Geometric Mean of BLEU and chrF++ |

### 1.3 상금 구조
| 순위 | 상금 |
|------|------|
| 1위 | $15,000 |
| 2위 | $10,000 |
| 3위 | $8,000 |
| 4위 | $7,000 |
| 5위 | $5,000 |
| 6위 | $5,000 |

### 1.4 제출 요구사항
- **Code Competition:** Notebook으로만 제출 가능
- **런타임 제한:** CPU/GPU 모두 9시간 이내
- **인터넷:** 비활성화
- **외부 데이터:** 공개적으로 이용 가능한 데이터/모델 허용
- **제출 파일명:** `submission.csv`

---

## 2. 데이터 구조

### 2.1 파일 목록 및 크기

| 파일명 | 크기 | 설명 |
|--------|------|------|
| `train.csv` | 1.55 MB | 학습 데이터 (문서 수준 번역) |
| `test.csv` | 0.00 MB | 테스트 데이터 (문장 수준, 더미 데이터) |
| `sample_submission.csv` | 0.00 MB | 제출 형식 예시 |
| `published_texts.csv` | 10.79 MB | 추가 transliteration (번역 없음) |
| `OA_Lexicon_eBL.csv` | 3.39 MB | 아카드어 어휘 사전 |
| `eBL_Dictionary.csv` | 1.55 MB | eBL 사전 |
| `publications.csv` | 553.70 MB | 학술 출판물 OCR 텍스트 |
| `bibliography.csv` | 0.14 MB | 서지 정보 |
| `Sentences_Oare_FirstWord_LinNum.csv` | 1.89 MB | 문장 정렬 보조 데이터 |
| `resources.csv` | 0.09 MB | 추가 리소스 목록 |

**총 데이터 크기:** 600.95 MB

### 2.2 학습 데이터 (train.csv)

**기본 정보:**
- **행 수:** 1,561개 문서
- **컬럼:** `oare_id`, `transliteration`, `translation`
- **결측치:** 없음

**Transliteration (아카드어) 통계:**
| 지표 | 문자 수 | 단어 수 |
|------|---------|---------|
| 최소 | 21 | 3 |
| 최대 | 932 | 187 |
| 평균 | 426.5 | 57.5 |
| 중앙값 | 365.0 | 49.0 |

**Translation (영어) 통계:**
| 지표 | 문자 수 | 단어 수 |
|------|---------|---------|
| 최소 | 6 | 1 |
| 최대 | 3,895 | 744 |
| 평균 | 499.7 | 90.5 |
| 중앙값 | 383.0 | 68.0 |

### 2.3 테스트 데이터 (test.csv)

**주의:** 현재 제공된 test.csv는 **더미 데이터**이며, 실제 채점 시 약 4,000개 문장으로 대체됨

**컬럼 구조:**
- `id`: 고유 식별자
- `text_id`: 문서 식별자
- `line_start`, `line_end`: 문장의 시작/끝 라인 번호
- `transliteration`: 아카드어 transliteration

**핵심 차이점:**
> ⚠️ **Train은 문서(document) 수준, Test는 문장(sentence) 수준**

---

## 3. 텍스트 패턴 분석

### 3.1 아카드어 Transliteration 패턴

**가장 빈번한 단어 Top 10:**
| 순위 | 단어 | 빈도 | 의미 |
|------|------|------|------|
| 1 | a-na | 3,922 | "~에게" (전치사) |
| 2 | ša | 3,441 | "~의" (소유격 표지) |
| 3 | kù.babbar | 3,164 | "은(silver)" |
| 4 | x | 2,693 | 판독 불가 문자 |
| 5 | ma-na | 2,451 | "미나" (무게 단위) |
| 6 | dumu | 1,893 | "아들" |
| 7 | gín | 1,855 | "세겔" (무게 단위) |
| 8 | ù | 1,841 | "그리고" |
| 9 | … | 1,352 | 생략/훼손 |
| 10 | i-na | 1,265 | "~에서" (전치사) |

**특수 마커:**
- `<gap>`: 작은 훼손/결손
- `<big_gap>`: 큰 훼손/결손
- `{ki}`, `{d}` 등: Determinatives (의미 분류 표지)
- `x`: 판독 불가능한 문자

### 3.2 영어 Translation 패턴

**가장 빈번한 단어 Top 10:**
| 순위 | 단어 | 빈도 |
|------|------|------|
| 1 | of | 8,809 |
| 2 | the | 6,895 |
| 3 | and | 4,590 |
| 4 | to | 4,289 |
| 5 | silver | 2,510 |
| 6 | ... | 2,451 |
| 7 | i | 1,968 |
| 8 | for | 1,941 |
| 9 | you | 1,912 |
| 10 | son | 1,899 |

**특수 표기:**
- `[...]`: 435회 등장 (결손 부분)
- `...`: 2,996회 등장 (생략)

### 3.3 소스-타겟 상관관계

- **길이 상관계수:** 0.817 (강한 양의 상관관계)
- 아카드어 길이와 영어 번역 길이가 비례

---

## 4. 보조 데이터 분석

### 4.1 published_texts.csv (8,000개 텍스트)
- 번역이 없는 추가 transliteration 데이터
- `transliteration_orig`: 원본
- `transliteration`: 정제된 버전
- 메타데이터: CDLI ID, 박물관 위치, 장르 등

### 4.2 OA_Lexicon_eBL.csv (39,332개 항목)
**단어 유형 분포:**
| 유형 | 개수 | 비율 |
|------|------|------|
| word (일반 단어) | 25,574 | 65% |
| PN (고유명사) | 13,424 | 34% |
| GN (지명) | 334 | 1% |

### 4.3 publications.csv (580MB)
- 880개 학술 PDF의 OCR 텍스트
- 다양한 언어의 번역 포함 (영어, 독일어, 프랑스어 등)
- **추가 학습 데이터 추출 가능성** ✨

### 4.4 Sentences_Oare_FirstWord_LinNum.csv
- 문장 수준 정렬 보조 데이터
- `line_number`: 점토판 상의 라인 번호
- Train 데이터를 문장 수준으로 분리하는 데 활용 가능

---

## 5. 핵심 도전 과제

### 5.1 Low-Resource Language
- 학습 데이터가 **1,561개 문서**로 매우 제한적
- 고대 언어이므로 사전 학습된 모델 없음

### 5.2 문서 vs 문장 정렬 불일치
- **Train:** 문서 전체 번역
- **Test:** 문장 단위 번역 필요
- → 문장 정렬(sentence alignment) 전처리 필요

### 5.3 형태론적 복잡성
- 아카드어는 굴절어(inflectional language)
- 단일 단어가 영어의 여러 단어에 해당

### 5.4 텍스트 훼손 처리
- 고대 점토판의 물리적 손상
- `<gap>`, `<big_gap>`, `x`, `...` 등의 특수 마커 처리 필요

### 5.5 고유명사 처리
- 인명(PN), 지명(GN)이 자주 등장
- Lexicon 활용하여 정규화 가능

---

## 6. 권장 접근 방법

### 6.1 데이터 전처리
1. **문장 정렬:** `Sentences_Oare_FirstWord_LinNum.csv` 활용
2. **특수 토큰 처리:** `<gap>`, `<big_gap>` → 특수 토큰으로 매핑
3. **정규화:** Lexicon으로 변형 단어 정규화

### 6.2 모델 선택
| 모델 | 장점 | 단점 |
|------|------|------|
| **NLLB** | 다국어, 저자원 언어 지원 | 아카드어 미포함 |
| **mBART** | Seq2seq 사전학습 | Fine-tuning 필요 |
| **T5/mT5** | 범용성 | 계산 비용 |
| **Custom Transformer** | 완전 맞춤화 가능 | 학습 데이터 부족 |

### 6.3 데이터 증강
1. `publications.csv`에서 추가 번역 쌍 추출
2. Back-translation
3. 동의어 치환 (Lexicon 활용)

### 6.4 앙상블 전략
- 여러 모델의 출력 결합
- BLEU와 chrF++ 모두 최적화

---

## 7. 시각화

EDA 시각화는 `CLAUDE/eda_visualizations.png`에 저장되었습니다.

포함된 그래프:
1. 아카드어 Transliteration 길이 분포
2. 영어 Translation 길이 분포
3. 소스-타겟 길이 상관관계
4. 단어 수 분포
5. Top 15 아카드어 단어
6. Lexicon 단어 유형 분포

---

## 8. 다음 단계

- [ ] 문장 수준 정렬 전처리 수행
- [ ] 베이스라인 모델 구축 (예: NLLB fine-tuning)
- [ ] publications.csv에서 추가 학습 데이터 추출
- [ ] 평가 파이프라인 구축 (BLEU + chrF++)
- [ ] 실험 및 하이퍼파라미터 튜닝

---

*이 리포트는 Claude에 의해 자동 생성되었습니다.*
