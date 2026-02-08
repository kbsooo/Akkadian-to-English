# V8 Strategy — Akkadian→English Machine Translation

## Executive Summary

V7은 제공된 데이터에서 **~8K 학습 쌍**을 추출하여 ByT5-small (300M)을 학습시킨다.
V8은 **외부 공개 코퍼스**를 통합하여 데이터를 **~70K+ 쌍**으로 확대하고, **ByT5-base (580M)** 또는 **ByT5-large (1.2B)**로 모델을 스케일업한다.

| | V7 | V8 (목표) |
|--|------|-----------|
| 학습 데이터 | ~8K pairs | **~70K+ pairs** |
| 모델 | ByT5-small (300M) | **ByT5-base (580M)** 또는 **ByT5-large** |
| 전략 | 제공 데이터 최대 활용 | 외부 코퍼스 + 2단계 학습 |
| 예상 BLEU | 25-35 | **40-55+** |

---

## Part 1: 현재 V7 데이터 활용 분석

### 1.1 V7이 사용하는 데이터

| Path | 소스 | 추출 방식 | 쌍 수 |
|------|------|----------|-------|
| A | train.csv (1,561 docs) | 문서 단위 직접 사용 | ~1,461 |
| B | train.csv ∩ Sentences_Oare (253 docs) | first-word anchor 문장 분할 | ~1,213 |
| C | published_texts ∩ Sentences_Oare − train (1,164 docs) | first-word anchor 문장 분할 | ~7,263 |
| **합계** | | | **~9,937 → 필터링 후 ~8,068** |

### 1.2 V7이 사용하지 않는 데이터

| 데이터 | 규모 | 미사용 이유 | V8 활용 가능성 |
|--------|------|------------|--------------|
| **EvaCun ORACC Parallel Corpus** | ~65K pairs (4.9MB) | V7에서 미발견 | **최우선** |
| **eBL Transliterated Fragments** | 25K tablets, 350K lines | V7에서 미발견 | **높음** |
| Sentences_Oare-only (283 texts) | 1,305 sentences | 로컬에 transliteration 없음 | 중간 (스크래핑 필요) |
| published_texts 번역 없는 문서 | 5,226 docs | 번역 없음 | 낮음 (스크래핑 필요) |
| publications.csv (OCR) | 216K pages, 554MB | 너무 노이즈 | 낮음 |
| bibliography.csv | 908 entries | 메타데이터만 | 없음 |
| resources.csv | 291 entries | URL/참고문헌 목록 | 간접적 (URL 정보) |

### 1.3 핵심 발견

**V7은 제공된 데이터의 한계 내에서는 거의 최대한 활용하고 있다.**
그러나 **외부 공개 코퍼스 2개**를 활용하면 데이터를 **8배 이상** 늘릴 수 있다.

---

## Part 2: 외부 데이터 소스 (V8 핵심)

### 2.1 EvaCun ORACC Parallel Corpus (최우선)

- **URL**: https://zenodo.org/records/17220688
- **DOI**: 10.5281/zenodo.17220687
- **라이선스**: CC0 (Public Domain) — 자유롭게 사용 가능
- **크기**: 15 MB (train 4.9MB + val 270KB × 3언어)
- **형태**: Line-aligned plain text (UTF-8), 한 줄 = 한 segment
- **추정 규모**: ~65,000 parallel pairs (train 4.9MB / ~75 bytes per line)

**파일 구조:**
```
akkadian_train.txt          # 아카드어 transliteration (로마자)
transcription_train.txt     # 아카드어 Unicode cuneiform
english_train.txt           # 영어 번역
akkadian_validation.txt
transcription_validation.txt
english_validation.txt
```

**활용 방법:**
- `akkadian_train.txt` (transliteration) ↔ `english_train.txt` (translation)을 직접 학습 쌍으로 사용
- ORACC는 여러 아카드어 방언 (Old Assyrian, Old Babylonian, Neo-Assyrian 등)을 포함
- Old Assyrian 이외 방언도 아카드어 전반 이해에 도움 → **pre-training 단계에서 활용**

**주의사항:**
- ORACC 데이터의 transliteration 정규화 방식이 V7과 다를 수 있음
- 방언 태그가 없을 수 있어 Old Assyrian만 필터링이 어려울 수 있음
- → 전체를 사용하되, 2단계 fine-tuning에서 Old Assyrian에 집중

**유저 필요 작업:**
- [ ] Zenodo에서 다운로드 (15 MB, 즉시 가능)
- [ ] transliteration 정규화 방식 확인 (V7 normalization과 차이점)
- [ ] 행 수 확인 (`wc -l akkadian_train.txt`)

---

### 2.2 eBL Transliterated Fragments (높은 가치)

- **URL**: https://github.com/ElectronicBabylonianLiterature/transliterated-fragments
- **규모**: ~25,000 tablets, 350,000+ lines
- **형태**: JSON (ATF format with `#tr.en:` 영어 번역 태그)
- **라이선스**: 논문 동반 공개 데이터 (확인 필요)
- **방언**: 다양 (Neo-Assyrian, Babylonian 등)

**영어 번역 존재 여부:**
JSON 내부에 `#tr.en:` 태그로 영어 번역이 포함된 항목이 있음:
```
#tr.en: If a man has been seized by a ghost: you parch (and) mix...
```

**활용 방법:**
- JSON 파싱하여 transliteration + `#tr.en:` 영어 번역 쌍 추출
- 모든 tablet에 번역이 있는 것은 아님 → 번역 있는 항목만 필터링
- 추정: 전체의 10-30%에 번역 존재 → **~2,500-7,500 tablets, ~35K-105K lines**

**유저 필요 작업:**
- [ ] GitHub repo clone (`git clone`)
- [ ] JSON 파일 구조 파악 (tablet별 1 JSON 파일인지 등)
- [ ] `#tr.en:` 태그가 있는 tablet 수 확인
- [ ] 라이선스 정확한 확인 (학술 데이터 재사용 조건)

---

### 2.3 ORACC 직접 접근

- **URL**: http://oracc.museum.upenn.edu/
- **규모**: 500K+ 텍스트 (lemmatized)
- **형태**: JSON API 또는 ATF 파일
- **Old Assyrian 프로젝트**: ORACC에 Old Assyrian 전용 sub-project 존재 여부 확인 필요

**EvaCun이 ORACC에서 추출한 것이므로, EvaCun으로 충분할 가능성 높음.**
ORACC 직접 접근은 EvaCun에 없는 추가 데이터가 필요할 때만.

---

### 2.4 Sentences_Oare-only 텍스트 (283 문서, 1,305 문장)

이 문서들은 Sentences_Oare에 번역이 있지만, published_texts.csv에 transliteration이 없다.

**활용 방법:**
- OARE 웹사이트 (`online_transcript` URL)에서 transliteration 스크래핑
- 또는 CDLI (`cdli_id` 경유)에서 transliteration 획득

**유저 필요 작업:**
- [ ] Sentences_Oare에서 text_uuid 추출 (published_texts에 없는 것들)
- [ ] OARE 웹사이트 접근하여 transliteration 수동 확인 또는 스크래핑 스크립트 작성
- [ ] 수작업 양: ~283개 문서 (자동화 가능하면 ~30분)

---

### 2.5 AICC Translation URLs (5,226 문서)

published_texts.csv의 `AICC_translation` 컬럼은 실제로 `aicuneiform.com/search?q=...` **URL**이다 (번역 텍스트가 아님).

**활용 방법:**
- URL 스크래핑하여 번역 텍스트 획득
- 5,004개 문서에 AICC URL 존재

**유저 필요 작업:**
- [ ] aicuneiform.com 접근하여 URL 구조 확인
- [ ] 해당 URL에 실제 영어 번역이 있는지 확인
- [ ] 스크래핑 스크립트 작성 (robots.txt 확인 필요)
- **우선순위: 낮음** (EvaCun + eBL로 충분할 가능성)

---

## Part 3: 데이터 확장 로드맵

### Phase 1: EvaCun 통합 (즉시, 유저 작업 최소)

```
현재 V7:  ~8,000 pairs
+ EvaCun: ~65,000 pairs (추정)
= ~73,000 pairs
```

**작업:**
1. Zenodo에서 다운로드 (15 MB)
2. `akkadian_train.txt` ↔ `english_train.txt` 쌍 구성
3. V7 normalization 적용 (EvaCun transliteration 형식 분석 후)
4. V7 데이터와 중복 제거 (exact match + fuzzy dedup)
5. 품질 필터링 (길이, 비율, 빈 문자열 등)

**예상 결과:** ~60K-65K 새로운 학습 쌍 추가

---

### Phase 2: eBL Fragments 추출 (유저 작업 중간)

```
Phase 1 이후: ~73,000 pairs
+ eBL:        ~35,000-105,000 pairs (추정, 번역 있는 것만)
= ~108,000-178,000 pairs
```

**작업:**
1. GitHub repo clone
2. JSON 파싱 스크립트 작성 (transliteration + `#tr.en:` 추출)
3. 번역 있는 항목만 필터링
4. V7 normalization 적용
5. 중복 제거 + 품질 필터링

**예상 결과:** ~35K-100K 새로운 쌍 (번역 커버리지에 따라)

---

### Phase 3: 추가 데이터 확보 (유저 수작업/스크래핑)

```
Phase 2 이후: ~108,000-178,000 pairs
+ Sentences_Oare-only: +1,305
+ AICC scraping:       +? (미확인)
= ~110,000-180,000 pairs
```

**우선순위 낮음** — Phase 1+2로 충분할 가능성 높음.

---

## Part 4: 모델 스케일업 전략

### 4.1 데이터 크기별 모델 선택

| 데이터 규모 | 추천 모델 | 파라미터 | 비고 |
|------------|----------|---------|------|
| ~8K (V7) | ByT5-small | 300M | 현재 V7 |
| ~20K-50K | **ByT5-base** | **580M** | Phase 1만으로 충분 |
| ~70K+ | **ByT5-base** | **580M** | 최적 (과적합 위험 낮음) |
| ~150K+ | ByT5-large | 1.2B | Phase 2까지 완료 시 고려 |

### 4.2 ByT5-base vs ByT5-large

| | ByT5-base | ByT5-large |
|--|-----------|------------|
| Parameters | 580M | 1.2B |
| d_model | 1536 | 2048 |
| Layers | 18 enc + 6 dec | 36 enc + 12 dec |
| T4 16GB | Batch 2-4 + grad ckpt | OOM (FP32) |
| A100 40GB | Batch 8-16 + BF16 | Batch 4-8 + BF16 |
| 최소 데이터 | ~20K+ | ~50K+ |
| 학습 시간 (A100) | ~4-8h | ~12-24h |

**추천: Phase 1 완료 후 ByT5-base부터 시작.** Phase 2까지 완료되면 ByT5-large 실험.

### 4.3 2단계 학습 전략 (Transfer Learning)

```
Stage 1: Pre-training on ALL Akkadian data (~70K-180K pairs)
  ├── EvaCun (all dialects: Old Babylonian, Neo-Assyrian, etc.)
  ├── eBL fragments (mixed dialects)
  └── Goal: 아카드어 전반의 문법/어휘 이해

Stage 2: Fine-tuning on Old Assyrian data (~8K pairs)
  ├── V7 train data (Old Assyrian only)
  ├── Lower LR (1e-5 → 3e-6)
  ├── Fewer epochs (3-5)
  └── Goal: Old Assyrian 특화 (대회 대상 방언)
```

**근거:** 아카드어 방언들은 문법 구조가 유사하므로 (Semitic language family 내), 넓은 Akkadian 데이터로 pre-training하면 Old Assyrian 성능도 향상된다. NLP에서 multilingual pre-training → monolingual fine-tuning과 동일한 원리.

---

## Part 5: V8 구현 계획

### 5.1 파일 구조

```
src/v8/
├── build_v8_data.py              # 데이터 빌드 (V7 + EvaCun + eBL)
├── akkadian_v8_pretrain.py       # Stage 1: 전체 Akkadian pre-training
├── akkadian_v8_finetune.py       # Stage 2: Old Assyrian fine-tuning
├── akkadian_v8_infer.py          # 추론 (V7 기반 + 개선)
└── analyze_evacun.py             # EvaCun 데이터 분석/정규화 스크립트

data/v8/
├── evacun/                       # EvaCun 원본 (Zenodo 다운로드)
│   ├── akkadian_train.txt
│   ├── english_train.txt
│   └── ...
├── ebl/                          # eBL fragments (GitHub clone)
│   └── *.json
├── v8_pretrain_train.csv         # Stage 1 학습 데이터 (all Akkadian)
├── v8_pretrain_val.csv
├── v8_finetune_train.csv         # Stage 2 학습 데이터 (Old Assyrian = V7)
├── v8_finetune_val.csv
└── v8_glossary.json              # V7 glossary 재사용
```

### 5.2 데이터 빌드 파이프라인

```
[1] V7 데이터 (8K pairs, Old Assyrian)
     └── build_v7_data.py 재사용

[2] EvaCun 데이터 (~65K pairs, mixed Akkadian)
     ├── Download from Zenodo
     ├── Parse akkadian_train.txt + english_train.txt
     ├── Normalize (V7 normalization 적용, 형식 차이 보정)
     ├── Dedup against V7
     └── Quality filter

[3] eBL 데이터 (~35K-100K pairs, mixed)
     ├── Clone GitHub repo
     ├── Parse JSON, extract #tr.en paired lines
     ├── Normalize
     ├── Dedup against V7 + EvaCun
     └── Quality filter

[4] 합치기
     ├── pretrain_data = [1] + [2] + [3]   (~110K-180K)
     ├── finetune_data = [1] only            (~8K, Old Assyrian)
     └── Train/Val split (90/10, stratified by source)
```

### 5.3 학습 파이프라인

```
Stage 1: Pre-training (A100 권장)
  ├── Model: ByT5-base (from scratch or HF checkpoint)
  ├── Data: v8_pretrain_train.csv (~100K+)
  ├── Epochs: 10-15
  ├── LR: 5e-5
  ├── Batch: 16 (A100 BF16) or 4 (T4 FP32 + grad ckpt)
  ├── Output: pretrained_akkadian_base/
  └── Time: ~6-12h (A100)

Stage 2: Fine-tuning (T4 가능)
  ├── Model: pretrained_akkadian_base/ (Stage 1 output)
  ├── Data: v8_finetune_train.csv (~8K, Old Assyrian)
  ├── Epochs: 5-8
  ├── LR: 3e-6 (10× lower than Stage 1)
  ├── Early stopping patience: 3
  ├── Output: final_model/
  └── Time: ~1-2h (T4)
```

---

## Part 6: 유저가 직접 해야 할 작업

### 즉시 가능 (다운로드만)

| 작업 | 방법 | 시간 |
|------|------|------|
| EvaCun 다운로드 | `wget https://zenodo.org/records/17220688/files/akkadian_train.txt` 등 6개 파일 | 1분 |
| EvaCun 행 수 확인 | `wc -l akkadian_train.txt` | 1초 |
| EvaCun 샘플 확인 | `head -20 akkadian_train.txt` / `head -20 english_train.txt` | 1초 |

### 약간의 작업 필요

| 작업 | 방법 | 시간 |
|------|------|------|
| eBL repo clone | `git clone https://github.com/ElectronicBabylonianLiterature/transliterated-fragments` | 5-10분 |
| eBL 번역 유무 확인 | `grep -r "#tr.en:" ebl_data/ \| wc -l` | 1분 |
| eBL 라이선스 확인 | repo의 LICENSE 파일 확인 | 1분 |

### 스크래핑 필요 (선택사항)

| 작업 | 방법 | 시간 |
|------|------|------|
| Sentences_Oare-only transliteration | OARE 웹사이트에서 283개 문서 스크래핑 | 1-2h |
| AICC 번역 | aicuneiform.com에서 5K URL 스크래핑 | 2-4h |

---

## Part 7: 리스크 분석

### 7.1 EvaCun 방언 불일치

**리스크:** EvaCun은 ORACC 전체를 포함하므로 Old Babylonian, Neo-Assyrian 등 다양한 방언 혼재.
대회는 Old Assyrian만 평가.

**완화:** 2단계 학습으로 해결. Stage 1에서 범아카드어 학습 → Stage 2에서 Old Assyrian 특화.
실제로 이 접근이 단일 방언만 학습하는 것보다 성능이 높은 경우가 많음 (cross-lingual transfer).

### 7.2 정규화 불일치

**리스크:** EvaCun/eBL의 transliteration 형식이 V7과 다를 수 있음 (예: 다른 gap 마킹, 다른 diacritc 처리)

**완화:**
- EvaCun 샘플을 먼저 확인하여 형식 차이 파악
- V7 normalization 함수를 확장하여 EvaCun 형식도 처리
- 테스트: 정규화 후 V7 데이터와 EvaCun 데이터의 문자 분포 비교

### 7.3 ByT5-base VRAM

**리스크:** ByT5-base (580M)은 T4 16GB에서 batch=2-4만 가능. 학습이 매우 느릴 수 있음.

**완화:**
- A100 사용 (BF16 + batch=16)
- T4에서는 batch=2 + grad_accum=8 + gradient_checkpointing
- 또는 Kaggle T4×2 사용

### 7.4 데이터 품질

**리스크:** 외부 데이터의 번역 품질이 V7 (OARE 검증 데이터)보다 낮을 수 있음.

**완화:**
- 품질 필터링 강화 (길이 비율, 빈 번역, 반복 패턴 제거)
- Stage 2 fine-tuning에서 고품질 V7 데이터만 사용
- 검증 셋은 항상 V7 val에서만 (대회 도메인과 일치하도록)

### 7.5 대회 규정: 외부 데이터 허용 여부

**리스크:** Kaggle 대회에서 외부 데이터 사용이 금지되어 있을 수 있음.

**확인 필요:**
- [ ] 대회 규칙 페이지에서 "External Data" 섹션 확인
- [ ] Discussion 탭에서 외부 데이터 관련 질문/답변 확인
- Kaggle은 일반적으로 **공개 데이터는 허용**하되 Discussion에 출처 공유 요구
- EvaCun은 CC0, eBL은 학술 공개 데이터 → 대부분 허용될 가능성 높음

---

## Part 8: 예상 성능

### V7 vs V8 예측

| | V7 (current) | V8 Phase 1 only | V8 Full |
|--|-------------|-----------------|---------|
| 학습 데이터 | ~8K | ~73K | ~110K-180K |
| 모델 | ByT5-small (300M) | ByT5-base (580M) | ByT5-base/large |
| 예상 BLEU | 25-35 | 38-48 | 45-55+ |
| 예상 chrF | 35-45 | 48-58 | 55-65+ |
| Geo mean | 30-40 | 43-53 | 50-60+ |

**근거:**
- 데이터 8× 증가 → 일반적으로 BLEU 10-15pt 향상
- 모델 스케일업 (300M → 580M) → BLEU 3-5pt 추가 향상
- 2단계 학습 (pre-train → fine-tune) → BLEU 2-5pt 추가 향상
- 합계: V7 대비 **~15-25pt BLEU 향상** 예상

---

## Part 9: 우선순위 요약

```
[1] EvaCun 다운로드 + 분석        ← 최우선, 즉시 가능, 가장 큰 임팩트
[2] 대회 규정 확인 (외부 데이터)   ← [1]과 병행
[3] EvaCun normalization 맞추기   ← [1] 분석 후
[4] V8 build script 작성         ← [3] 완료 후
[5] ByT5-base pre-training       ← [4] 완료 후
[6] Old Assyrian fine-tuning     ← [5] 완료 후
[7] eBL fragments 추출 (선택)     ← [5] 결과 보고 결정
[8] ByT5-large 실험 (선택)        ← 데이터 충분 시
```

---

## Part 10: 즉시 실행 가능한 명령어

```bash
# EvaCun 다운로드
mkdir -p data/v8/evacun
cd data/v8/evacun
wget https://zenodo.org/records/17220688/files/akkadian_train.txt
wget https://zenodo.org/records/17220688/files/english_train.txt
wget https://zenodo.org/records/17220688/files/akkadian_validation.txt
wget https://zenodo.org/records/17220688/files/english_validation.txt

# 규모 확인
wc -l akkadian_train.txt english_train.txt
head -5 akkadian_train.txt
head -5 english_train.txt

# eBL 클론
cd data/v8
git clone https://github.com/ElectronicBabylonianLiterature/transliterated-fragments ebl
cd ebl
# 번역 포함 파일 수 확인
grep -rl "#tr.en:" . | wc -l
```
