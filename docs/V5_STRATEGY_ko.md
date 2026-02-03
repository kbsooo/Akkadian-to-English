# V5 데이터/학습 전략 (요약)

## 목표
- OG 데이터를 V5 정규화로 다시 가공하여 `data/v5/` 생성
- Sentence-level 분포를 맞추고, publications 영어 후보를 보조로 활용
- DAPT는 published_texts 전체 transliteration 사용

---

## 1. 데이터 생성 (data/v5)

### 입력
- `data/train.csv`
- `data/Sentences_Oare_FirstWord_LinNum.csv`
- `data/published_texts.csv`
- `data/publications.csv`

### 출력
- `v5_doc_train.csv`, `v5_doc_val.csv`
- `v5_sentence_train.csv`, `v5_sentence_val.csv`
- `v5_publications_candidates.csv`
- `v5_publications_doc_pairs.csv`
- `v5_dapt_translit.txt`
- `v5_stats.json`

---

## 2. V5 정규화 규칙
- `[x]` → `<gap>`
- `…` / `[… …]` → `<big_gap>`
- `[content]` → content (괄호만 제거)
- `<content>` → content (삽입 표기 제거)
- half bracket 제거: `˹ ˺`, `⌈⌉⌊⌋`
- scribal notation 제거: `! ? /` + `:`
- diacritics → ASCII, subscript → 숫자
- literal `<gap>/<big_gap>` 보호 후 복원
- apostrophe line number (`1'`, `1''`)만 제거

---

## 3. Sentence-level 데이터 생성
- **Annotated 우선**: Sentences_Oare의 `first_word_transcription`
  - 없으면 `first_word_spelling`로 fallback
  - `extract_pairs_from_annotations` 사용
- **Rule-based 보조**: 나머지 문서는 규칙 분할
- 필터: `min_src_len`, `min_tgt_len`, `tgt/src ratio`
- `oare_id` 기준 split 유지

---

## 4. Publications 활용
- English 후보만 사용 (doc-level)
- alias 매칭 + 번역 후보 추출
- 가장 긴 번역 1개를 doc-level pair로 저장
- 비영어/unknown은 사용하지 않음

---

## 5. DAPT
- `published_texts` 전체 transliteration 정규화
- 중복 제거 + 길이 필터

---

## 6. 학습 전략 (2-stage)

### Stage A (Optional)
- `v5_publications_doc_pairs.csv`
- 낮은 LR, 1~2 epoch

### Stage B (Main)
- `v5_sentence_train.csv` / `v5_sentence_val.csv`
- LR 1e-4, 6~10 epoch

---

## 실행 예시
```bash
uv run python src/v5/build_v5_data.py --data-dir data --out-dir data/v5
uv run python src/v5/akkadian_v5_train.py
uv run python src/v5/akkadian_v5_infer.py
```
