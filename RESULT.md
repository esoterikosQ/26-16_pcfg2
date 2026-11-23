# RESULT.md — PCFG 파서 구현 결과 정리

문헌 두 편(Jurafsky & Martin *SLP* App E, Manning & Schütze *FSNLP* Ch 11)의
**확률적 문맥자유문법(PCFG) 구구조 파서**를 밑바닥부터 구현하고, 문헌이 제공하는
정답 숫자(golden number)로 검증했다. `plan.md`의 Phase 0–3을 모두 완료했다.
(Phase 4는 선택 항목으로 진행하지 않음 — 근거는 §7.)

---

## 1. 개요

- **목표:** 최소 노력으로 동작하는 PCFG 파서. 문헌 예제 숫자 재현을 각 단계의
  완료 기준(DoD)으로 삼음.
- **범위(구현함):** 트리 확률 계산, 확률적 CKY + Viterbi, Treebank 지도 학습
  (규칙 카운팅→정규화), CNF 정/역변환, OOV/UNK 폴백, PARSEVAL 평가, 트리 시각화.
- **범위(제외):** Inside-Outside EM, Outside, Inside 알고리즘(파싱 정답성과 무관),
  의존 파싱, 신경망 파서 — `plan.md` 설계 판단 그대로.
- **검증:** pytest 회귀 테스트 **33개 전부 통과**.

---

## 2. 실행 환경

- **conda 가상환경** `phdq_pcfg` (Python 3.11).
- 의존성: `nltk`, `pytest` (`requirements.txt`) + 시각화용 `matplotlib`.
- 데이터: **NLTK Penn Treebank 샘플** (`nltk.download('treebank')`, 무료, 3,914문장).

```bash
conda activate phdq_pcfg
pip install -r requirements.txt          # nltk, pytest
pip install matplotlib                    # 시각화용
python -c "import nltk; nltk.download('treebank')"

pytest                                    # 전체 회귀 테스트 (33개)
python -m scripts.parse_and_visualize --sentence "she bought a house near the city ."
```

---

## 3. 모듈 구조

```
pcfg/
  tree.py        # 트리 자료구조, 브래킷 I/O, fencepost↔Manning 인덱스 매핑
  grammar.py     # PCFG 표현, LHS/RHS 양방향 인덱싱, 로그확률
  tree_prob.py   # Phase 1: 트리 확률 (로그공간 규칙확률 곱)
  cky.py         # Phase 2: 확률적 CKY + Viterbi(max+백포인터) + 단항 클로저
  induce.py      # Phase 3: 규칙 유도, CNF 정/역변환, UNK, 폴백, 전처리
  eval.py        # PARSEVAL (라벨 P/R/F1)
  data.py        # NLTK treebank 로더
  draw.py        # 트리 → PNG 시각화 (matplotlib, headless)
tests/
  fixtures.py        # Fig E.1, Fig E.4, Table 11.2 문법·트리 하드코딩
  test_tree.py       # Phase 0: 브래킷 왕복, 인덱스 매핑
  test_tree_prob.py  # Phase 1: §3.1, §3.2
  test_cky.py        # Phase 2: §3.3, §3.4
  test_induce.py     # Phase 3: CNF 왕복, UNK, 폴백, PARSEVAL, 통합 F1
scripts/
  parse_and_visualize.py  # 문장+트리뱅크 → 파스 트리 → PNG
outputs/
  parse_results.txt, trees/tree_01.png ... tree_10.png
```

전체 소스 약 1,500줄(순수 파이썬).

---

## 4. 규약 (Phase 0에서 확정, 이후 불변)

- **로그 공간 계산**으로 언더플로우 방지. CKY/Viterbi는 max라 단순 로그 합으로 충분
  (logsumexp 불필요 — 이를 요구하던 Inside를 범위에서 제외).
- **인덱싱:** fencepost 0..n, 반열림 구간 `[i, j)`를 코드 전역 표준으로. 문헌 표
  (Manning 1..m 포함 구간)는 `tree.py`의 매핑 헬퍼로만 변환.
- **문법 LHS·RHS 양방향 인덱싱** 유지 (CKY 속도에 결정적).

---

## 5. Phase별 구현 + 검증

### Phase 0 — 자료구조 & I/O
- `Tree`(중첩 구조, 리프=str), Penn 브래킷 파서/직렬화(라벨 없는 최상단 래퍼 처리).
- fencepost↔Manning(1-indexed 포함 구간) 매핑 헬퍼 + 단위 테스트.
- **통과:** 브래킷 왕복 일치, 인덱스 매핑 역함수 검증.

### Phase 1 — 트리 확률
- 트리의 모든 국소 규칙 logprob 합 → `P(T)=exp(합)` (Jurafsky Eq E.2).
- **통과(§3.1 Jurafsky Fig E.1):** P(T_left) ≈ 2.2×10⁻⁶.
- **통과(§3.2 Manning Table 11.2):** P(t₁)=0.0009072, P(t₂)=0.0006804,
  문장확률=0.0015876.

### Phase 2 — 확률적 CKY + Viterbi
- CNF 이항/어휘 규칙 삼중 루프, 셀 값 max + 백포인터, `BUILD_TREE` 복원.
- 단항 규칙(A→B)은 **일반화 CKY의 단항 클로저**로 처리(Phase 3 대비 미리 지원).
- **통과(§3.3 Fig E.4):** NP(0,2)=.30·.40·.02=.0024 등 셀 값.
- **통과(§3.4 Manning):** 차트 중간 셀(VP(2,3)=0.126, PP(4,5)=0.18, NP(3,5)=0.01296,
  S(1,3)=0.0126), 경쟁 셀 **VP(2,5)=0.009072**(V NP 경로, Viterbi max ≠ Inside sum),
  S(1,5)=0.0009072, 그리고 **최적 파스 t₁ 복원**.

### Phase 3 — Treebank 지도 학습
- **규칙 유도:** Count(α→β)/Count(α) 정규화.
- **전처리:** `-NONE-` 흔적 제거, 함수태그·인덱스 단순화(`NP-SBJ→NP`), 다중 루트를
  `TOP`으로 래핑.
- **CNF 정변환:** 우분지 이항화, 인공 노드에 표식 `|`(비중첩, Klein-Manning h=∞).
- **CNF 역변환:** 표식 노드를 제거해 원본 n-항 복원(PARSEVAL 전제).
- **OOV/UNK:** 희귀어(빈도 ≤ 임계) → `<UNK>` 치환, 파스 시 미등록어 백오프.
- **폴백:** 파스 실패 시 우분지 트리 강제 반환(전 문장 커버리지 보장).
- **PARSEVAL:** 구성소 (라벨,i,j) 다중집합 교집합 → 라벨 P/R/F1.
- **통과:** CNF 왕복 일치, UNK 경로 파스, 폴백 커버리지, F1 산출.

---

## 6. 검증 앵커 요약 (문헌 골든 넘버)

| 항목 | 문헌 | 기대값 | 결과 |
|---|---|---|---|
| 트리확률 P(T_left) | Jurafsky Fig E.1 | 2.2×10⁻⁶ | ✓ |
| 트리확률 P(T_right) | Jurafsky Fig E.2 | **3.04×10⁻⁷** (erratum 교정) | ✓ |
| 트리확률 P(t₁)/P(t₂) | Manning Table 11.2 | 0.0009072 / 0.0006804 | ✓ |
| 문장확률 | Manning | 0.0015876 | ✓ |
| CKY 셀 NP(0,2) | Jurafsky Fig E.4 | 0.0024 | ✓ |
| 차트 셀 VP(2,5) | Manning §3.4 | 0.009072 | ✓ |
| 최적 파스 | Manning | t₁ 복원 | ✓ |

### erratum 기록 (P(T_right))
Fig E.1 문법은 `VP→Verb NP NP = .05`(이 값이라야 VP 규칙 확률 합=1.0)인데, Fig E.2의
오른쪽 트리는 같은 규칙을 `.10`으로 곱해 6.1×10⁻⁷을 제시한다 — 문법과 어긋나는
문헌 오타. **정규화된 충실한 문법(.05)을 채택**하여 골든 넘버를 3.04×10⁻⁷로 교정했다.
(`plan.md §3.1`, `tests/fixtures.py`, `tests/test_tree_prob.py`에 명시.)

---

## 7. 실데이터 결과 (트리뱅크 지도 학습)

- **설정:** train ≈ 3,500문장 → PCFG 유도(비단말 ~3,100, 어휘 ~5,400), held-out
  문장 파싱, PARSEVAL.
- **PARSEVAL F1 ≈ 0.47** (마이크로 평균, 라벨 포함, 루트·POS 제외).
- **전 문장 커버리지 달성**(폴백 0~소수), 파스 실패는 우분지 폴백으로 보장.

### 왜 문헌의 70%대보다 낮은가 — 데이터 규모
문헌의 "순수 treebank PCFG F1 70%대 초반"은 **전체 WSJ(~4만 문장)** 기준이다. NLTK
샘플은 그 **약 1/10**이라 문법이 희소해 F1이 낮다. 파이프라인 자체는 정상임을
확인함: 첫 학습 문장은 F1=1.0으로 완벽 재현, 골든 넘버 전부 통과. 즉 **모델 결함이
아니라 데이터 크기 효과**다. (학습 데이터를 300→3,500으로 늘리자 F1이 0.33→0.47로
상승하는 추세로 교차 확인.)

### 직접 작성 문장 시각화 (트리뱅크 밖 실문장)
트리뱅크에 없는 뉴스체 자연어 10문장을 유도 문법으로 파싱 → **10문장 전부 폴백 없이
PCFG 파스**, 언어학적으로 타당한 구조 산출. 예:
- `the company said it will buy the firm .` → 내포절 `VP → VBD SBAR[S ...]` 정확 포착.
- `she bought a house near the city .` → 전치사구 `PP(near the city)`를 VP에 부착.

결과: `outputs/parse_results.txt`, `outputs/trees/tree_01.png ~ tree_10.png`.

---

## 8. 성능/속도 노트

- CKY는 문장당 O(m³·|규칙|), **순수 파이썬**. 20단어 문장 ≈ 0.6s.
- 이항화 라벨을 비중첩(h=∞ 표준)으로 바꾸자 라벨 공유가 늘어 **파싱 4배 가속**
  (동일 F1). 
- GPU는 이 희소·분기 위주 구현에 부적합(밀집 텐서 CKY로 재작성해야 이득 — 범위 밖).
  범위 내 개선책: CPU 병렬화(문장 독립), numpy 벡터화, 빔/역치 가지치기, PyPy.

---

## 9. 알려진 한계 / 미구현

- **F1 상한은 데이터 규모에 묶임** (NLTK 샘플). 전체 WSJ가 없으면 문헌 수치 재현 불가.
- **간소화 PARSEVAL:** 정식 `evalb`의 구두점 제외·라벨 병합(`ADVP=PRT` 등) 규칙 미적용.
- **UNK 모델이 단일 `<UNK>`:** 형태·접미사 기반 시그니처 미구현(개선 여지).
- **이항화 h=∞ 고정:** 마르코프화(h=1,2) 토글 미구현.
- **Phase 4(부모 주석·어휘화) 미구현:** 소데이터에선 희소성 악화로 이득이 불확실
  (오히려 하락 가능) → 값싼 개선(마르코프화·UNK 시그니처) 우선이 합리적이라 보류.
  자세한 확장 제안은 `PCFG2.md` 참고.

---

## 10. 테스트 현황

```
pytest -q  →  33 passed
```
- `test_tree.py`(7) 브래킷 왕복·인덱스 매핑
- `test_tree_prob.py`(7) §3.1/§3.2 + 정규화
- `test_cky.py`(7) §3.3/§3.4 셀 값·최적 파스·no-parse
- `test_induce.py`(12) CNF 왕복·전처리·UNK·폴백·PARSEVAL·통합 F1

모든 골든 넘버는 회귀 테스트로 고정되어 있어, 알고리즘 수정 후 즉시 재검증 가능.
