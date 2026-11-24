# 한국어 PCFG 파서 작업 로그

- 시각은 KST. 위치 표기: **[gsm]** 로컬(`phdq_pcfg2/`), **[itcerdo]** `~/projects/phdq_pcfg2` (명령의 상대경로는 별도 표기가 없으면 저장소 루트, `.venv` 활성화 상태).
- 기록 규칙은 `CLAUDE.md` 참조. 새 기록은 맨 아래에 시간순으로 추가한다.

---

## 2026-11-24

### 07:09–07:50 · 상위 설계·하위 프로젝트 검토, 방향 결정 [gsm]

- 작업: `phdq_prob/korean_gec_noisy_channel_design.md`(HNC 설계안 A·B·C)와 `phdq_noisy`, `phdq_pcfg`, `phdq_pcfg2`를 읽고 역할을 대응시킴. pcfg2는 pcfg 코드 복사본이며 `data/`·`plan/`이 비어 있음을 확인.
- 결정(사용자): 말단 단위 = 형태소, 세종 구문분석 말뭉치 사용(보유, 추후 배치), C++ CKY 이식을 초기에 진행, HNC 공통 인프라(`phdq_prob`)와 병행.
- 결과: `CLAUDE.md`(한국어판), `plan.md` 작성.

### 07:52–08:06 · 원격 작업 방식 확정 [gsm, itcerdo]

- 작업: `phdq_bart2`의 `ssh.md`·`plan.md`·`plan2.md`·`NEURON.md`에서 운용 규칙 확인(gsm은 편집·기록, itcerdo에서 실행, GitHub로 코드 이동, `log.md` 기록, 지정 디렉토리 밖 접근 금지). 사용자 결정: itcerdo `~/projects/phdq_pcfg2` 신규, 저장소 `esoterikosQ/26-16_pcfg2`.

```bash
ssh -o BatchMode=yes itcerdo 'hostname; nproc; free -g | head -2; python3 --version; g++ --version | head -1; cmake --version | head -1; which tmux git uv'
gh repo view esoterikosQ/26-16_pcfg2 --json visibility,isEmpty
```

- 결과: itcerdo 24코어·RAM 60GB·Python 3.12.3·g++ 13.3·cmake 3.28·tmux·git 있음, uv 없음. GitHub 저장소는 비어 있고 **public** → 데이터·`papers/`·말뭉치 문장이 든 `outputs/`는 `.gitignore`로 제외.
- 제약: `phdq_pcfg`, `phdq_noisy`, `phdq_bart2`는 수정 금지(읽기 전용 참조).

### 08:06–08:12 · K0 저장소 초기화·itcerdo 환경 구성 [gsm, itcerdo]

- 작업: gsm에서 git 초기화 후 `esoterikosQ/26-16_pcfg2`에 push. itcerdo에 `~/projects/phdq_pcfg2` 생성·clone, `.venv` 구성(캐시·NLTK 데이터를 저장소 안으로 고정), 기존 테스트 실행.

```bash
# [gsm]
git init -b main && git add -A && git commit && git push -u origin main      # 62ae14a
# [itcerdo]
mkdir -p ~/projects/phdq_pcfg2 && cd ~/projects/phdq_pcfg2 && git clone https://github.com/esoterikosQ/26-16_pcfg2.git .
python3 -m venv .venv    # activate 끝에 PIP_CACHE_DIR·XDG_CACHE_HOME·MPLCONFIGDIR·NLTK_DATA 를 저장소 안으로 지정
pip install -r requirements.txt
python -c "import nltk, os; nltk.download('treebank', download_dir=os.environ['NLTK_DATA'])"
pytest -q
```

- 결과: **33 passed (1.32s)**. Python 3.12.3, pybind11 3.1.0. NLTK treebank는 `nltk_data/corpora/treebank`에 받아짐. NLTK가 "상위 디렉토리가 그룹 쓰기 가능(umask 0002)" 경고를 냈으나 저장 위치는 저장소 안이며 지정 디렉토리 밖에는 쓰지 않음. → **K0 완료.**

### 08:13–08:25 · K1 C++ CKY 엔진 1차 구현 [gsm]

- 작업: `pcfg/cky.py`(참조)의 순회 순서·동점 처리를 그대로 옮긴 C++ 엔진과 래퍼·테스트·벤치 스크립트 작성.
  - `pcfg/_ckycpp.cpp`(pybind11 모듈 `pcfg._ckycpp`): 셀 = 삽입 순서 보존 희소 벡터(+라벨 정렬 인덱스), 셀 작성 중에만 비단말 수 크기의 위치 배열(scratch) 사용. 이항 (B, C) 조회는 개방 주소 해시. 후보 `(lp_rule + lp_b) + lp_c`, 엄격한 `>` 갱신, 단항 클로저는 패스마다 셀 스냅숏. 백포인터로 트리를 전위 부호화해 반환.
  - `setup.py`: `Pybind11Extension`, C++17, `-O3 -ffp-contract=off` (fast-math 없음).
  - `pcfg/cky_cpp.py`: 문법 RHS 인덱스를 목록 순서·중복 그대로 정수 id 배열로 컴파일(`math.log(p)`는 Python에서 계산), `PCFG._rev`로 캐시 무효화(`grammar.py`의 `add()`에서 증가). 같은 `ParseResult` 반환, 차트는 `cell_logprob`·`cell_prob`·`n` + 비교용 `cell_score`·`cell_back`.
  - `pcfg/induce.py`: `parse_sentence(..., engine="py"|"cpp")`, 기본값 `"py"`.
  - 테스트: `tests/test_cky.py` 골든 넘버를 `parse` 픽스처(`["py", "cpp"]`)로 두 엔진 모두 검증(미빌드 시 skip 아닌 실패), no-parse의 `logprob == -inf` 확인 추가. `tests/test_cky_cpp.py`: 경계 사례, 무작위 문법 셀 전체 동치(삽입 순서·값·백포인터), `engine=` 전환, 트리뱅크 held-out 동치.
  - `scripts/bench_cky.py`: 길이 구간별 두 엔진 속도 비교.
- 명령: (편집만)
- 결과: 빌드·테스트 전. **규칙 위반 기록**: `_ckycpp.cpp`의 문자열 몇 곳(익명 → 이름 있는 namespace 등)을 고치려고 gsm에서 `python3` 한 줄 스크립트를 1회 실행함(빌드·테스트·데이터 처리는 아니지만 "gsm에서 python 실행 금지"에 해당). 이후 편집은 편집 도구로만 함.

### 08:26–08:40 · K1 1차 빌드·테스트, 이항 조합 최적화 [itcerdo, gsm]

- 작업: 1차 구현(970a39a)을 itcerdo에서 빌드·테스트. 트리뱅크 동치 테스트 시간과 전체 문법(3500문장)에서의 C++ 속도를 따로 측정(임시 스크립트를 stdin으로 실행, 파일 생성 없음).

```bash
# [itcerdo]
git pull && python setup.py build_ext --inplace    # 경고 없음. 실제 플래그: ... -O2 -Wall ... -std=c++17 -O3 -ffp-contract=off
pytest -q -k "not treebank_heldout"
pytest -q -k treebank_heldout --durations=3
python - <<"EOF"  …  EOF                            # 길이별 Python/C++ 시간 측정
```

- 결과:
  - `pytest -k "not treebank_heldout"`: **48 passed (1.39s)** (골든 넘버 두 엔진 각 7개, 무작위 문법 셀 전체 동치 등).
  - 트리뱅크 동치(학습 1000문장 ≤25단어, held-out 200문장 ≤25단어): **통과, 불일치 0/200** (파스 성공 198). 단 **128.8s** 걸려 기본 테스트로는 너무 김 → Python 엔진 시간이 길이에 따라 급증(10단어 0.06s, 18단어 0.54s, 25단어 2.1s/문장). held-out 상한을 **≤18단어**로 낮춤(예상 약 40s).
  - 전체 문법(3500문장: 비단말 3113, 이항 6476(쌍 5485), 단항 119, 어휘 6668) C++ 1차: 길이 40대 0.43s, 50대 0.94s/문장, 셀 평균 라벨 수 300–440. 원인: 분할마다 |왼쪽|×|오른쪽| 쌍 전부를 해시 조회.
  - 최적화(순서 보존): 왼쪽 라벨 B마다 짝 목록(규칙이 있는 C)이 오른쪽 셀보다 짧으면 짝 중 오른쪽 셀에 있는 C만 모아 오른쪽 셀 삽입 순위로 정렬해 처리, 아니면 기존 해시 조회. 규칙 없는 (B, C)는 Python에서도 무동작이므로 relax 호출열이 Python과 같음. 무작위 문법 동치 테스트를 300 → 1000 문법, 길이 ≤10으로 확대.

### 08:14–08:52 · K1 C++ CKY 엔진 구현 (백그라운드 에이전트), 벤치마크 tmux 실행 중 [gsm, itcerdo]

- 작업: 백그라운드 에이전트가 `pcfg/_ckycpp.cpp`(pybind11)·`pcfg/cky_cpp.py`·`setup.py`·동치 테스트·`scripts/bench_cky.py` 구현. 커밋 `970a39a`(1차 구현), `4d8a79f`(이항 조합을 짝 목록 기반으로 최적화, 순회 순서 보존). 에이전트는 이 로그를 남기기 전에 사용자 네트워크 차단 준비로 08:50경 중단됨 → 이 항목은 메인 세션이 상태를 확인해 대신 기록.

```bash
# [itcerdo] tmux 세션 pcfg2-k1-bench (08:43 시작)
python -u -m scripts.bench_cky --py-budget 300 > outputs/bench_cky_k1.txt 2>&1
```

- 결과(08:52 시점 부분 출력): train 3,500 / held-out 414문장, 문법 비단말 3,113·단어 5,374·이항 6,476·단항 119·어휘 6,668.

| 길이 | C++ s/문장 | Python s/문장 | 가속 | 트리 일치 |
|---|---|---|---|---|
| ≤10 | 0.0006 | 0.146 | 231× | 34/34 |
| 11–20 | 0.0049 | 2.915 | 576× | 103/103 |

- 미완료: 벤치마크 나머지 구간(21–30, 31–40, 41+), 전체 테스트 결과의 로그 기록, `plan.md` K1 상태 갱신. 재접속 후 `outputs/bench_cky_k1.txt`를 확인해 마무리한다.

## 2026-11-24 (재접속 후)

### 10:54–11:05 · K1 마무리: 벤치마크 결과·전체 테스트·코드 검토 [gsm, itcerdo]

- 작업: 네트워크 차단 중 tmux `pcfg2-k1-bench`에서 끝난 벤치마크 결과 확인, itcerdo에서 `git pull`(`566f1c5`)·재빌드·전체 테스트, 에이전트가 남긴 코드 검토(`pcfg/cky_cpp.py` 컴파일·캐시·트리 복원, `pcfg/_ckycpp.cpp` 메모리 구조).

```bash
cat outputs/bench_cky_k1.txt
python setup.py build_ext --inplace && pytest -q
```

- 결과: **49 passed (41s)** — 골든 넘버 테스트가 Python·C++ 두 엔진 모두 통과, 영어 트리뱅크 동치 테스트 통과. 벤치마크(train 3,500 / held-out 414문장):

| 길이 | 문장 수 | 평균 길이 | C++ s/문장 | C++ 최대 s | Python s/문장 (측정 수) | 가속 | 트리·logprob 일치 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ≤10 | 34 | 7.9 | 0.0006 | 0.002 | 0.146 (34) | 231× | 34/34 |
| 11–20 | 129 | 16.2 | 0.0049 | 0.010 | 2.915 (103) | 576× | 103/103 |
| 21–30 | 166 | 25.1 | 0.0179 | 0.038 | 14.516 (22) | 883× | 22/22 |
| 31–40 | 69 | 34.4 | 0.0461 | 0.095 | 52.619 (7) | 1136× | 7/7 |
| 41+ | 16 | 45.4 | 0.1134 | 0.212 | — (예산 초과) | — | — |

- 코드 검토: 규칙 로그확률은 Python `math.log`로 계산해 전달, `-ffp-contract=off`, 셀은 삽입 순서 보존 희소 벡터 + 라벨 정렬 인덱스, 셀 계산 중에만 라벨 수 크기 임시 배열, 트리 복원은 반복형(재귀 깊이 문제 없음). 문제 없음. → **K1 완료.**
- **규칙 위반 기록**: 빌드 출력을 `/dev/null` 대신 `/tmp/null`로 잘못 리다이렉트해 itcerdo 지정 디렉토리 밖에 파일(18바이트)을 만듦. 확인 즉시 `rm -f /tmp/null`로 삭제.

### 11:00–11:10 · 모두의 말뭉치 구문 분석(DP) 반입·스키마 확인 [gsm, itcerdo]

- 작업: 사용자가 지정한 외장 디스크 `/Volumes/T31/NIKL`의 `DP_json.zip`(2020 v2.0), `DP2024.zip`, `DP2025.zip`을 itcerdo `data/raw/`로 전송·압축 해제(`DP_json.zip` 내부 디렉토리명이 CP949라 `unzip -O cp949`). 스키마·라벨·지배소 방향·투사성 확인.

```bash
# [gsm] /Volumes/T31/NIKL
rsync -a DP_json.zip DP2024.zip DP2025.zip itcerdo:projects/phdq_pcfg2/data/raw/    # 8초, 바이트 수 일치
# [itcerdo] data/
unzip -O cp949 raw/DP_json.zip → NIKL_DP_v2.0/ ; unzip raw/DP2024.zip ; unzip raw/DP2025.zip -d NIKL_DP_2025_v1.0
```

- 결과: `NIKL_DP_v2.0`(NX 신문 2019·SX 구어 2020, 1.7GB), `NIKL_DP_2024_v1.0`(1.1GB), `NIKL_DP_2025_v1.0`(128MB). itcerdo 디스크 여유 68GB.
  - 문장 = `word`(어절 form·begin·end) + `DP`(word_id·head·label·dependent). **형태소 분석 없음.**
  - 라벨 = 세종 구구조 표지(`NP`, `NP_SBJ`, `NP_OBJ`, `NP_AJT`, `VP`, `VP_MOD`, `VNP`, `AP`, `DP`, `IP` …, 28–37종).
  - 표본(2025 NX 13,907 · 2025 SX 16,439 · 2020 NX 20,000문장): 루트 1개 100%, **지배소가 오른쪽 100%(핵어말)**, 비투사 문장 0% / 0% / 0.51%.
- 결정: 의존 구조지만 세종식 구 표지·핵어말·투사적이므로 **결정적으로 구구조로 변환해 K2 트리뱅크로 쓴다**(세종 구문분석 말뭉치는 확보 시 비교용). `CLAUDE.md`·`plan.md` 갱신.

### 11:10–11:22 · K2 변환기·트리뱅크 구축 [gsm, itcerdo]

- 작업: `pcfg/korean.py` — 지배소가 자기 라벨로 투사하며 가까운 의존소부터 묶는 이진 변환, Kiwi 토큰을 시작 위치로 어절에 배정, 말단 `형태/품사`·전종단 `#품사`(Kiwi `NP` 대명사와 구 표지 `NP` 충돌 방지)·괄호 이스케이프, 품사별 UNK `<UNK/NNG>`. `pcfg/induce.py`에 `preprocess_fn`·`unk_fn` 훅(기본값은 영어 동작 그대로). 커밋 `c404952`.

```bash
pytest -q                                              # 58 passed
python -m scripts.build_dp_treebank --limit 20000      # 시험: 98.99% 변환
tmux new-session -d -s pcfg2-k2 "python -m scripts.build_dp_treebank > outputs/k2/build.log 2>&1 < /dev/null"
```

- 결과 (44초): 655,181문장 → **651,934문장 변환(99.50%)**. 제외: Kiwi 형태소 배정 실패 2,632, 비투사 604, 루트 수 오류 11. dev·test 중 train과 같은 문장 제거 2,529 / 3,686.

| 분할 | 문장 | 형태소 | 형태소/문장 |
|---|---:|---:|---:|
| train | 593,990 | 10,223,434 | 17.2 |
| dev | 23,671 | 518,124 | 21.9 |
| test | 28,058 | 560,644 | 20.0 |

| 출처 | 문장 | 변환 |
|---|---:|---:|
| 2020 신문 NXDP1902103231 | 150,082 | 148,027 |
| 2020 구어 SXDP2002103180 | 221,489 | 221,416 |
| 2024 신문 NXDP2402503041 | 67,934 | 67,177 |
| 2024 구어 SXDP2402503041 | 185,330 | 184,984 |
| 2025 신문 NXDP2502512313 | 13,907 | 13,893 |
| 2025 구어 SXDP2502512312 | 16,439 | 16,437 |

→ **K2 완료.**

### 11:22–11:35 · K3 한국어 기준선 [itcerdo]

- 작업: 전체 train으로 유도(품사별 UNK, 빈도 1 이하 UNK) → dev 전체 파싱(C++ 엔진, 12프로세스) → PARSEVAL(루트·전종단 제외). 기능 표지 유지/제거 학습 비교. 먼저 train 10만·dev 500 시험 실행.

```bash
python -m scripts.eval_korean --max-train 100000 --dev-n 500 --workers 8 --tag try100k
tmux new-session -d -s pcfg2-k3 "python -m scripts.eval_korean --dev-n 0 --workers 12 --tag base_full; python -m scripts.eval_korean --dev-n 0 --workers 12 --strip-ftags --tag noft_full"
```

- 결과 (dev 23,671문장, 폴백 1문장):

| 설정 | 비단말 | 이항 규칙 | F1 라벨 | F1 기능표지 제거 | F1 무라벨 | 파싱 시간 |
|---|---:|---:|---:|---:|---:|---:|
| 시험: train 10만, dev 500 | 16,040 | 32,454 | 0.557 | 0.612 | 0.623 | 0.5 s |
| **base_full** (기능 표지 유지) | 50,286 | 97,971 | **0.575** (P .618 R .537) | **0.630** | **0.644** | 25.5 s |
| noft_full (기능 표지 제거 학습) | 41,935 | 79,900 | — | 0.560 | 0.573 | 3.3 s |

  - 문장당 파싱 시간(base_full): 형태소 ≤20 0.9 ms, 21–40 10 ms, 41–60 43 ms, 61–80 109 ms, 80 초과 262 ms(최대 1.3 s).
  - 해석: (1) **기능 표지를 지우고 학습하면 나빠짐**(같은 평가 기준 0.630 → 0.560) → 기능 표지는 주어·목적어·부사어 구분이 상태 주석 역할을 하므로 유지. (2) 재현율이 정밀도보다 8점 낮음 → 파서 입력에 어절 경계가 없어 여러 어절의 형태소를 한 어절 노드로 합치는 경향으로 추정. HNC 후보는 띄어쓰기가 정해져 있으므로 어절 경계를 입력에 주는 실험(K4-1)을 진행.

### 11:35–11:50 · K4-1 어절 경계 표시 [gsm, itcerdo]

- 작업: 재현율 손실이 "파서 입력에 어절 경계가 없어 여러 어절을 한 어절 노드로 합침" 때문이라는 가설 검증. HNC 후보는 띄어쓰기가 정해져 있으므로 경계를 입력에 주는 것은 정당. `mark_eojeol_boundaries`(어절 노드 = 자식이 모두 전종단인 노드의 첫 형태소에 `▁`), 품사별 UNK도 경계 구분(`<UNK▁/NNG>`).
  - 1차(`d83f663`): **말단에만** `▁` → 효과 없음. 어휘 확률만 갈라지고 전종단(`#NNG`)이 같아 문법이 경계를 못 봄(비단말 수 50,286 그대로). 설계 실수.
  - 2차(`48704ca`): **전종단에도** 표시(`#▁NNG`) → 어절 노드 규칙이 어절 첫 형태소로 시작하도록 학습.

```bash
python -m scripts.eval_korean --dev-n 0 --workers 12 --mark-eojeol --tag eojeol_full
```

- 결과 (dev 23,671문장):

| 설정 | F1 라벨 (P / R) | F1 기능표지 제거 | F1 무라벨 | 폴백 | 파싱 시간 |
|---|---:|---:|---:|---:|---:|
| 기준선 base_full | 0.575 (.618 / .537) | 0.630 | 0.644 | 1 | 25.5 s |
| 1차: 말단만 표시 | 0.575 (.618 / .537) | 0.630 | 0.643 | 1 | 25.8 s |
| **2차: 전종단까지 표시** | **0.626** (.619 / .633) | **0.686** | **0.703** | **1,027 (4.3%)** | 4.9 s |

- 해석: 가설 확인 — 재현율 .537 → .633, 라벨 F1 +5.1점, 파싱 5배 빨라짐(경계가 탐색을 제약). 대신 **파스 실패가 4.3%로 늘어남**: 이항화 중간 라벨이 어절 내부 품사열 전체를 기억(h=∞)해, 처음 보는 품사열이 나오면 경계 제약 때문에 다른 분할로 피해 갈 수 없음. HNC에서는 파스 실패를 유한 점수로 처리해야 하므로(K5) 실패율 자체를 낮추는 수평 마르코프화(K4-2)를 이어서 시험.

### 11:50–12:00 · K4-2 수평 마르코프화 [gsm, itcerdo]

- 작업: `binarize(h=)`·`induce_pcfg(markov_h=)`·`eval_korean --markov-h` 추가(중간 라벨에 남은 꼬리 중 앞 h개만 유지, 역변환은 표식만 보므로 무관). 테스트 61 passed(마르코프화 왕복 2개 추가). 커밋 `a0b573d`. 어절 경계 표시와 함께 h=1, 2 비교.

```bash
for h in 1 2; do python -m scripts.eval_korean --dev-n 0 --workers 12 --mark-eojeol --markov-h $h --tag eojeol_h$h; done
```

- 결과 (dev 23,671문장):

| 설정 | 비단말 | 이항 규칙 | F1 라벨 (P / R) | F1 기능표지 제거 | F1 무라벨 | 폴백 | 파싱 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 기준선 (h=∞, 경계 없음) | 50,286 | 97,971 | 0.575 (.618 / .537) | 0.630 | 0.644 | 1 | 25.5 s |
| 경계 표시, h=∞ | 50,342 | 97,971 | 0.626 (.619 / .633) | 0.686 | 0.703 | 1,027 | 4.9 s |
| **경계 표시, h=1** | **1,263** | **17,615** | **0.673** (.672 / .673) | **0.738** | **0.751** | **44 (0.19%)** | 5.4 s |
| 경계 표시, h=2 | 7,492 | 38,352 | 0.665 (.663 / .666) | 0.729 | 0.743 | 234 | 5.1 s |

- 해석: h=1이 최선 — 어절 내부 품사열을 품사 쌍 단위로 일반화해 처음 보는 품사열도 파싱(실패 1,027 → 44), 문법은 1/40로 작아짐. 기준선 대비 라벨 F1 **+9.8점**. 영어 NLTK 표본 기준선(0.47)보다 높으나 데이터 규모(59만 문장 vs 3,500)와 변환 트리의 규칙성(핵어말 이진 투사) 차이가 커서 직접 비교는 하지 않는다.
- **현재 최선 설정 채택: 기능 표지 유지 + 어절 경계 표시(전종단 `#▁품사`) + h=1 + 품사별 UNK.**
