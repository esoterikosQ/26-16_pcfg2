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
