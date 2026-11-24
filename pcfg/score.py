"""K5 — HNC 사전확률 항 log P_PCFG(t*(y)) 채점기.

상위 연구(``phdq_prob``)의 교정 후보 y는 Kiwi 형태소열 ``(형태, 품사, 왼쪽 띄어쓰기)``로 주어진다.
이 모듈은 그 열을 트리뱅크와 같은 말단(어절 첫 형태소 ``▁``, ``형태/품사``)으로 바꿔 C++ CKY로
비터비 파스를 구하고 로그확률을 돌려준다.

- 파스 성공: ``logprob = log P(t*, y)`` (비터비 트리와 문장의 결합 확률), ``parsed=True``.
- 파스 실패(문법이 문장 전체를 덮지 못함): 같은 차트에서 구성소 조각들을 이어 붙이는 최적 분할을
  구해 ``Σ log P(조각) + 조각 수 × frag_logpenalty`` 를 돌려준다(``parsed=False``). 다시 파싱하지
  않으므로 비용이 거의 없고, 후보 재채점에서 점수가 언제나 유한하다(설계서 §4.1의 전 지지 조건과 같은 취지).
  성공 파스가 있으면 조각 점수는 계산하지 않는다(근사: 실패 문장은 dev에서 0.3%).
- 길이 정규화(SLOR 등)는 상위 연구의 λ 결합에서 다루므로 형태소 수를 함께 돌려준다.

학습 설정은 K4까지의 최선(`plan.md`): 기능 표지 유지 + 어절 경계 표시 + 수평 마르코프화 h=1 +
부모 주석 v=2 + 품사·어절경계별 UNK.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import List, Sequence

from .cky_cpp import parse as parse_cpp
from .induce import MARKER, InducedGrammar, induce_pcfg
from .korean import (
    EOJEOL_MARK, mark_eojeol_boundaries, preprocess_parent, terminal, unk_by_tag_eojeol,
)
from .tree import Tree

NEG_INF = float("-inf")
FRAG_LOGPENALTY = math.log(1e-4)   # 파스 실패 시 조각 하나당 벌점
EMPTY_CELL_LOGPROB = math.log(1e-12)  # 어휘 규칙이 전혀 없는 형태소(이론상 없음)의 하한

BEST_CONFIG = {"mark_eojeol": True, "markov_h": 1, "parent": True, "unk_threshold": 1}


@dataclass
class ScoreResult:
    logprob: float
    n_morph: int
    parsed: bool
    n_fragments: int


def induce_best(trees: Sequence[Tree], unk_threshold: int = 1) -> InducedGrammar:
    """K4 최선 설정으로 PCFG 유도. ``trees``는 ``data/treebank``의 변환 트리(표시 전)."""
    marked = [mark_eojeol_boundaries(t) for t in trees]
    return induce_pcfg(marked, unk_threshold=unk_threshold, preprocess_fn=preprocess_parent,
                       unk_fn=unk_by_tag_eojeol, markov_h=BEST_CONFIG["markov_h"])


class PCFGScorer:
    def __init__(self, induced: InducedGrammar, frag_logpenalty: float = FRAG_LOGPENALTY):
        self.ind = induced
        self.frag_logpenalty = frag_logpenalty
        lex = induced.grammar._lexical_by_word
        self._unk_tokens = {w for w in lex if w.startswith("<UNK")}

    # --- 직렬화 -------------------------------------------------------------
    def dumps(self) -> bytes:
        return pickle.dumps({"induced": self.ind, "frag_logpenalty": self.frag_logpenalty,
                             "config": BEST_CONFIG}, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def loads(cls, data: bytes) -> "PCFGScorer":
        d = pickle.loads(data)
        return cls(d["induced"], d["frag_logpenalty"])

    def save(self, path) -> None:
        with open(path, "wb") as f:
            f.write(self.dumps())

    @classmethod
    def load(cls, path) -> "PCFGScorer":
        with open(path, "rb") as f:
            return cls.loads(f.read())

    # --- 입력 변환 ----------------------------------------------------------
    @staticmethod
    def terminals(morphs: Sequence) -> List[str]:
        """``(형태, 품사[, 왼쪽 띄어쓰기])`` 열 → 말단. 첫 형태소와 띄어쓰기 뒤 형태소에 ``▁``."""
        out = []
        for k, m in enumerate(morphs):
            space = bool(m[2]) if len(m) > 2 else False
            mark = EOJEOL_MARK if (k == 0 or space) else ""
            out.append(mark + terminal(m[0], m[1]))
        return out

    def _map(self, term: str) -> str:
        if term in self.ind.vocab:
            return term
        unk = unk_by_tag_eojeol(term)
        if unk in self._unk_tokens:
            return unk
        mark = EOJEOL_MARK if term.startswith(EOJEOL_MARK) else ""
        return f"<UNK{mark}/NNG>"  # 학습에 없던 품사: 가장 큰 개방 부류로 백오프

    # --- 채점 ---------------------------------------------------------------
    def score(self, morphs: Sequence) -> ScoreResult:
        return self.score_terminals(self.terminals(morphs))

    def score_terminals(self, terms: Sequence[str]) -> ScoreResult:
        n = len(terms)
        if n == 0:
            return ScoreResult(NEG_INF, 0, False, 0)
        r = parse_cpp([self._map(t) for t in terms], self.ind.grammar)
        if r.tree is not None:
            return ScoreResult(r.logprob, n, True, 1)
        lp, k = self._fragments(r.chart, n)
        return ScoreResult(lp, n, False, k)

    def _fragments(self, chart, n: int):
        """구성소 조각(이항화 표식 라벨 제외)으로 [0, n)을 덮는 최적 분할."""
        best = [NEG_INF] * (n + 1)
        nfrag = [0] * (n + 1)
        best[0] = 0.0
        for j in range(1, n + 1):
            for i in range(j):
                if best[i] == NEG_INF:
                    continue
                vals = [lp for lab, lp in chart.cell_score(i, j).items() if MARKER not in lab]
                if not vals:
                    if j - i != 1:
                        continue
                    span = EMPTY_CELL_LOGPROB
                else:
                    span = max(vals)
                cand = best[i] + span + self.frag_logpenalty
                if cand > best[j]:
                    best[j], nfrag[j] = cand, nfrag[i] + 1
        return best[n], nfrag[n]
