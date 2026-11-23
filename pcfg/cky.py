"""Phase 2 — 확률적 CKY + Viterbi 최적 파스 복원.

Jurafsky Fig E.3 의사코드 + Manning 11.3.3 (Viterbi δ/ψ 누적자).
- CNF 가정: 이항 규칙 A→B C 와 어휘 규칙 A→w 만 처리(단항은 Phase 3에서).
- 셀 값 갱신에 max + 백포인터. 로그 공간(단순 로그 합)으로 언더플로우 방지.
- 인덱싱: fencepost 0..n, 반열림 구간 [i, j) (코드 전역 표준).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from .grammar import PCFG
from .tree import Tree

NEG_INF = float("-inf")

# 백포인터: 어휘 ('lex', word), 이항 ('bin', k, B, C), 단항 ('unary', B)
BackPtr = Union[Tuple[str, str], Tuple[str, int, str, str], Tuple[str, str]]


@dataclass
class Chart:
    n: int
    # score[(i, j)][A] = 최적 log 확률 (반열림 구간 [i, j), 비단말 A)
    score: Dict[Tuple[int, int], Dict[str, float]]
    back: Dict[Tuple[int, int], Dict[str, BackPtr]]

    def cell_logprob(self, i: int, j: int, label: str) -> float:
        return self.score.get((i, j), {}).get(label, NEG_INF)

    def cell_prob(self, i: int, j: int, label: str) -> float:
        """구간 [i, j)에서 비단말 label의 Viterbi 확률(선형). 없으면 0."""
        lp = self.cell_logprob(i, j, label)
        return 0.0 if lp == NEG_INF else math.exp(lp)


@dataclass
class ParseResult:
    tree: Optional[Tree]
    prob: float
    logprob: float
    chart: Chart


def _apply_unary(cell: Dict[str, float], cell_back: Dict[str, BackPtr], grammar: PCFG) -> None:
    """단항 클로저: A→B 규칙으로 셀 내 값을 max 전파 (일반화 CKY).

    로그 확률은 ≤0이라 단항 적용은 값을 키우지 못하므로(체인이 이득을 줄 수 없음)
    고정점에 수렴한다. 안전하게 변화가 없을 때까지 반복한다.
    """
    changed = True
    while changed:
        changed = False
        for B, lp_b in list(cell.items()):
            for A, p in grammar.unary_parents(B):
                cand = math.log(p) + lp_b
                if cand > cell.get(A, NEG_INF):
                    cell[A] = cand
                    cell_back[A] = ("unary", B)
                    changed = True


def parse(words: List[str], grammar: PCFG) -> ParseResult:
    n = len(words)
    score: Dict[Tuple[int, int], Dict[str, float]] = {}
    back: Dict[Tuple[int, int], Dict[str, BackPtr]] = {}

    def ensure(i: int, j: int):
        if (i, j) not in score:
            score[(i, j)] = {}
            back[(i, j)] = {}

    # --- 어휘 규칙: 폭 1 셀 [i, i+1) 채우기 (Fig E.3의 첫 루프) ---
    for i in range(n):
        ensure(i, i + 1)
        for A, p in grammar.lexical_parents(words[i]):
            lp = math.log(p)
            cell = score[(i, i + 1)]
            if lp > cell.get(A, NEG_INF):
                cell[A] = lp
                back[(i, i + 1)][A] = ("lex", words[i])
        _apply_unary(score[(i, i + 1)], back[(i, i + 1)], grammar)

    # --- 이항 규칙: 폭 2..n 셀 채우기 (Fig E.3의 삼중 루프) ---
    for width in range(2, n + 1):
        for i in range(0, n - width + 1):
            j = i + width
            ensure(i, j)
            cell = score[(i, j)]
            cell_back = back[(i, j)]
            for k in range(i + 1, j):
                left = score.get((i, k), {})
                right = score.get((k, j), {})
                if not left or not right:
                    continue
                for B, lp_b in left.items():
                    for C, lp_c in right.items():
                        for A, p in grammar.binary_parents(B, C):
                            cand = math.log(p) + lp_b + lp_c
                            if cand > cell.get(A, NEG_INF):
                                cell[A] = cand
                                cell_back[A] = ("bin", k, B, C)
            # 단항 규칙 처리 (일반화 CKY): 이항 조합 뒤 단항 클로저 적용.
            _apply_unary(cell, cell_back, grammar)

    chart = Chart(n=n, score=score, back=back)

    start = grammar.start
    top_lp = chart.cell_logprob(0, n, start)
    if top_lp == NEG_INF:
        return ParseResult(tree=None, prob=0.0, logprob=NEG_INF, chart=chart)

    tree = _build_tree(0, n, start, back)
    return ParseResult(tree=tree, prob=math.exp(top_lp), logprob=top_lp, chart=chart)


def _build_tree(i: int, j: int, label: str, back) -> Tree:
    """백포인터를 따라 최적 파스를 복원 (Fig E.3의 BUILD_TREE)."""
    bp = back[(i, j)][label]
    if bp[0] == "lex":
        _, word = bp
        return Tree(label, [word])
    if bp[0] == "unary":
        _, B = bp
        return Tree(label, [_build_tree(i, j, B, back)])
    _, k, B, C = bp
    return Tree(label, [
        _build_tree(i, k, B, back),
        _build_tree(k, j, C, back),
    ])
