"""K1 — C++(pybind11) CKY 엔진 래퍼. 참조 구현 `pcfg.cky.parse`의 드롭인 대체.

- `compile_grammar`: PCFG의 RHS 인덱스(`_lexical_by_word`, `_binary_by_rhs`,
  `_unary_by_child`)를 목록 순서·중복 그대로 정수 id 배열로 옮긴다. 규칙 로그확률은
  여기(Python)에서 `math.log(p)`로 계산해 넘기므로 Python 엔진과 같은 double을 쓴다.
  결과는 문법별로 캐시하고 `PCFG._rev`(add() 개정 번호)가 바뀌면 다시 컴파일한다.
  (인덱스 사전을 add()를 거치지 않고 직접 고치면 캐시가 낡을 수 있다.)
- `parse`: `pcfg.cky.parse`와 같은 `ParseResult`를 돌려준다(같은 트리, 비트 단위로 같은
  logprob). `prob`은 Python에서 `math.exp(logprob)`로 계산한다.
- 비단말 라벨과 단어는 별개의 이름공간이다.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from typing import Dict, List, Tuple

from . import _ckycpp
from .cky import NEG_INF, BackPtr, ParseResult
from .grammar import PCFG
from .tree import Tree


@dataclass
class CompiledGrammar:
    rev: int
    labels: List[str]          # 라벨 id → 라벨
    label_id: Dict[str, int]   # 라벨 → 라벨 id
    word_id: Dict[str, int]    # 단어 → 단어 id (어휘 규칙이 있는 단어만)
    native: "_ckycpp.Grammar"


_CACHE: "weakref.WeakKeyDictionary[PCFG, CompiledGrammar]" = weakref.WeakKeyDictionary()


def compile_grammar(grammar: PCFG) -> CompiledGrammar:
    cached = _CACHE.get(grammar)
    if cached is not None and cached.rev == grammar._rev:
        return cached

    labels: List[str] = []
    label_id: Dict[str, int] = {}

    def lid(label: str) -> int:
        i = label_id.get(label)
        if i is None:
            i = label_id[label] = len(labels)
            labels.append(label)
        return i

    lid(grammar.start)

    # 어휘: 단어 → [(A, lp)]  (pcfg.cky: `for A, p in grammar.lexical_parents(w)`)
    word_id: Dict[str, int] = {}
    lex_off, lex_lhs, lex_lp = [0], [], []
    for word, rules in grammar._lexical_by_word.items():
        word_id[word] = len(word_id)
        for a, p in rules:
            lex_lhs.append(lid(a))
            lex_lp.append(math.log(p))
        lex_off.append(len(lex_lhs))

    # 이항: (B, C) → [(A, lp)]  (`grammar.binary_parents(B, C)`)
    bin_b, bin_c, bin_off, bin_lhs, bin_lp = [], [], [0], [], []
    for (b, c), rules in grammar._binary_by_rhs.items():
        if not rules:
            continue
        bin_b.append(lid(b))
        bin_c.append(lid(c))
        for a, p in rules:
            bin_lhs.append(lid(a))
            bin_lp.append(math.log(p))
        bin_off.append(len(bin_lhs))

    # 단항: B → [(A, lp)]  (`grammar.unary_parents(B)`) — 라벨 id 전체에 대한 CSR
    unary_rows: Dict[int, List[Tuple[int, float]]] = {}
    for b, rules in grammar._unary_by_child.items():
        if rules:
            unary_rows[lid(b)] = [(lid(a), math.log(p)) for a, p in rules]
    un_off, un_lhs, un_lp = [0], [], []
    for b in range(len(labels)):
        for a, lp in unary_rows.get(b, ()):
            un_lhs.append(a)
            un_lp.append(lp)
        un_off.append(len(un_lhs))

    native = _ckycpp.Grammar(
        len(labels), len(word_id),
        lex_off, lex_lhs, lex_lp,
        un_off, un_lhs, un_lp,
        bin_b, bin_c, bin_off, bin_lhs, bin_lp,
    )
    compiled = CompiledGrammar(rev=grammar._rev, labels=labels, label_id=label_id,
                               word_id=word_id, native=native)
    _CACHE[grammar] = compiled
    return compiled


class CppChart:
    """C++ 차트 뷰. `pcfg.cky.Chart`의 조회 API(`n`, `cell_logprob`, `cell_prob`)를 제공."""

    def __init__(self, n: int, native: "_ckycpp.Chart", compiled: CompiledGrammar,
                 words: List[str]):
        self.n = n
        self._native = native
        self._compiled = compiled
        self._words = list(words)

    def cell_logprob(self, i: int, j: int, label: str) -> float:
        a = self._compiled.label_id.get(label)
        if a is None:
            return NEG_INF
        return self._native.logprob(i, j, a)

    def cell_prob(self, i: int, j: int, label: str) -> float:
        """구간 [i, j)에서 비단말 label의 Viterbi 확률(선형). 없으면 0."""
        lp = self.cell_logprob(i, j, label)
        return 0.0 if lp == NEG_INF else math.exp(lp)

    def cell_score(self, i: int, j: int) -> Dict[str, float]:
        """셀 [i, j)의 label → logprob (삽입 순서 = Python 엔진 `chart.score[(i, j)]` 순서)."""
        labels = self._compiled.labels
        return {labels[a]: lp for a, lp, _, _, _ in self._native.entries(i, j)}

    def cell_back(self, i: int, j: int) -> Dict[str, BackPtr]:
        """셀 [i, j)의 백포인터를 Python 엔진 형식(`chart.back[(i, j)]`)으로."""
        labels = self._compiled.labels
        out: Dict[str, BackPtr] = {}
        for a, _, k, b, c in self._native.entries(i, j):
            if c >= 0:
                out[labels[a]] = ("bin", k, labels[b], labels[c])
            elif b >= 0:
                out[labels[a]] = ("unary", labels[b])
            else:
                out[labels[a]] = ("lex", self._words[i])
        return out


def parse(words: List[str], grammar: PCFG) -> ParseResult:
    compiled = compile_grammar(grammar)
    word_id = compiled.word_id
    native_chart = _ckycpp.parse(compiled.native, [word_id.get(w, -1) for w in words])
    n = len(words)
    chart = CppChart(n, native_chart, compiled, words)

    start = compiled.label_id[grammar.start]
    top_lp = native_chart.logprob(0, n, start)
    if top_lp == NEG_INF:
        return ParseResult(tree=None, prob=0.0, logprob=NEG_INF, chart=chart)

    tree = _decode_tree(native_chart.tree(start), words, compiled.labels)
    return ParseResult(tree=tree, prob=math.exp(top_lp), logprob=top_lp, chart=chart)


def _decode_tree(code: List[int], words: List[str], labels: List[str]) -> Tree:
    """C++ 전위 부호(label, kind[, i])를 Tree로 (pcfg.cky `_build_tree`와 같은 모양)."""
    it = iter(code)

    def rec() -> Tree:
        label = labels[next(it)]
        kind = next(it)
        if kind == 0:  # 어휘: 리프는 그 위치의 입력 단어
            return Tree(label, [words[next(it)]])
        if kind == 1:  # 단항
            return Tree(label, [rec()])
        left = rec()   # 이항
        right = rec()
        return Tree(label, [left, right])

    return rec()
