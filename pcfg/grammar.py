"""PCFG 문법 표현: 규칙 + 확률, LHS/RHS 양방향 인덱싱.

규약 (CLAUDE.md):
- 문법은 LHS·RHS 양방향 인덱싱을 유지한다 (CKY 속도에 결정적).
- CKY/Viterbi는 로그 공간에서 계산한다 → logprob 캐시를 제공.

규칙 종류:
- 이항(binary):   A -> B C   (B, C 는 비단말)
- 단항(unary):    A -> B     (B 는 비단말) — Phase 3 CNF 변환에서 등장
- 어휘(lexical):  A -> w      (w 는 종단 단어)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple


class PCFG:
    def __init__(self, start: str = "S"):
        self.start = start

        # 규칙 확률: (lhs, rhs_tuple) -> prob
        self.prob: Dict[Tuple[str, Tuple[str, ...]], float] = {}

        # RHS 인덱스 (CKY 조합에 사용)
        self._binary_by_rhs: Dict[Tuple[str, str], List[Tuple[str, float]]] = defaultdict(list)
        self._unary_by_child: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self._lexical_by_word: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        # LHS 인덱스 (트리 확률·정규화 검증에 사용)
        self._by_lhs: Dict[str, List[Tuple[Tuple[str, ...], float]]] = defaultdict(list)

        self.nonterminals: set[str] = set()

        # 개정 번호: add() 때마다 증가. C++ 엔진(cky_cpp)의 컴파일 캐시 무효화 키.
        self._rev = 0

    # ------------------------------------------------------------------
    # 규칙 추가
    # ------------------------------------------------------------------
    def add(self, lhs: str, rhs, prob: float, lexical: bool = False) -> None:
        self._rev += 1
        rhs = tuple(rhs)
        key = (lhs, rhs)
        self.prob[key] = prob
        self._by_lhs[lhs].append((rhs, prob))
        self.nonterminals.add(lhs)

        if lexical:
            assert len(rhs) == 1, "lexical rule RHS must be a single word"
            self._lexical_by_word[rhs[0]].append((lhs, prob))
        elif len(rhs) == 2:
            self._binary_by_rhs[(rhs[0], rhs[1])].append((lhs, prob))
        elif len(rhs) == 1:
            self._unary_by_child[rhs[0]].append((lhs, prob))
        # 길이 >= 3 (비-CNF)인 규칙은 확률/LHS 인덱스에만 보관한다.
        # CKY는 CNF만 다루므로 RHS 인덱스에는 넣지 않는다(트리 확률 계산엔 무관).

    def add_binary(self, lhs: str, b: str, c: str, prob: float) -> None:
        self.add(lhs, (b, c), prob)

    def add_unary(self, lhs: str, child: str, prob: float) -> None:
        self.add(lhs, (child,), prob)

    def add_lexical(self, lhs: str, word: str, prob: float) -> None:
        self.add(lhs, (word,), prob, lexical=True)

    # ------------------------------------------------------------------
    # 조회
    # ------------------------------------------------------------------
    def binary_parents(self, b: str, c: str) -> List[Tuple[str, float]]:
        """RHS (B, C)를 갖는 규칙들의 (A, prob) 목록."""
        return self._binary_by_rhs.get((b, c), [])

    def unary_parents(self, child: str) -> List[Tuple[str, float]]:
        return self._unary_by_child.get(child, [])

    def lexical_parents(self, word: str) -> List[Tuple[str, float]]:
        return self._lexical_by_word.get(word, [])

    def rules_for(self, lhs: str) -> List[Tuple[Tuple[str, ...], float]]:
        return self._by_lhs.get(lhs, [])

    def prob_of(self, lhs: str, rhs) -> float:
        return self.prob[(lhs, tuple(rhs))]

    def logprob_of(self, lhs: str, rhs) -> float:
        return math.log(self.prob[(lhs, tuple(rhs))])

    # ------------------------------------------------------------------
    # 검증
    # ------------------------------------------------------------------
    def check_normalized(self, tol: float = 1e-9) -> List[str]:
        """LHS별 확률 합이 1인지 검사. 위반한 LHS 목록을 반환(비면 정상)."""
        bad: List[str] = []
        for lhs, rules in self._by_lhs.items():
            total = sum(p for _, p in rules)
            if abs(total - 1.0) > tol:
                bad.append(f"{lhs}: sum={total}")
        return bad
