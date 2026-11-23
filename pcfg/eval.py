"""Phase 3 — PARSEVAL 평가 (라벨 정밀도/재현율/F1).

구성소(constituent) = (라벨, i, j): 반열림 구간 [i, j)를 지배하는 비단말 노드.
전종단(POS 태그)과 리프는 제외한다(표준 PARSEVAL). 두 트리의 구성소
다중집합(multiset)을 비교해 교집합 크기로 정밀도/재현율/F1을 계산한다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

from .tree import Tree


def constituents(tree: Tree, include_root: bool = True) -> Counter:
    """트리의 (라벨, i, j) 구성소 다중집합. 전종단·리프 제외."""
    result: Counter = Counter()

    def rec(node: Tree, start: int, is_root: bool) -> int:
        if node.is_preterminal():
            return start + 1  # POS 태그는 구성소로 세지 않음
        cur = start
        for c in node.children:
            if isinstance(c, str):
                cur += 1
            else:
                cur = rec(c, cur, False)
        end = cur
        if include_root or not is_root:
            result[(node.label, start, end)] += 1
        return end

    rec(tree, 0, True)
    return result


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    matched: int
    gold: int
    test: int


def _prf(matched: int, gold: int, test: int) -> PRF:
    precision = matched / test if test else 0.0
    recall = matched / gold if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return PRF(precision, recall, f1, matched, gold, test)


def evaluate_pair(gold: Tree, test: Tree, include_root: bool = False) -> PRF:
    g = constituents(gold, include_root=include_root)
    t = constituents(test, include_root=include_root)
    matched = sum((g & t).values())  # 다중집합 교집합
    return _prf(matched, sum(g.values()), sum(t.values()))


def evaluate_corpus(pairs: List[Tuple[Tree, Tree]],
                    include_root: bool = False) -> PRF:
    """(gold, test) 쌍들에 대한 코퍼스 수준 마이크로 평균 PARSEVAL."""
    tot_matched = tot_gold = tot_test = 0
    for gold, test in pairs:
        g = constituents(gold, include_root=include_root)
        t = constituents(test, include_root=include_root)
        tot_matched += sum((g & t).values())
        tot_gold += sum(g.values())
        tot_test += sum(t.values())
    return _prf(tot_matched, tot_gold, tot_test)
