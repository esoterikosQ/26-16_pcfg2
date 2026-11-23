"""Phase 1 — 트리 확률.

트리의 모든 국소 규칙 확률을 곱해 P(T)를 계산한다 (Jurafsky Eq. E.2,
Manning: "국소 하위트리를 만든 규칙들의 확률을 곱한다").

언더플로우 방지를 위해 로그 공간 합산 후 되돌린다(옵션). 문헌 예제는
값이 크지 않아 선형 곱도 안전하지만, 규약(로그 공간)을 따른다.
"""

from __future__ import annotations

import math
from typing import Union

from .grammar import PCFG
from .tree import Tree


def _rule_of(node: Tree) -> tuple[str, tuple[str, ...]]:
    """노드가 나타내는 국소 규칙 (LHS, RHS)."""
    if node.is_preterminal():
        return node.label, (node.children[0],)  # 어휘 규칙 A -> w
    rhs = tuple(c.label for c in node.children)  # 내부 규칙 A -> B C ...
    return node.label, rhs


def log_tree_prob(tree: Tree, grammar: PCFG) -> float:
    """log P(T). 트리의 각 확장 규칙 logprob 합."""
    total = 0.0
    stack: list[Union[Tree, str]] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            continue
        lhs, rhs = _rule_of(node)
        total += math.log(grammar.prob_of(lhs, rhs))
        if not node.is_preterminal():
            stack.extend(node.children)
    return total


def tree_prob(tree: Tree, grammar: PCFG) -> float:
    """P(T) = exp(log P(T))."""
    return math.exp(log_tree_prob(tree, grammar))
