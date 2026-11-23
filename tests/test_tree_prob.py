"""Phase 1 검증 앵커: 트리 확률 (§3.1 Jurafsky, §3.2 Manning)."""

import math

import pytest

from pcfg.tree_prob import tree_prob
from tests.fixtures import (
    jurafsky_e1_grammar,
    jurafsky_e1_tree_left,
    jurafsky_e1_tree_right,
    manning_grammar,
    manning_tree_t1,
    manning_tree_t2,
)


# --- §3.1 Jurafsky Fig E.1 ---
def test_jurafsky_left_tree_prob():
    g = jurafsky_e1_grammar()
    p = tree_prob(jurafsky_e1_tree_left(), g)
    assert p == pytest.approx(2.2e-6, rel=2e-2)
    # 정확 곱: .05·.20·.20·.20·.75·.30·.60·.10·.40
    assert p == pytest.approx(0.05*0.20*0.20*0.20*0.75*0.30*0.60*0.10*0.40, rel=1e-12)


def test_jurafsky_right_tree_prob():
    # erratum 주의(§ fixtures): Fig E.2는 VP→Verb NP NP를 .10으로 곱해 6.1e-7을
    # 제시하나, Fig E.1 문법의 정규화 값은 .05다. 정규화된 충실한 문법을 채택하므로
    # 골든 넘버를 6.1e-7 대신 .05 기반 3.0375e-7로 교정한다.
    g = jurafsky_e1_grammar()
    p = tree_prob(jurafsky_e1_tree_right(), g)
    assert p == pytest.approx(0.05*0.05*0.20*0.15*0.75*0.75*0.30*0.60*0.10*0.40, rel=1e-12)
    assert p == pytest.approx(3.0375e-7, rel=1e-9)


def test_jurafsky_left_more_probable():
    g = jurafsky_e1_grammar()
    assert tree_prob(jurafsky_e1_tree_left(), g) > tree_prob(jurafsky_e1_tree_right(), g)


# --- §3.2 Manning Table 11.2 ---
def test_manning_t1_prob():
    g = manning_grammar()
    assert tree_prob(manning_tree_t1(), g) == pytest.approx(0.0009072, rel=1e-9)


def test_manning_t2_prob():
    g = manning_grammar()
    assert tree_prob(manning_tree_t2(), g) == pytest.approx(0.0006804, rel=1e-9)


def test_manning_sentence_prob_cross_check():
    # 문장 확률 = 파스 확률의 합 (§3.2 교차검증).
    g = manning_grammar()
    total = tree_prob(manning_tree_t1(), g) + tree_prob(manning_tree_t2(), g)
    assert total == pytest.approx(0.0015876, rel=1e-9)


def test_grammar_normalization():
    # 문헌 문법은 LHS별 확률 합이 1이어야 한다.
    assert manning_grammar().check_normalized() == []
    assert jurafsky_e1_grammar().check_normalized() == []
