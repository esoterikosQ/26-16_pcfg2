"""Phase 2 검증 앵커: 확률적 CKY 셀 값 (§3.3) + 최적 파스·차트 셀 (§3.4)."""

import pytest

from pcfg.cky import parse
from pcfg.tree_prob import tree_prob
from tests.fixtures import (
    jurafsky_e4_grammar,
    manning_grammar,
    manning_tree_t1,
    manning_tree_t2,
    MANNING_SENTENCE,
)


# ---------------------------------------------------------------------------
# §3.3 Jurafsky Fig E.4 개별 셀 값 (셀 단위 검증만; unary 없어 부분 파싱)
# ---------------------------------------------------------------------------
def test_e4_np_cell_value():
    g = jurafsky_e4_grammar()
    words = ["the", "flight", "includes", "a", "meal"]
    r = parse(words, g)
    # NP(0,2) = P(NP→Det N)·P(Det→the)·P(N→flight) = .30·.40·.02 = .0024
    assert r.chart.cell_prob(0, 2, "NP") == pytest.approx(0.30 * 0.40 * 0.02, rel=1e-12)
    assert r.chart.cell_prob(0, 2, "NP") == pytest.approx(0.0024, rel=1e-9)


def test_e4_lexical_cells():
    g = jurafsky_e4_grammar()
    words = ["the", "flight", "includes", "a", "meal"]
    r = parse(words, g)
    assert r.chart.cell_prob(0, 1, "Det") == pytest.approx(0.40)
    assert r.chart.cell_prob(1, 2, "N") == pytest.approx(0.02)
    assert r.chart.cell_prob(2, 3, "V") == pytest.approx(0.05)
    assert r.chart.cell_prob(4, 5, "N") == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# §3.4 Manning Table 11.2 — 차트 중간 셀 + 최적 파스 복원
# fencepost 매핑: Manning 위치 i..j(1-indexed 포함) -> [i-1, j)
# ---------------------------------------------------------------------------
def test_manning_intermediate_cells():
    g = manning_grammar()
    r = parse(MANNING_SENTENCE, g)
    c = r.chart
    # VP(2,3)=0.126 -> [1,3)
    assert c.cell_prob(1, 3, "VP") == pytest.approx(0.126, rel=1e-9)
    # PP(4,5)=0.18 -> [3,5)
    assert c.cell_prob(3, 5, "PP") == pytest.approx(0.18, rel=1e-9)
    # NP(3,5)=0.01296 -> [2,5)
    assert c.cell_prob(2, 5, "NP") == pytest.approx(0.01296, rel=1e-9)
    # S(1,3)=0.0126 -> [0,3)
    assert c.cell_prob(0, 3, "S") == pytest.approx(0.0126, rel=1e-9)


def test_manning_competition_cell_vp_2_5():
    # VP(2,5) -> [1,5): 두 유도 경쟁, Viterbi(max)=0.009072 (V NP 경로)
    g = manning_grammar()
    r = parse(MANNING_SENTENCE, g)
    assert r.chart.cell_prob(1, 5, "VP") == pytest.approx(0.009072, rel=1e-9)


def test_manning_top_cell_s_1_5():
    # S(1,5) -> [0,5): 0.0009072, 최적 파스 t1의 확률과 일치
    g = manning_grammar()
    r = parse(MANNING_SENTENCE, g)
    assert r.chart.cell_prob(0, 5, "S") == pytest.approx(0.0009072, rel=1e-9)


def test_manning_optimal_parse_is_t1():
    g = manning_grammar()
    r = parse(MANNING_SENTENCE, g)
    assert r.tree == manning_tree_t1()
    assert r.tree != manning_tree_t2()
    assert r.prob == pytest.approx(0.0009072, rel=1e-9)
    # 복원된 트리의 확률 == 차트 top 확률 (일관성)
    assert tree_prob(r.tree, g) == pytest.approx(r.prob, rel=1e-9)


def test_manning_no_parse_returns_none():
    g = manning_grammar()
    r = parse(["astronomers", "flabbergasted"], g)
    assert r.tree is None
    assert r.prob == 0.0
