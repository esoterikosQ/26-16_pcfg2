"""Phase 0 검증: 브래킷 왕복 + 인덱스 매핑 헬퍼."""

from pcfg.tree import (
    Tree,
    from_bracket,
    to_bracket,
    manning_span_to_fencepost,
    fencepost_span_to_manning,
    word_position_to_fencepost,
)
from tests.fixtures import manning_tree_t1, jurafsky_e1_tree_left


def test_bracket_round_trip_simple():
    s = "(S (NP (DT the) (NN flight)) (VP (VB includes)))"
    tree = from_bracket(s)
    assert to_bracket(tree) == s


def test_bracket_round_trip_fixture_trees():
    for tree in (manning_tree_t1(), jurafsky_e1_tree_left()):
        assert from_bracket(to_bracket(tree)) == tree


def test_top_wrapper_unlabeled():
    # Penn Treebank는 라벨 없는 최상단 래퍼를 쓴다.
    s = "( (S (NP (PRP I)) (VP (VBP run))) )"
    tree = from_bracket(s)
    assert tree.label == "S"
    assert tree.leaves() == ["I", "run"]


def test_leaves_order():
    tree = manning_tree_t1()
    assert tree.leaves() == ["astronomers", "saw", "stars", "with", "ears"]


def test_manning_to_fencepost():
    # 전체 문장 위치 1..5 -> [0, 5)
    assert manning_span_to_fencepost(1, 5) == (0, 5)
    # 단일 단어 위치 3..3 -> [2, 3)
    assert manning_span_to_fencepost(3, 3) == (2, 3)


def test_fencepost_to_manning_inverse():
    for i, j in [(1, 5), (2, 3), (3, 3), (4, 5)]:
        start, end = manning_span_to_fencepost(i, j)
        assert fencepost_span_to_manning(start, end) == (i, j)


def test_word_position_to_fencepost():
    assert word_position_to_fencepost(1) == (0, 1)
    assert word_position_to_fencepost(5) == (4, 5)
