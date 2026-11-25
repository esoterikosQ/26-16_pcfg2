"""K5 HNC 채점기 검증 (예문은 직접 작성, 파일 없이)."""

import math

from pcfg.korean import dp_to_tree
from pcfg.score import PCFGScorer, induce_best

M_SUBJ = [("나", "NP"), ("는", "JX")]
M_PRED = [("먹", "VV"), ("었", "EP"), ("다", "EF")]


def _scorer():
    # 주어 조사를 달리한 트리를 하나 섞어 문법이 결정적(log P = 0)이 되지 않게 한다(v=3 주석에서도).
    trees = [dp_to_tree([3, 3, 0], ["NP_SBJ", "NP_OBJ", "VP"],
                        [M_SUBJ, [(n, "NNG"), ("을", "JKO")], M_PRED]) for n in ("사과", "빵", "밥")]
    trees.append(dp_to_tree([3, 3, 0], ["NP_SBJ", "NP_OBJ", "VP"],
                            [[("내", "NP"), ("가", "JKS")], [("국", "NNG"), ("을", "JKO")], M_PRED]))
    return PCFGScorer(induce_best(trees))


def _morphs(noun):
    # (형태, 품사, 왼쪽 띄어쓰기) — HNC 형태소열 형식
    return [("나", "NP", False), ("는", "JX", False), (noun, "NNG", True), ("을", "JKO", False),
            ("먹", "VV", True), ("었", "EP", False), ("다", "EF", False)]


def test_terminals_mark_eojeol_starts():
    assert PCFGScorer.terminals(_morphs("떡")) == [
        "▁나/NP", "는/JX", "▁떡/NNG", "을/JKO", "▁먹/VV", "었/EP", "다/EF"]


def test_grammatical_sentence_parses_with_finite_score():
    sc = _scorer()
    r = sc.score(_morphs("떡"))  # 떡: 미등록 → <UNK▁/NNG>
    assert r.parsed and r.n_fragments == 1 and r.n_morph == 7
    assert math.isfinite(r.logprob) and r.logprob < 0


def test_unparseable_order_gets_finite_fragment_score():
    sc = _scorer()
    m = _morphs("떡")
    swapped = [("먹", "VV", False), ("었", "EP", False), ("다", "EF", False),
               ("나", "NP", True), ("는", "JX", False), ("떡", "NNG", True), ("을", "JKO", False)]
    good, bad = sc.score(m), sc.score(swapped)
    assert not bad.parsed and bad.n_fragments >= 2 and math.isfinite(bad.logprob)
    assert good.logprob > bad.logprob


def test_unseen_tag_backs_off_to_noun_unk():
    sc = _scorer()
    assert sc._map("▁무엇/XYZ") == "<UNK▁/NNG>"


def test_serialization_round_trip():
    sc = _scorer()
    sc2 = PCFGScorer.loads(sc.dumps())
    assert sc2.score(_morphs("떡")).logprob == sc.score(_morphs("떡")).logprob
