"""K2 한국어 트리뱅크 변환·유도 검증 (예문은 직접 작성)."""

from collections import namedtuple

import pytest

from pcfg.induce import induce_pcfg, parse_sentence
from pcfg.korean import (
    assign_morphemes, dp_to_tree, escape_form, is_projective, morpheme_terminals, preprocess_keep,
    preprocess_strip, single_root, split_of, strip_function_tag, terminal, unk_by_tag,
)
from pcfg.tree import from_bracket, to_bracket

M_SUBJ = [("나", "NP"), ("는", "JX")]
M_OBJ = [("사과", "NNG"), ("를", "JKO")]
M_PRED = [("먹", "VV"), ("었", "EP"), ("다", "EF")]


def test_dp_to_tree_head_final_binary_projection():
    t = dp_to_tree([3, 3, 0], ["NP_SBJ", "NP_OBJ", "VP"], [M_SUBJ, M_OBJ, M_PRED])
    assert to_bracket(t) == (
        "(VP (NP_SBJ (#NP 나/NP) (#JX 는/JX)) "
        "(VP (NP_OBJ (#NNG 사과/NNG) (#JKO 를/JKO)) (VP (#VV 먹/VV) (#EP 었/EP) (#EF 다/EF))))"
    )
    assert t.leaves() == morpheme_terminals([M_SUBJ, M_OBJ, M_PRED])


def test_dp_to_tree_nested_modifier():
    # 맛있는 사과를 먹었다: 맛있는 → 사과를 → 먹었다
    mod = [("맛있", "VA"), ("는", "ETM")]
    t = dp_to_tree([2, 3, 0], ["VP_MOD", "NP_OBJ", "VP"], [mod, M_OBJ, M_PRED])
    assert to_bracket(t).startswith("(VP (NP_OBJ (VP_MOD (#VA 맛있/VA) (#ETM 는/ETM)) (NP_OBJ (#NNG")
    assert t.leaves() == morpheme_terminals([mod, M_OBJ, M_PRED])


def test_pronoun_tag_does_not_collide_with_phrase_label():
    t = dp_to_tree([2, 0], ["NP_SBJ", "VP"], [M_SUBJ, [("가", "VV"), ("ㄴ다", "EF")]])
    labels = set()

    def rec(n):
        labels.add(n.label)
        for c in n.children:
            if not isinstance(c, str):
                rec(c)
    rec(t)
    assert "#NP" in labels and "NP" not in labels


def test_projectivity_and_root():
    assert is_projective([3, 3, 0])
    assert is_projective([2, 3, 0, 3]) is True
    assert not is_projective([3, 4, 0, 3])     # 호 (1,3)과 (2,4)가 교차
    assert single_root([3, 3, 0]) and not single_root([0, 3, 0])


Tok = namedtuple("Tok", "form tag start len")


def test_assign_morphemes():
    words = [(0, 2), (3, 6)]                    # "나는 사과를"
    toks = [Tok("나", "NP", 0, 1), Tok("는", "JX", 1, 1), Tok("사과", "NNG", 3, 2), Tok("를", "JKO", 5, 1)]
    assert assign_morphemes(words, toks) == [M_SUBJ, M_OBJ]
    # 어절 경계를 넘는 토큰 → 실패
    assert assign_morphemes(words, [Tok("나는사", "NNG", 0, 4)]) is None
    # 형태소가 없는 어절 → 실패
    assert assign_morphemes(words, toks[:2]) is None


def test_escape_and_bracket_round_trip():
    assert terminal("(", "SSO") == "-LRB-/SSO" and escape_form("a b") == "a_b"
    t = dp_to_tree([2, 0], ["NP", "VP"], [[("(", "SSO"), ("사과", "NNG"), (")", "SSC")], M_PRED])
    assert from_bracket(to_bracket(t)) == t


def test_labels_unk_and_split():
    assert unk_by_tag("사과/NNG") == "<UNK/NNG>"
    assert unk_by_tag("-LRB-/SSO") == "<UNK/SSO>"
    assert strip_function_tag("NP_SBJ") == "NP" and strip_function_tag("#NP") == "#NP"
    assert strip_function_tag("VP|<#EP-#EF>") == "VP|<#EP-#EF>"
    t = dp_to_tree([2, 0], ["NP_SBJ", "VP"], [M_SUBJ, M_PRED])
    assert "NP_SBJ" not in to_bracket(preprocess_strip(t)) and preprocess_keep(t) == t
    assert split_of("DOC1") == split_of("DOC1")


@pytest.mark.parametrize("engine", ["py", "cpp"])
def test_induce_and_parse_with_tag_unk(engine):
    nouns = ["사과", "빵", "밥"]
    trees = [dp_to_tree([3, 3, 0], ["NP_SBJ", "NP_OBJ", "VP"],
                        [M_SUBJ, [(n, "NNG"), ("을", "JKO")], M_PRED]) for n in nouns]
    ind = induce_pcfg(trees, unk_threshold=1, preprocess_fn=preprocess_keep, unk_fn=unk_by_tag)
    assert "<UNK/NNG>" in ind.grammar._lexical_by_word
    gold = dp_to_tree([3, 3, 0], ["NP_SBJ", "NP_OBJ", "VP"], [M_SUBJ, [("떡", "NNG"), ("을", "JKO")], M_PRED])
    tree, fallback = parse_sentence(gold.leaves(), ind, engine=engine)
    assert not fallback
    assert tree == gold


def test_mark_eojeol_boundaries():
    from pcfg.korean import mark_eojeol_boundaries, unk_by_tag_eojeol
    t = dp_to_tree([3, 3, 0], ["NP_SBJ", "NP_OBJ", "VP"], [M_SUBJ, M_OBJ, M_PRED])
    m = mark_eojeol_boundaries(t)
    assert m.leaves() == ["▁나/NP", "는/JX", "▁사과/NNG", "를/JKO", "▁먹/VV", "었/EP", "다/EF"]
    assert unk_by_tag_eojeol("▁사과/NNG") == "<UNK▁/NNG>" and unk_by_tag_eojeol("를/JKO") == "<UNK/JKO>"
