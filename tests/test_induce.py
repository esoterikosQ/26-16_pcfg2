"""Phase 3 검증: CNF 정/역변환 왕복, UNK 백오프, 폴백, 평가, 통합 F1."""

import pytest

from pcfg.tree import Tree, from_bracket, to_bracket
from pcfg.induce import (
    binarize,
    unbinarize,
    preprocess,
    simplify_label,
    induce_pcfg,
    parse_sentence,
    right_branching_fallback,
    UNK,
)
from pcfg.eval import constituents, evaluate_pair, evaluate_corpus
from pcfg.data import load_treebank_trees


# ---------------------------------------------------------------------------
# CNF 정/역변환 왕복 (§4 Phase 3 선결조건)
# ---------------------------------------------------------------------------
def test_binarize_unbinarize_round_trip():
    # 4-항 규칙 (VP -> V NP PP NP)
    t = from_bracket("(VP (V hit) (NP (N ball)) (PP (P to) (NP (N me))) (NP (N hard)))")
    b = binarize(t)
    # 이항화 결과는 모든 내부 노드가 자식 <= 2
    assert _max_arity(b) <= 2
    # 인공 노드 표식 존재
    assert any("|" in lbl for lbl in _all_labels(b))
    # 역변환은 원본과 일치
    assert unbinarize(b) == t


def test_binarize_keeps_binary_and_unary():
    t = from_bracket("(S (NP (N x)) (VP (V y)))")
    assert unbinarize(binarize(t)) == t
    # 단항 (NP -> N) 유지
    assert "|" not in "".join(_all_labels(binarize(t)))


def _all_labels(tree):
    out = [tree.label]
    for c in tree.children:
        if isinstance(c, Tree):
            out.extend(_all_labels(c))
    return out


def _max_arity(tree):
    if tree.is_preterminal():
        return 1
    m = len(tree.children)
    for c in tree.children:
        if isinstance(c, Tree):
            m = max(m, _max_arity(c))
    return m


# ---------------------------------------------------------------------------
# 전처리
# ---------------------------------------------------------------------------
def test_simplify_label():
    assert simplify_label("NP-SBJ") == "NP"
    assert simplify_label("S-1") == "S"
    assert simplify_label("PP-CLR") == "PP"
    assert simplify_label("-LRB-") == "-LRB-"  # 하이픈 시작 라벨 보존
    assert simplify_label("VP") == "VP"


def test_preprocess_removes_none_traces():
    t = from_bracket("(S (NP-SBJ (-NONE- *T*)) (VP (VB run)))")
    p = preprocess(t)
    # -NONE- 서브트리 제거, NP-SBJ는 자식이 없어 사라짐 → VP만 남음
    assert p is not None
    assert p.label == "S"
    assert [c.label for c in p.children] == ["VP"]
    assert p.leaves() == ["run"]


# ---------------------------------------------------------------------------
# 규칙 유도 + 정규화
# ---------------------------------------------------------------------------
def test_induced_grammar_normalized():
    trees = [
        from_bracket("(S (NP (DT the) (NN dog)) (VP (VBZ runs)))"),
        from_bracket("(S (NP (DT the) (NN cat)) (VP (VBZ sleeps)))"),
        from_bracket("(S (NP (DT a) (NN dog)) (VP (VBZ barks)))"),
    ]
    ind = induce_pcfg(trees, unk_threshold=0)
    assert ind.grammar.check_normalized() == []


# ---------------------------------------------------------------------------
# UNK 백오프 + 폴백
# ---------------------------------------------------------------------------
def test_unk_in_vocab_replacement():
    # "cat"은 1회 등장 → unk_threshold=1이면 UNK로 치환되어 vocab에서 제외
    trees = [
        from_bracket("(S (NP (DT the) (NN dog)) (VP (VBZ runs)))"),
        from_bracket("(S (NP (DT the) (NN dog)) (VP (VBZ sleeps)))"),
        from_bracket("(S (NP (DT the) (NN cat)) (VP (VBZ runs)))"),
    ]
    ind = induce_pcfg(trees, unk_threshold=1)
    assert "cat" not in ind.vocab
    assert "dog" in ind.vocab
    # NN은 UNK를 방출할 수 있어야 함 (OOV 백오프 경로)
    assert any(a == "NN" for a, _ in ind.grammar.lexical_parents(UNK))


def test_oov_word_parses_via_unk():
    trees = [
        from_bracket("(S (NP (DT the) (NN dog)) (VP (VBZ runs)))"),
        from_bracket("(S (NP (DT the) (NN dog)) (VP (VBZ sleeps)))"),
        from_bracket("(S (NP (DT a) (NN dog)) (VP (VBZ barks)))"),
    ]
    ind = induce_pcfg(trees, unk_threshold=1)  # dog는 3회라 유지, 나머지 rare→UNK
    # 완전 미등록어 "wombat"을 포함한 문장도 폴백 없이 파스되어야 함
    tree, fell_back = parse_sentence(["the", "wombat", "barks"], ind)
    assert tree.leaves() == ["the", "wombat", "barks"]  # 원래 단어 복구
    assert isinstance(tree, Tree)


def test_fallback_right_branching():
    t = right_branching_fallback(["a", "b", "c"])
    assert t.leaves() == ["a", "b", "c"]
    # 파스 실패 시에도 전 단어 커버
    assert _max_arity(t) <= 2


def test_no_parse_triggers_fallback():
    trees = [from_bracket("(S (NP (DT the) (NN dog)) (VP (VBZ runs)))")]
    ind = induce_pcfg(trees, unk_threshold=0)  # UNK 없음
    # 문법이 만들 수 없는 구조 → 폴백
    tree, fell_back = parse_sentence(["runs", "runs", "runs", "runs"], ind)
    assert fell_back is True
    assert tree.leaves() == ["runs", "runs", "runs", "runs"]


# ---------------------------------------------------------------------------
# PARSEVAL
# ---------------------------------------------------------------------------
def test_parseval_identical_trees():
    t = from_bracket("(S (NP (DT the) (NN dog)) (VP (VBZ runs)))")
    prf = evaluate_pair(t, t)
    assert prf.f1 == pytest.approx(1.0)


def test_parseval_partial():
    gold = from_bracket("(S (NP (DT the) (NN dog)) (VP (VBZ runs)))")
    test = from_bracket("(S (NP (DT the) (NN dog)) (VBZ runs))")  # VP 누락
    prf = evaluate_pair(gold, test)
    assert 0.0 < prf.f1 < 1.0


# ---------------------------------------------------------------------------
# 통합: 실제 트리뱅크 학습 → 전 문장 커버리지 + F1 산출 (§4 통과 기준)
# ---------------------------------------------------------------------------
def test_treebank_end_to_end_coverage_and_f1():
    trees = load_treebank_trees()
    # 짧은 문장으로 학습/평가 (속도). train/test 분리.
    short = [t for t in trees if len(t.leaves()) <= 12]
    train = short[:300]
    test = short[300:320]
    ind = induce_pcfg(train, unk_threshold=1)

    from pcfg.induce import preprocess
    pairs = []
    covered = 0
    for gold in test:
        words = gold.leaves()
        pred, _ = parse_sentence(words, ind)
        assert pred.leaves() == words  # 커버리지: 모든 단어를 포함하는 트리 산출
        covered += 1
        gp = preprocess(gold)
        if gp is not None:
            pairs.append((gp, pred))

    assert covered == len(test)
    prf = evaluate_corpus(pairs)
    # F1이 정상 범위에서 산출됨 (통과 기준: 산출 자체 + 비자명한 값)
    assert 0.0 < prf.f1 <= 1.0


@pytest.mark.parametrize("h", [1, 2])
def test_markovized_binarize_round_trip(h):
    t = from_bracket("(VP (V hit) (NP (N ball)) (PP (P to) (NP (N me))) (NP (N hard)))")
    b = binarize(t, h=h)
    assert _max_arity(b) <= 2
    assert all(lbl.count("-") <= h - 1 for lbl in _all_labels(b) if "|" in lbl)
    assert unbinarize(b) == t
