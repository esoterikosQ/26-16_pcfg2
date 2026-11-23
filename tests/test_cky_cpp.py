"""K1 검증: C++ 엔진(pcfg.cky_cpp)과 Python 참조 엔진(pcfg.cky)의 동치.

- 같은 최적 트리(`to_bracket`)와 비트 단위로 같은 로그확률(`==`; no-parse는 둘 다 -inf).
- 작은 무작위 문법(동점·중복 규칙·단항 순환 포함)에서는 셀 전체(삽입 순서·값·백포인터)까지
  비교해 순회 순서·동점 처리·단항 클로저 스냅숏을 검증한다.
- 영어 트리뱅크 held-out 문장에서 트리·로그확률 일치 (CLAUDE.md 검증 앵커 "엔진 동치").
- C++ 확장이 빌드되지 않았으면 import 단계에서 실패한다(skip하지 않는다).
"""

from __future__ import annotations

import random

import pytest

from pcfg import cky, cky_cpp
from pcfg.data import load_treebank_trees
from pcfg.grammar import PCFG
from pcfg.induce import UNK, induce_pcfg, parse_sentence, preprocess
from pcfg.tree import from_bracket, to_bracket
from tests.fixtures import MANNING_SENTENCE, manning_grammar

NEG_INF = float("-inf")


def _assert_same_result(rp, rc):
    assert rc.logprob == rp.logprob  # 비트 단위 (no-parse는 둘 다 -inf)
    assert rc.prob == rp.prob
    if rp.tree is None:
        assert rc.tree is None
    else:
        assert rc.tree is not None
        assert to_bracket(rc.tree) == to_bracket(rp.tree)


def _assert_same_chart(rp, rc, n):
    """모든 셀의 (라벨, 값) 삽입 순서와 백포인터가 같은지."""
    assert rc.chart.n == rp.chart.n == n
    for width in range(1, n + 1):
        for i in range(0, n - width + 1):
            j = i + width
            assert list(rc.chart.cell_score(i, j).items()) == list(rp.chart.score[(i, j)].items()), (i, j)
            assert list(rc.chart.cell_back(i, j).items()) == list(rp.chart.back[(i, j)].items()), (i, j)


# ---------------------------------------------------------------------------
# 경계 사례
# ---------------------------------------------------------------------------
def test_empty_sentence_no_parse():
    g = manning_grammar()
    r = cky_cpp.parse([], g)
    assert r.tree is None and r.prob == 0.0 and r.logprob == NEG_INF
    assert r.chart.n == 0
    _assert_same_result(cky.parse([], g), r)


def test_chart_lookup_unknown_label_and_span():
    g = manning_grammar()
    r = cky_cpp.parse(MANNING_SENTENCE, g)
    assert r.chart.cell_logprob(0, 5, "NOPE") == NEG_INF
    assert r.chart.cell_prob(0, 5, "NOPE") == 0.0
    assert r.chart.cell_logprob(0, 2, "S") == NEG_INF   # 라벨은 있으나 셀에 없음
    assert r.chart.cell_logprob(3, 3, "NP") == NEG_INF  # 빈 구간
    assert r.chart.cell_logprob(0, 9, "S") == NEG_INF   # 범위 밖
    assert r.chart.cell_logprob(0, 5, "S") == cky.parse(MANNING_SENTENCE, g).logprob


def test_manning_full_chart_identical():
    g = manning_grammar()
    rp, rc = cky.parse(MANNING_SENTENCE, g), cky_cpp.parse(MANNING_SENTENCE, g)
    _assert_same_result(rp, rc)
    _assert_same_chart(rp, rc, len(MANNING_SENTENCE))


def test_word_and_label_namespaces_are_separate():
    # 단어 "NP"는 라벨 NP와 무관해야 한다.
    g = PCFG(start="S")
    g.add_binary("S", "NP", "VP", 1.0)
    g.add_lexical("NP", "VP", 1.0)
    g.add_lexical("VP", "NP", 1.0)
    for words in (["VP", "NP"], ["NP", "VP"]):
        rp, rc = cky.parse(words, g), cky_cpp.parse(words, g)
        _assert_same_result(rp, rc)
        _assert_same_chart(rp, rc, len(words))
    assert cky_cpp.parse(["VP", "NP"], g).tree is not None
    assert cky_cpp.parse(["NP", "VP"], g).tree is None


def test_compile_cache_invalidated_by_add():
    g = PCFG(start="S")
    g.add_binary("S", "A", "A", 1.0)
    g.add_lexical("A", "a", 0.5)
    assert cky_cpp.parse(["a", "b"], g).tree is None
    g.add_lexical("A", "b", 0.5)  # _rev 증가 → 재컴파일
    rc = cky_cpp.parse(["a", "b"], g)
    assert rc.tree is not None
    _assert_same_result(cky.parse(["a", "b"], g), rc)


def test_ties_keep_first_candidate():
    # 모든 분할·유도가 같은 확률(1.0)이라 전부 동점: 먼저 나온 후보(k 최소)가 남아야 한다.
    g = PCFG(start="X")
    g.add_binary("X", "X", "X", 1.0)
    g.add_unary("X", "Y", 1.0)
    g.add_unary("Y", "X", 1.0)
    g.add_lexical("X", "a", 1.0)
    g.add_lexical("Y", "a", 1.0)
    words = ["a"] * 6
    rp, rc = cky.parse(words, g), cky_cpp.parse(words, g)
    _assert_same_result(rp, rc)
    _assert_same_chart(rp, rc, len(words))


# ---------------------------------------------------------------------------
# 무작위 문법: 셀 전체 동치 (동점·중복 규칙·단항 순환·OOV 포함)
# ---------------------------------------------------------------------------
def _random_grammar(rng: random.Random):
    labels = [f"N{x}" for x in range(rng.randint(2, 6))]
    words = [f"w{x}" for x in range(rng.randint(1, 4))]
    if rng.random() < 0.3:
        words.append(labels[0])  # 라벨과 같은 문자열의 단어
    tie_probs = [1.0, 0.5, 0.25, 0.1]

    def p() -> float:
        return rng.choice(tie_probs) if rng.random() < 0.5 else rng.uniform(0.01, 1.0)

    g = PCFG(start=labels[0])
    for _ in range(rng.randint(1, 15)):
        g.add_binary(rng.choice(labels), rng.choice(labels), rng.choice(labels), p())
    for _ in range(rng.randint(0, 6)):
        g.add_unary(rng.choice(labels), rng.choice(labels), p())
    for w in words:
        for _ in range(rng.randint(0, 3)):
            g.add_lexical(rng.choice(labels), w, p())
    return g, words + ["<oov>"]


def test_random_grammars_full_chart_equivalence():
    rng = random.Random(20261124)
    n_parsed = 0
    for _ in range(1000):
        g, vocab = _random_grammar(rng)
        for _ in range(5):
            words = [rng.choice(vocab) for _ in range(rng.randint(0, 10))]
            rp, rc = cky.parse(words, g), cky_cpp.parse(words, g)
            _assert_same_result(rp, rc)
            _assert_same_chart(rp, rc, len(words))
            n_parsed += rp.tree is not None
    assert n_parsed > 100  # 파스 성공 사례가 충분히 섞여 있어야 의미가 있다


# ---------------------------------------------------------------------------
# parse_sentence(engine=)
# ---------------------------------------------------------------------------
def test_parse_sentence_engine_switch():
    trees = [
        from_bracket("(S (NP (DT the) (NN dog)) (VP (VBZ runs)))"),
        from_bracket("(S (NP (DT the) (NN dog)) (VP (VBZ sleeps)))"),
        from_bracket("(S (NP (DT a) (NN dog)) (VP (VBZ barks)))"),
    ]
    ind = induce_pcfg(trees, unk_threshold=1)
    for words in (["the", "wombat", "barks"], ["runs", "runs", "runs"], ["the", "dog", "runs"]):
        tp, fp = parse_sentence(words, ind, engine="py")
        tc, fc = parse_sentence(words, ind, engine="cpp")
        assert (to_bracket(tc), fc) == (to_bracket(tp), fp)
    with pytest.raises(ValueError):
        parse_sentence(["the", "dog"], ind, engine="java")


# ---------------------------------------------------------------------------
# 영어 트리뱅크 held-out 동치 (K1 완료 기준 2)
# ---------------------------------------------------------------------------
TRAIN_MAX_LEN = 25     # 학습: 전처리 후 길이 <= 25 인 문장의 앞 1000개
N_TRAIN = 1000
HELDOUT_MAX_LEN = 18   # 평가: 그 뒤 문장 중 길이 <= 18 (Python 엔진 시간 때문에 제한, 약 40초)
N_HELDOUT = 200
N_FULL_CHART = 20      # 그중 앞 N_FULL_CHART 문장은 셀 전체까지 비교


def test_treebank_heldout_equivalence():
    pairs = []
    for t in load_treebank_trees():
        p = preprocess(t)
        if p is not None and len(p.leaves()) <= TRAIN_MAX_LEN:
            pairs.append((t, p))
    train = [t for t, _ in pairs[:N_TRAIN]]
    held = [p for _, p in pairs[N_TRAIN:] if len(p.leaves()) <= HELDOUT_MAX_LEN][:N_HELDOUT]
    assert len(held) == N_HELDOUT

    ind = induce_pcfg(train, unk_threshold=1)
    g = ind.grammar
    mismatches = []
    n_parsed = 0
    for idx, gold in enumerate(held):
        words = [w if w in ind.vocab else UNK for w in gold.leaves()]
        rp, rc = cky.parse(words, g), cky_cpp.parse(words, g)
        bp = None if rp.tree is None else to_bracket(rp.tree)
        bc = None if rc.tree is None else to_bracket(rc.tree)
        if bp != bc or rp.logprob != rc.logprob:
            mismatches.append((idx, len(words), rp.logprob, rc.logprob))
            continue
        n_parsed += rp.tree is not None
        if idx < N_FULL_CHART:
            _assert_same_chart(rp, rc, len(words))
    assert mismatches == []
    assert n_parsed > 0
