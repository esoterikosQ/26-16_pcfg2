"""문헌 예제 문법·트리를 원문 규칙명·확률 그대로 하드코딩 (검증 앵커용).

- Jurafsky Fig E.1  : "Book the dinner flight" (트리 확률, §3.1)
- Jurafsky Fig E.4  : "The flight includes a meal" (CNF, CKY 셀 값, §3.3)
- Manning Table 11.2: "astronomers saw stars with ears" (CNF, 트리 확률·CKY, §3.2/§3.4)
"""

from __future__ import annotations

from pcfg.grammar import PCFG
from pcfg.tree import Tree


# ---------------------------------------------------------------------------
# Jurafsky Fig E.1 (완전 문법; 이항 아님 — 트리 확률 검증 전용)
# ---------------------------------------------------------------------------
def jurafsky_e1_grammar() -> PCFG:
    g = PCFG(start="S")
    # Grammar
    g.add("S", ("NP", "VP"), 0.80)
    g.add("S", ("Aux", "NP", "VP"), 0.15)
    g.add("S", ("VP",), 0.05)
    g.add("NP", ("Pronoun",), 0.35)
    g.add("NP", ("Proper-Noun",), 0.30)
    g.add("NP", ("Det", "Nominal"), 0.20)
    g.add("NP", ("Nominal",), 0.15)
    g.add("Nominal", ("Noun",), 0.75)
    g.add("Nominal", ("Nominal", "Noun"), 0.20)
    g.add("Nominal", ("Nominal", "PP"), 0.05)
    g.add("VP", ("Verb",), 0.35)
    g.add("VP", ("Verb", "NP"), 0.20)
    g.add("VP", ("Verb", "NP", "PP"), 0.10)
    g.add("VP", ("Verb", "PP"), 0.15)
    g.add("VP", ("Verb", "NP", "NP"), 0.05)
    g.add("VP", ("VP", "PP"), 0.15)
    g.add("PP", ("Preposition", "NP"), 1.0)
    # Lexicon
    for w, p in [("that", .10), ("a", .30), ("the", .60)]:
        g.add_lexical("Det", w, p)
    for w, p in [("book", .10), ("trip", .30), ("meal", .05), ("money", .05),
                 ("flight", .40), ("dinner", .10)]:
        g.add_lexical("Noun", w, p)
    for w, p in [("book", .30), ("include", .30), ("prefer", .40)]:
        g.add_lexical("Verb", w, p)
    for w, p in [("I", .40), ("she", .05), ("me", .15), ("you", .40)]:
        g.add_lexical("Pronoun", w, p)
    for w, p in [("Houston", .60), ("NWA", .40)]:
        g.add_lexical("Proper-Noun", w, p)
    for w, p in [("does", .60), ("can", .40)]:
        g.add_lexical("Aux", w, p)
    for w, p in [("from", .30), ("to", .30), ("on", .20), ("near", .15), ("through", .05)]:
        g.add_lexical("Preposition", w, p)
    return g


def jurafsky_e1_tree_left() -> Tree:
    """T_left: "Book a flight that serves dinner" 의미. P = 2.2e-6 (§3.1)."""
    return Tree("S", [
        Tree("VP", [
            Tree("Verb", ["book"]),
            Tree("NP", [
                Tree("Det", ["the"]),
                Tree("Nominal", [
                    Tree("Nominal", [Tree("Noun", ["dinner"])]),
                    Tree("Noun", ["flight"]),
                ]),
            ]),
        ]),
    ])


def jurafsky_e1_tree_right() -> Tree:
    """T_right: 비상식적 의미 ("Book [the dinner] [flight]", 이중목적어).

    erratum 주의: Fig E.2는 이 트리의 VP→Verb NP NP를 .10으로 곱해 6.1e-7을
    제시하지만, Fig E.1 문법의 정규화 값은 .05다(.05라야 VP 규칙 합=1.0).
    본 프로젝트는 정규화된 충실한 문법(.05)을 채택하므로 P(T_right)=3.0375e-7.
    """
    return Tree("S", [
        Tree("VP", [
            Tree("Verb", ["book"]),
            Tree("NP", [
                Tree("Det", ["the"]),
                Tree("Nominal", [Tree("Noun", ["dinner"])]),
            ]),
            Tree("NP", [
                Tree("Nominal", [Tree("Noun", ["flight"])]),
            ]),
        ]),
    ])


# ---------------------------------------------------------------------------
# Jurafsky Fig E.4 미니 문법 (이미 CNF; §3.3 셀 값 검증)
# ---------------------------------------------------------------------------
def jurafsky_e4_grammar() -> PCFG:
    g = PCFG(start="S")
    g.add_binary("S", "NP", "VP", 0.80)
    g.add_binary("NP", "Det", "N", 0.30)
    g.add_binary("VP", "V", "NP", 0.20)
    g.add_lexical("V", "includes", 0.05)
    g.add_lexical("Det", "the", 0.40)
    g.add_lexical("Det", "a", 0.40)
    g.add_lexical("N", "meal", 0.01)
    g.add_lexical("N", "flight", 0.02)
    return g


# ---------------------------------------------------------------------------
# Manning Table 11.2 (순수 CNF; §3.2 트리 확률, §3.4 CKY·최적 파스)
# ---------------------------------------------------------------------------
def manning_grammar() -> PCFG:
    g = PCFG(start="S")
    g.add_binary("S", "NP", "VP", 1.0)
    g.add_binary("PP", "P", "NP", 1.0)
    g.add_binary("VP", "V", "NP", 0.7)
    g.add_binary("VP", "VP", "PP", 0.3)
    g.add_binary("NP", "NP", "PP", 0.4)
    g.add_lexical("P", "with", 1.0)
    g.add_lexical("V", "saw", 1.0)
    g.add_lexical("NP", "astronomers", 0.1)
    g.add_lexical("NP", "ears", 0.18)
    g.add_lexical("NP", "saw", 0.04)
    g.add_lexical("NP", "stars", 0.18)
    g.add_lexical("NP", "telescopes", 0.1)
    return g


MANNING_SENTENCE = ["astronomers", "saw", "stars", "with", "ears"]


def manning_tree_t1() -> Tree:
    """t1 (V NP 부착). P = 0.0009072 (§3.2). Viterbi 최적 파스."""
    return Tree("S", [
        Tree("NP", ["astronomers"]),
        Tree("VP", [
            Tree("V", ["saw"]),
            Tree("NP", [
                Tree("NP", ["stars"]),
                Tree("PP", [
                    Tree("P", ["with"]),
                    Tree("NP", ["ears"]),
                ]),
            ]),
        ]),
    ])


def manning_tree_t2() -> Tree:
    """t2 (VP PP 부착). P = 0.0006804 (§3.2)."""
    return Tree("S", [
        Tree("NP", ["astronomers"]),
        Tree("VP", [
            Tree("VP", [
                Tree("V", ["saw"]),
                Tree("NP", ["stars"]),
            ]),
            Tree("PP", [
                Tree("P", ["with"]),
                Tree("NP", ["ears"]),
            ]),
        ]),
    ])
