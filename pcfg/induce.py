"""Phase 3 — Treebank 지도 학습.

- 규칙 카운팅 → 정규화: P(α→β) = Count(α→β) / Count(α).  (Manning 지도 학습 공식)
- CNF 정변환: 이항화(+단항은 CKY의 일반화 단항 클로저로 처리) — 확률 보존.
- CNF 역변환(de-binarization): 이항화 인공 노드(표식 `|`)를 제거해 원본 n-항 복원.
- OOV/UNK: 희귀어를 <UNK>로 치환, 미등록어는 파스 시 <UNK>로 백오프.
- 파스 실패 폴백: 우분지 트리 강제 반환(테스트셋 전 문장 커버리지 보장).

트리뱅크 전처리(표준):
- `-NONE-` 흔적(빈 요소) 제거.
- 함수 태그·인덱스 제거: `NP-SBJ`→`NP`, `S-1`→`S` 등.
- 루트 라벨이 여럿(S, SINV, ...)이므로 `TOP` 노드로 감싸 단일 시작 기호를 만든다.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional, Tuple

from .cky import parse
from .grammar import PCFG
from .tree import Tree

UNK = "<UNK>"
TOP = "TOP"
MARKER = "|"


# ---------------------------------------------------------------------------
# 전처리
# ---------------------------------------------------------------------------
def simplify_label(label: str) -> str:
    """함수 태그·공지시 인덱스 제거. `-LRB-` 등 하이픈으로 시작하는 라벨은 보존."""
    if label and label[0] != "-":
        return re.split(r"[-=]", label)[0]
    return label


def preprocess(tree: Tree) -> Optional[Tree]:
    """`-NONE-` 흔적 제거 + 라벨 단순화. 전부 제거되면 None."""
    if tree.is_preterminal():
        if tree.label == "-NONE-":
            return None
        return Tree(simplify_label(tree.label), [tree.children[0]])
    new_children = []
    for c in tree.children:
        pc = preprocess(c)
        if pc is not None:
            new_children.append(pc)
    if not new_children:
        return None
    return Tree(simplify_label(tree.label), new_children)


def add_top(tree: Tree) -> Tree:
    return Tree(TOP, [tree])


def strip_top(tree: Tree) -> Tree:
    if tree.label == TOP and len(tree.children) == 1 and isinstance(tree.children[0], Tree):
        return tree.children[0]
    return tree


# ---------------------------------------------------------------------------
# CNF 정/역변환
# ---------------------------------------------------------------------------
def binarize(tree: Tree, marker: str = MARKER) -> Tree:
    """n-항(>2) 규칙을 우분지 이항화. 인공 중간 노드에 표식(marker)을 남긴다.

    중간 노드 라벨은 (부모, 남은 꼬리 라벨열)로만 결정되는 비중첩 방식
    (Klein & Manning h=∞ 무손실 이항화). 동일 부모·동일 꼬리는 라벨을 공유해
    규칙이 과도하게 희소해지는 것을 막는다.
    """
    if tree.is_preterminal():
        return Tree(tree.label, [tree.children[0]])
    children = [binarize(c, marker) for c in tree.children]
    if len(children) <= 2:
        return Tree(tree.label, children)

    parent = tree.label
    labels = [c.label for c in children]
    n = len(children)

    def inter_label(s: int) -> str:  # children[s:] 를 덮는 중간 노드 라벨
        return f"{parent}{marker}<{'-'.join(labels[s:])}>"

    # 가장 안쪽: 마지막 두 자식
    inter = Tree(inter_label(n - 2), [children[n - 2], children[n - 1]])
    for s in range(n - 3, 0, -1):
        inter = Tree(inter_label(s), [children[s], inter])
    return Tree(parent, [children[0], inter])


def unbinarize(tree, marker: str = MARKER):
    """이항화 인공 노드(표식 포함 라벨)를 제거해 원래 n-항 트리로 복원."""
    if isinstance(tree, str):
        return tree
    if tree.is_preterminal():
        return Tree(tree.label, [tree.children[0]])
    new_children = []
    for c in tree.children:
        uc = unbinarize(c, marker)
        if isinstance(uc, Tree) and marker in uc.label:
            new_children.extend(uc.children)  # 인공 노드: 자식을 부모로 승격
        else:
            new_children.append(uc)
    return Tree(tree.label, new_children)


# ---------------------------------------------------------------------------
# 규칙 수집 + UNK
# ---------------------------------------------------------------------------
def _word_counts(trees: List[Tree]) -> Counter:
    c: Counter = Counter()
    for t in trees:
        for w in t.leaves():
            c[w] += 1
    return c


def _collect_rules(tree: Tree, vocab: set, counts_lhs: Counter,
                   counts_rule: Counter) -> None:
    """이항화된 트리에서 규칙을 카운팅. 희귀어(vocab 밖)는 UNK로 치환."""
    if isinstance(tree, str):
        return
    if tree.is_preterminal():
        word = tree.children[0]
        token = word if word in vocab else UNK
        counts_lhs[tree.label] += 1
        counts_rule[(tree.label, (token,))] += 1
        return
    rhs = tuple(c.label for c in tree.children)
    counts_lhs[tree.label] += 1
    counts_rule[(tree.label, rhs)] += 1
    for c in tree.children:
        _collect_rules(c, vocab, counts_lhs, counts_rule)


class InducedGrammar:
    """유도된 PCFG + 어휘(vocab)를 묶는 컨테이너."""

    def __init__(self, grammar: PCFG, vocab: set, start: str = TOP):
        self.grammar = grammar
        self.vocab = vocab
        self.start = start


def induce_pcfg(trees: List[Tree], unk_threshold: int = 1) -> InducedGrammar:
    """트리뱅크 트리들에서 CNF PCFG를 유도.

    unk_threshold: 이 빈도 이하(<=)인 단어를 UNK로 치환.
    """
    # 1) 전처리 + TOP 래핑
    processed: List[Tree] = []
    for t in trees:
        pt = preprocess(t)
        if pt is not None:
            processed.append(add_top(pt))

    # 2) 어휘: 빈도 > unk_threshold 인 단어만 유지
    wc = _word_counts(processed)
    vocab = {w for w, c in wc.items() if c > unk_threshold}

    # 3) 이항화 + 규칙 카운팅
    counts_lhs: Counter = Counter()
    counts_rule: Counter = Counter()
    for t in processed:
        bt = binarize(t)
        _collect_rules(bt, vocab, counts_lhs, counts_rule)

    # 4) 정규화 → PCFG
    g = PCFG(start=TOP)
    for (lhs, rhs), c in counts_rule.items():
        prob = c / counts_lhs[lhs]
        is_lexical = len(rhs) == 1 and (rhs[0] in vocab or rhs[0] == UNK)
        # 단항 비단말(A→B, B가 비단말)과 어휘(A→w) 구분: 어휘는 vocab/UNK 토큰.
        g.add(lhs, rhs, prob, lexical=is_lexical)
    return InducedGrammar(grammar=g, vocab=vocab, start=TOP)


# ---------------------------------------------------------------------------
# 파싱 (UNK 백오프 + 폴백) → 원본 n-항 트리 반환
# ---------------------------------------------------------------------------
def _map_unknowns(words: List[str], vocab: set) -> List[str]:
    return [w if w in vocab else UNK for w in words]


def right_branching_fallback(words: List[str], label: str = "S",
                             pos: str = "X") -> Tree:
    """파스 실패 시 우분지 트리 강제 반환 (커버리지 보장)."""
    if len(words) == 1:
        return Tree(label, [Tree(pos, [words[0]])])
    first = Tree(pos, [words[0]])
    rest = right_branching_fallback(words[1:], label=label, pos=pos)
    return Tree(label, [first, rest])


def parse_sentence(words: List[str], induced: InducedGrammar) -> Tuple[Tree, bool]:
    """문장을 파싱해 (원본 n-항 트리, 폴백여부) 반환.

    - OOV 단어는 <UNK>로 백오프.
    - 파스 성공 시 CNF 역변환 + TOP 제거.
    - 실패 시 우분지 폴백.
    """
    mapped = _map_unknowns(words, induced.vocab)
    result = parse(mapped, induced.grammar)
    if result.tree is None:
        return right_branching_fallback(list(words)), True

    tree = unbinarize(result.tree)
    tree = strip_top(tree)
    _restore_words(tree, words)  # UNK 리프를 원래 단어로 복구
    return tree, False


def _restore_words(tree: Tree, words: List[str]) -> None:
    """파스 트리의 리프(UNK 포함)를 원래 단어열로 좌→우 복구 (in-place)."""
    idx = [0]

    def rec(node: Tree) -> None:
        for i, c in enumerate(node.children):
            if isinstance(c, str):
                node.children[i] = words[idx[0]]
                idx[0] += 1
            else:
                rec(c)

    rec(tree)
