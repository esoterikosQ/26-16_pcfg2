"""트리 자료구조 + 브래킷(Penn Treebank) I/O + 인덱스 매핑 헬퍼.

규약 (CLAUDE.md / plan.md §6-2):
- 코드 전역 표준: fencepost 인덱스 0..n, 반열림 구간 [i, j).
- 문헌 표기 변환은 이 파일의 매핑 헬퍼로만 수행한다.
  - Manning: 단어 위치 1..m, 포함 구간.
"""

from __future__ import annotations

from typing import List, Union


class Tree:
    """구구조 트리 노드.

    - 내부 노드: label(str) + children(List[Tree]).
    - 종단(단어)은 자식이 없는 Tree(label=단어)로 표현하지 않고, 부모(전종단,
      preterminal)의 children에 문자열로 들어간다. 즉 leaf word는 str, 그 외는 Tree.
    - 전종단(preterminal)은 children이 [str] 하나인 Tree (예: (DT the)).
    """

    __slots__ = ("label", "children")

    def __init__(self, label: str, children: List[Union["Tree", str]]):
        self.label = label
        self.children = children

    # --- 술어 ---
    def is_preterminal(self) -> bool:
        return len(self.children) == 1 and isinstance(self.children[0], str)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    # --- 파생 정보 ---
    def leaves(self) -> List[str]:
        """좌→우 단어 나열."""
        out: List[str] = []
        for c in self.children:
            if isinstance(c, str):
                out.append(c)
            else:
                out.extend(c.leaves())
        return out

    # --- 동치/표현 ---
    def __eq__(self, other) -> bool:
        if not isinstance(other, Tree):
            return NotImplemented
        return self.label == other.label and self.children == other.children

    def __repr__(self) -> str:
        return to_bracket(self)


# ----------------------------------------------------------------------------
# 브래킷 I/O
# ----------------------------------------------------------------------------

def to_bracket(tree: Union[Tree, str]) -> str:
    """트리를 Penn Treebank 브래킷 문자열로 직렬화."""
    if isinstance(tree, str):
        return tree
    if tree.is_preterminal():
        return f"({tree.label} {tree.children[0]})"
    inner = " ".join(to_bracket(c) for c in tree.children)
    return f"({tree.label} {inner})"


def _tokenize(s: str) -> List[str]:
    tokens: List[str] = []
    buf = []
    for ch in s:
        if ch in "()":
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)
        elif ch.isspace():
            if buf:
                tokens.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def from_bracket(s: str) -> Tree:
    """브래킷 문자열을 Tree로 파싱.

    Penn Treebank는 종종 최상단을 라벨 없는 괄호 `( (S ...) )`로 감싼다.
    그 경우 내부 트리를 반환한다.
    """
    tokens = _tokenize(s)
    pos = 0

    def parse() -> Union[Tree, str]:
        nonlocal pos
        assert tokens[pos] == "(", f"expected '(' at {pos}: {tokens[pos:pos+3]}"
        pos += 1  # consume '('
        # 라벨 (없을 수도 있음: 최상단 래퍼)
        label = None
        if tokens[pos] != "(" and tokens[pos] != ")":
            label = tokens[pos]
            pos += 1
        children: List[Union[Tree, str]] = []
        while tokens[pos] != ")":
            if tokens[pos] == "(":
                children.append(parse())
            else:
                children.append(tokens[pos])
                pos += 1
        pos += 1  # consume ')'
        if label is None:
            # 라벨 없는 래퍼: 자식이 하나면 그것을 반환
            assert len(children) == 1, "unlabeled node must wrap a single child"
            return children[0]
        return Tree(label, children)

    result = parse()
    assert isinstance(result, Tree), "top-level must be a Tree"
    return result


# ----------------------------------------------------------------------------
# 인덱스 매핑 헬퍼 (plan.md §6-2, Phase 0)
# ----------------------------------------------------------------------------

def manning_span_to_fencepost(i: int, j: int) -> tuple[int, int]:
    """Manning 포함 구간(1-indexed) i..j -> fencepost 반열림 [i-1, j).

    예) 단어 위치 1..5 전체 -> [0, 5).
        단일 단어 위치 p..p -> [p-1, p).
    """
    return (i - 1, j)


def fencepost_span_to_manning(start: int, end: int) -> tuple[int, int]:
    """fencepost 반열림 [start, end) -> Manning 포함 구간(1-indexed) i..j."""
    return (start + 1, end)


def word_position_to_fencepost(p: int) -> tuple[int, int]:
    """Manning 단어 위치 p(1-indexed) -> 그 단어의 fencepost 구간 [p-1, p)."""
    return (p - 1, p)
