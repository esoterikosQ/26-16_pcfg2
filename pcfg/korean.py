"""한국어 트리뱅크: 모두의 말뭉치 구문 분석(DP) → 형태소 말단 구구조 트리.

결정(2026-11-24, `plan.md` K2):
- 모두의 말뭉치 구문 분석 말뭉치는 의존 구조지만, 라벨이 세종 구구조 표지(구 표지_기능 표지,
  예: ``NP_SBJ``, ``VP_MOD``)이고 지배소가 항상 오른쪽(핵어말)이며 거의 모두 투사적이다.
  따라서 결정적으로 구구조로 바꿀 수 있다: 지배소 h는 자신의 라벨로 투사하며, 가까운 의존소부터
  하나씩 묶어 올라간다.

      (L_h (P_d1) (L_h (P_d2) ( ... (L_h (P_dk) (L_h 어절_h)))))     d1 < d2 < ... < dk < h

  P_d는 의존소 d가 지배하는 구(라벨 L_d). 비투사 문장은 연속 구간을 이룰 수 없어 제외한다.
- 어절은 Kiwi 형태소로 펼친다(DP에는 형태소 분석이 없음). HNC 후보도 Kiwi 형태소이므로 말단이
  일치한다. 어절 노드 아래에 형태소 전종단을 평평하게 둔다: ``(NP_OBJ (#NNG 사과/NNG) (#JKO 를/JKO))``.
- 말단(단어) = ``형태/품사``. 품사가 다른 같은 형태는 다른 형태소다. 미등록 형태소는 품사별
  UNK(``<UNK/NNG>``)로 백오프한다.
- 전종단 라벨 = ``#품사``. Kiwi 품사 ``NP``(대명사)와 구 표지 ``NP``(명사구)가 섞이지 않게 한다.
- 브래킷 형식을 깨는 문자(괄호·공백)는 말단에서 이스케이프한다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

from .tree import Tree

POS_PREFIX = "#"
_ESCAPES = (("(", "-LRB-"), (")", "-RRB-"), (" ", "_"))


# ---------------------------------------------------------------------------
# 말단·전종단 표기
# ---------------------------------------------------------------------------
def escape_form(form: str) -> str:
    for a, b in _ESCAPES:
        form = form.replace(a, b)
    return form


def terminal(form: str, tag: str) -> str:
    """형태소 → 말단 기호 ``형태/품사``."""
    return f"{escape_form(form)}/{tag}"


def preterminal(tag: str) -> str:
    return POS_PREFIX + tag


def tag_of(term: str) -> str:
    return term.rsplit("/", 1)[1] if "/" in term else ""


def unk_by_tag(term: str) -> str:
    """미등록 형태소 → 품사별 UNK (``induce_pcfg(..., unk_fn=unk_by_tag)``)."""
    return f"<UNK/{tag_of(term)}>"


def strip_function_tag(label: str) -> str:
    """``NP_SBJ`` → ``NP``. 전종단(#품사)과 이항화 표식이 든 라벨은 그대로 둔다."""
    if label.startswith(POS_PREFIX) or "|" in label:
        return label
    return label.split("_", 1)[0]


def preprocess_keep(tree: Tree) -> Tree:
    """한국어 전처리(기능 표지 유지). Penn 규칙(하이픈 분리 등)을 적용하지 않는다."""
    return _copy(tree, lambda x: x)


def preprocess_strip(tree: Tree) -> Tree:
    """한국어 전처리(기능 표지 제거: NP_SBJ → NP)."""
    return _copy(tree, strip_function_tag)


def _copy(tree: Tree, relabel) -> Tree:
    if tree.is_preterminal():
        return Tree(tree.label, [tree.children[0]])
    return Tree(relabel(tree.label), [_copy(c, relabel) for c in tree.children])


def relabel_tree(tree: Tree, relabel) -> Tree:
    """평가용 라벨 사상(예: 기능 표지 제거 후 PARSEVAL)."""
    return _copy(tree, relabel)


# ---------------------------------------------------------------------------
# 부모 주석 (K4-3, Klein & Manning 2003의 v=2)
# ---------------------------------------------------------------------------
PARENT_SEP = "^"


def annotate_parent(tree: Tree, parent: Optional[str] = None) -> Tree:
    """구 노드 라벨에 부모 라벨을 붙인다(``NP_SBJ`` → ``NP_SBJ^VP``). 전종단과 루트는 그대로."""
    if tree.is_preterminal():
        return Tree(tree.label, [tree.children[0]])
    label = tree.label if parent is None else f"{tree.label}{PARENT_SEP}{parent}"
    return Tree(label, [annotate_parent(c, tree.label) for c in tree.children])


def preprocess_parent(tree: Tree) -> Tree:
    """기능 표지 유지 + 부모 주석."""
    return annotate_parent(preprocess_keep(tree))


def strip_parent(label: str) -> str:
    return label.split(PARENT_SEP, 1)[0]


# ---------------------------------------------------------------------------
# 어절 경계 표시 (K4-1)
# ---------------------------------------------------------------------------
EOJEOL_MARK = "▁"


def mark_eojeol_boundaries(tree: Tree) -> Tree:
    """어절 첫 형태소의 전종단·말단에 ``▁``를 붙인 사본 (``#▁NNG`` → ``▁사과/NNG``).

    변환 트리에서 어절 노드 = 자식이 모두 전종단인 노드다(투사 노드는 구 자식을 가짐).
    말단에만 붙이면 어휘 확률만 갈라지고 문법은 경계를 보지 못하므로(2026-11-24 실측) 전종단에도
    붙여, 어절 노드 규칙이 "어절 첫 형태소로 시작"을 학습하게 한다.
    HNC 후보는 띄어쓰기가 정해져 있으므로(형태소의 ``space_before``) 파서 입력에 어절 경계를
    주는 것은 정당하다.
    """
    if tree.is_preterminal():
        return Tree(tree.label, [tree.children[0]])
    kids = [mark_eojeol_boundaries(c) for c in tree.children]
    if all(isinstance(c, Tree) and c.is_preterminal() for c in tree.children):
        first = kids[0]
        label = POS_PREFIX + EOJEOL_MARK + first.label[len(POS_PREFIX):]
        kids[0] = Tree(label, [EOJEOL_MARK + first.children[0]])
    return Tree(tree.label, kids)


def unk_by_tag_eojeol(term: str) -> str:
    """품사별 UNK + 어절 첫 형태소 여부 (``<UNK▁/NNG>``)."""
    mark = EOJEOL_MARK if term.startswith(EOJEOL_MARK) else ""
    return f"<UNK{mark}/{tag_of(term)}>"


# ---------------------------------------------------------------------------
# DP 문장
# ---------------------------------------------------------------------------
@dataclass
class DPSentence:
    sent_id: str
    doc_id: str
    source: str               # 파일 이름 (예: NXDP2502512313)
    genre: str                # 'written'(NX) | 'spoken'(SX)
    form: str
    words: List[Tuple[int, int]]   # 어절별 [begin, end) (form 기준 문자 위치)
    word_forms: List[str]
    heads: List[int]               # 어절별 지배소(1-기반), 루트는 0
    labels: List[str]
    morphs: Optional[List[List[Tuple[str, str]]]] = field(default=None)


def dp_files(data_dir: Path) -> List[Path]:
    return sorted(Path(data_dir).glob("NIKL_DP_*/*.json"))


def iter_dp_sentences(paths: Sequence[Path]) -> Iterator[DPSentence]:
    import ijson
    for p in paths:
        genre = "spoken" if p.name.startswith("SX") else "written"
        with open(p, "rb") as f:
            for doc in ijson.items(f, "document.item"):
                for s in doc.get("sentence") or []:
                    dp = s.get("DP") or []
                    words = s.get("word") or []
                    if not dp or len(dp) != len(words):
                        continue
                    yield DPSentence(
                        sent_id=s.get("id", ""), doc_id=doc.get("id", ""), source=p.stem, genre=genre,
                        form=s.get("form", ""),
                        words=[(int(w["begin"]), int(w["end"])) for w in words],
                        word_forms=[w["form"] for w in words],
                        heads=[max(int(d["head"]), 0) for d in dp],
                        labels=[d["label"] for d in dp],
                    )


def is_projective(heads: Sequence[int]) -> bool:
    """1-기반 지배소 목록(루트 0)의 투사성: 호끼리 교차하지 않는가."""
    arcs = [(min(i, h), max(i, h)) for i, h in enumerate(heads, 1) if h > 0]
    arcs.sort()
    for x, (a, b) in enumerate(arcs):
        for c, d in arcs[x + 1:]:
            if c >= b:
                break
            if a < c < b < d:
                return False
    return True


def single_root(heads: Sequence[int]) -> bool:
    return sum(1 for h in heads if h == 0) == 1


# ---------------------------------------------------------------------------
# 형태소 배정 (Kiwi 토큰 → 어절)
# ---------------------------------------------------------------------------
def assign_morphemes(words: Sequence[Tuple[int, int]], tokens) -> Optional[List[List[Tuple[str, str]]]]:
    """Kiwi 토큰(form, tag, start, len)을 시작 위치가 속한 어절에 배정.

    토큰이 어절 경계를 넘거나, 어느 어절에도 속하지 않거나, 형태소가 없는 어절이 생기면 None.
    """
    out: List[List[Tuple[str, str]]] = [[] for _ in words]
    w = 0
    for t in tokens:
        start, end = t.start, t.start + t.len
        while w < len(words) and start >= words[w][1]:
            w += 1
        if w >= len(words) or start < words[w][0] or end > words[w][1]:
            return None
        out[w].append((t.form, t.tag))
    if any(not m for m in out):
        return None
    return out


# ---------------------------------------------------------------------------
# 의존 → 구구조
# ---------------------------------------------------------------------------
def dp_to_tree(heads: Sequence[int], labels: Sequence[str],
               morphs: Sequence[Sequence[Tuple[str, str]]]) -> Tree:
    """핵어말 의존 트리 → 구구조 트리 (모듈 docstring의 변환 규칙).

    의존소가 지배소 오른쪽에 있으면(이 말뭉치에는 없음) 왼쪽 의존소를 모두 묶은 뒤 가까운
    순서로 오른쪽에 묶는다. 투사성은 호출 전에 확인해야 한다.
    """
    n = len(heads)
    deps: List[List[int]] = [[] for _ in range(n + 1)]
    root = 0
    for i, h in enumerate(heads, 1):
        if h == 0:
            root = i
        else:
            deps[h].append(i)
    if root == 0:
        raise ValueError("루트가 없음")

    def eojeol(i: int) -> Tree:
        return Tree(labels[i - 1], [Tree(preterminal(t), [terminal(f, t)]) for f, t in morphs[i - 1]])

    def build(h: int) -> Tree:
        node = eojeol(h)
        lab = labels[h - 1]
        left = sorted((d for d in deps[h] if d < h), reverse=True)
        right = sorted(d for d in deps[h] if d > h)
        for d in left:
            node = Tree(lab, [build(d), node])
        for d in right:
            node = Tree(lab, [node, build(d)])
        return node

    return build(root)


def morpheme_terminals(morphs: Sequence[Sequence[Tuple[str, str]]]) -> List[str]:
    return [terminal(f, t) for m in morphs for f, t in m]


# ---------------------------------------------------------------------------
# 분할
# ---------------------------------------------------------------------------
def split_of(doc_id: str) -> str:
    """문서 id의 MD5 해시로 고정 분할 (train 90 / dev 5 / test 5)."""
    h = int(hashlib.md5(doc_id.encode("utf-8")).hexdigest(), 16) % 100
    if h < 90:
        return "train"
    if h < 95:
        return "dev"
    return "test"
