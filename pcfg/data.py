"""NLTK Penn Treebank 샘플 로더 (Phase 3 데이터).

nltk.download('treebank')로 받는 무료 샘플(약 3,900문장)을 사용한다.
NLTK 트리를 이 프로젝트의 Tree(pcfg.tree.Tree)로 변환한다.
"""

from __future__ import annotations

from typing import List

from .tree import Tree


def _from_nltk(nltk_tree) -> Tree:
    """nltk.tree.Tree -> pcfg.tree.Tree 변환."""
    children = []
    for child in nltk_tree:
        if isinstance(child, str):
            children.append(child)
        else:
            children.append(_from_nltk(child))
    return Tree(nltk_tree.label(), children)


def load_treebank_trees(limit: int | None = None) -> List[Tree]:
    """NLTK treebank의 파싱된 문장들을 Tree 리스트로 반환."""
    from nltk.corpus import treebank

    trees: List[Tree] = []
    for i, sent in enumerate(treebank.parsed_sents()):
        if limit is not None and i >= limit:
            break
        trees.append(_from_nltk(sent))
    return trees
