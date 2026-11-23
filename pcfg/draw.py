"""구구조 트리를 PNG 이미지로 렌더링 (headless, matplotlib Agg).

- 내부 노드(비단말)는 상자 라벨, 리프(단어)는 하단 텍스트.
- 리프에 x 좌표를 순서대로 배정하고, 부모 x는 자식 x의 평균으로 둔다.
- GUI 불필요(Agg 백엔드). NLTK Tk 의존성 없이 파일로 저장한다.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .tree import Tree


def _layout(tree: Tree):
    """각 노드에 (x, depth)를 배정. 리프는 왼→오 순서로 x=0,1,2,...

    반환: nodes = [(x, depth, label, is_leaf)], edges = [((x1,d1),(x2,d2))].
    """
    nodes: List[Tuple[float, int, str, bool]] = []
    edges: List[Tuple[Tuple[float, int], Tuple[float, int]]] = []
    leaf_x = [0.0]

    def rec(node: Union[Tree, str], depth: int) -> float:
        if isinstance(node, str):
            x = leaf_x[0]
            leaf_x[0] += 1.0
            nodes.append((x, depth, node, True))
            return x
        child_xs = [rec(c, depth + 1) for c in node.children]
        x = sum(child_xs) / len(child_xs)
        nodes.append((x, depth, node.label, False))
        for c, cx, cd in zip(node.children, child_xs,
                             [depth + 1] * len(node.children)):
            edges.append(((x, depth), (cx, cd)))
        return x

    rec(tree, 0)
    return nodes, edges


def save_tree_png(tree: Tree, path: str, title: str = "", dpi: int = 150) -> None:
    nodes, edges = _layout(tree)
    max_depth = max(d for _, d, _, _ in nodes)
    max_x = max(x for x, _, _, _ in nodes)

    # depth가 아래로 갈수록 y가 작아지도록 y = max_depth - depth
    def y_of(depth: int) -> float:
        return max_depth - depth

    width = max(6.0, (max_x + 1) * 0.9)
    height = max(3.0, (max_depth + 1) * 0.9)
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    for (x1, d1), (x2, d2) in edges:
        ax.plot([x1, x2], [y_of(d1), y_of(d2)], color="#888888",
                linewidth=0.8, zorder=1)

    for x, depth, label, is_leaf in nodes:
        y = y_of(depth)
        if is_leaf:
            ax.text(x, y - 0.28, label, ha="center", va="top", fontsize=9,
                    color="#1a5", style="italic", zorder=3)
        else:
            ax.text(x, y, label, ha="center", va="center", fontsize=9,
                    color="black", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#eef",
                              edgecolor="#557", linewidth=0.6))

    ax.set_xlim(-0.8, max_x + 0.8)
    ax.set_ylim(-0.8, max_depth + 0.6)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, pad=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
