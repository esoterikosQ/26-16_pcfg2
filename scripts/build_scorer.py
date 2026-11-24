"""K5 채점기 구축: 전체 train 트리로 최선 설정 PCFG를 유도해 저장.

    python -m scripts.build_scorer        # → data/models/pcfg_best.pkl (git 제외: 말뭉치 파생물)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcfg.score import BEST_CONFIG, PCFGScorer, induce_best  # noqa: E402
from pcfg.tree import from_bracket  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/treebank/train.trees"))
    ap.add_argument("--out", type=Path, default=Path("data/models/pcfg_best.pkl"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    trees = [from_bracket(x) for x in args.train.read_text(encoding="utf-8").splitlines()]
    t_read = time.time() - t0
    t0 = time.time()
    scorer = PCFGScorer(induce_best(trees, unk_threshold=BEST_CONFIG["unk_threshold"]))
    t_induce = time.time() - t0
    scorer.save(args.out)
    g = scorer.ind.grammar
    print(json.dumps({"out": str(args.out), "bytes": args.out.stat().st_size, "train_trees": len(trees),
                      "nonterminals": len(g.nonterminals), "vocab": len(scorer.ind.vocab),
                      "config": BEST_CONFIG, "read_sec": round(t_read, 1), "induce_sec": round(t_induce, 1)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
