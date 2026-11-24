"""K3 한국어 PCFG 기준선 평가: train 트리로 유도 → dev 표본 파싱(C++ 엔진) → PARSEVAL.

    python -m scripts.eval_korean --dev-n 2000 --workers 8 --tag base

- PARSEVAL 세 가지: 라벨 그대로(기능 표지 포함), 기능 표지 제거(NP_SBJ→NP), 무라벨.
  모두 루트·전종단 제외(`pcfg.eval`).
- 출력: 표준출력(JSON 요약), outputs/k3/<tag>/ (예측 트리·요약, git 제외 — 말뭉치 문장 포함).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcfg.eval import evaluate_corpus  # noqa: E402
from pcfg.induce import induce_pcfg, parse_sentence  # noqa: E402
from pcfg.korean import (  # noqa: E402
    mark_eojeol_boundaries, preprocess_keep, preprocess_strip, relabel_tree, strip_function_tag,
    unk_by_tag, unk_by_tag_eojeol,
)
from pcfg.tree import from_bracket, to_bracket  # noqa: E402

_IND = None


def _parse(item):
    idx, words = item
    t = time.time()
    tree, fallback = parse_sentence(words, _IND, engine="cpp")
    return idx, to_bracket(tree), fallback, time.time() - t


def _read(path: Path, limit: int, seed: int):
    lines = path.read_text(encoding="utf-8").splitlines()
    if limit and limit < len(lines):
        rng = random.Random(seed)
        lines = rng.sample(lines, limit)
    return [from_bracket(x) for x in lines]


def _bucket(n: int) -> str:
    for e in (20, 40, 60, 80):
        if n <= e:
            return f"≤{e}"
    return ">80"


def main() -> None:
    global _IND
    ap = argparse.ArgumentParser()
    ap.add_argument("--treebank", type=Path, default=Path("data/treebank"))
    ap.add_argument("--max-train", type=int, default=0, help="학습 트리 수 제한(0=전체)")
    ap.add_argument("--dev-n", type=int, default=2000)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-len", type=int, default=0, help="평가 문장 최대 형태소 수(0=제한 없음)")
    ap.add_argument("--strip-ftags", action="store_true", help="기능 표지를 지우고 학습")
    ap.add_argument("--mark-eojeol", action="store_true", help="어절 첫 형태소에 ▁ 표시(K4-1)")
    ap.add_argument("--markov-h", type=int, default=0, help="이항화 수평 마르코프화 차수(0 = h=∞, K4-2)")
    ap.add_argument("--unk-threshold", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()
    out = Path("outputs/k3") / args.tag
    out.mkdir(parents=True, exist_ok=True)
    prep = preprocess_strip if args.strip_ftags else preprocess_keep

    unk_fn = unk_by_tag_eojeol if args.mark_eojeol else unk_by_tag
    t0 = time.time()
    train = _read(args.treebank / "train.trees", args.max_train, args.seed)
    if args.mark_eojeol:
        train = [mark_eojeol_boundaries(t) for t in train]
    t_read = time.time() - t0
    t0 = time.time()
    _IND = induce_pcfg(train, unk_threshold=args.unk_threshold, preprocess_fn=prep, unk_fn=unk_fn,
                       markov_h=args.markov_h or None)
    t_induce = time.time() - t0
    g = _IND.grammar
    n_lex = sum(len(v) for v in g._lexical_by_word.values())
    n_bin = sum(len(v) for v in g._binary_by_rhs.values())
    n_un = sum(len(v) for v in g._unary_by_child.values())
    gram = {"train_trees": len(train), "nonterminals": len(g.nonterminals), "vocab": len(_IND.vocab),
            "lexical_rules": n_lex, "binary_rules": n_bin, "unary_rules": n_un,
            "read_sec": round(t_read, 1), "induce_sec": round(t_induce, 1)}
    print(json.dumps({"grammar": gram}, ensure_ascii=False), flush=True)
    del train

    dev = _read(args.treebank / f"{args.split}.trees", 0, args.seed)
    if args.mark_eojeol:
        dev = [mark_eojeol_boundaries(t) for t in dev]
    if args.max_len:
        dev = [t for t in dev if len(t.leaves()) <= args.max_len]
    rng = random.Random(args.seed + 1)
    if args.dev_n and args.dev_n < len(dev):
        dev = rng.sample(dev, args.dev_n)
    golds = [prep(t) for t in dev]
    items = [(i, t.leaves()) for i, t in enumerate(golds)]

    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        results = sorted(pool.imap_unordered(_parse, items, chunksize=8))
    wall = time.time() - t0

    preds = [from_bracket(b) for _, b, _, _ in results]
    fallbacks = sum(fb for _, _, fb, _ in results)
    by_len = defaultdict(list)
    for (i, _, _, sec), (_, words) in zip(results, items):
        by_len[_bucket(len(words))].append(sec)

    pairs = list(zip(golds, preds))
    full = evaluate_corpus(pairs)
    noft = evaluate_corpus([(relabel_tree(a, strip_function_tag), relabel_tree(b, strip_function_tag))
                            for a, b in pairs])
    unl = evaluate_corpus([(relabel_tree(a, lambda _: "X"), relabel_tree(b, lambda _: "X")) for a, b in pairs])

    def prf(p):
        return {"P": round(p.precision, 4), "R": round(p.recall, 4), "F1": round(p.f1, 4)}

    summary = {
        "tag": args.tag, "split": args.split, "eval_sents": len(golds), "strip_ftags": args.strip_ftags,
        "mark_eojeol": args.mark_eojeol, "markov_h": args.markov_h or "inf",
        "unk_threshold": args.unk_threshold, "fallbacks": fallbacks,
        "parseval_labeled": prf(full), "parseval_no_ftag": prf(noft), "parseval_unlabeled": prf(unl),
        "parse_wall_sec": round(wall, 1), "workers": args.workers,
        "sec_per_sent_by_len": {k: {"n": len(v), "mean": round(sum(v) / len(v), 4), "max": round(max(v), 3)}
                                for k, v in sorted(by_len.items())},
        "grammar": gram,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "pred.trees", "w", encoding="utf-8") as f:
        for _, b, _, _ in results:
            f.write(b + "\n")


if __name__ == "__main__":
    main()
