#!/usr/bin/env python
"""K1 — Python 참조 엔진 vs C++ 엔진 CKY 속도 비교 (문장 길이 구간별).

NLTK 트리뱅크 앞 --train-size 문장으로 PCFG를 유도하고, 나머지(held-out) 문장을
전처리(-NONE- 제거) 후 UNK 매핑한 단어열을 두 엔진으로 파싱한다.
- C++: 모든 held-out 문장.
- Python: 구간별 시간 예산(--py-budget 초)을 넘거나 길이가 --py-max-len을 넘으면 중단.
- 속도비는 두 엔진이 모두 파싱한 같은 문장들의 총 시간 비. 같은 문장에서 트리·logprob
  일치도 함께 센다.

사용 예 (itcerdo, 저장소 루트, .venv 활성화):
    python -m scripts.bench_cky
    python -m scripts.bench_cky --py-budget 300 --py-max-len 40
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import List, Tuple

from pcfg import cky, cky_cpp
from pcfg.data import load_treebank_trees
from pcfg.induce import UNK, induce_pcfg, preprocess
from pcfg.tree import to_bracket

BUCKETS: List[Tuple[str, int, int]] = [
    ("≤10", 1, 10),
    ("11–20", 11, 20),
    ("21–30", 21, 30),
    ("31–40", 31, 40),
    ("41+", 41, 10**9),
]


def _bracket(result):
    return None if result.tree is None else to_bracket(result.tree)


def main() -> None:
    ap = argparse.ArgumentParser(description="Python vs C++ CKY 속도 비교")
    ap.add_argument("--train-size", type=int, default=3500)
    ap.add_argument("--unk-threshold", type=int, default=1)
    ap.add_argument("--py-budget", type=float, default=180.0,
                    help="구간별 Python 엔진 시간 예산(초)")
    ap.add_argument("--py-max-len", type=int, default=40,
                    help="이 길이를 넘는 문장은 Python 엔진으로 파싱하지 않음")
    args = ap.parse_args()

    trees = load_treebank_trees()
    train, held = trees[:args.train_size], trees[args.train_size:]

    t0 = time.perf_counter()
    ind = induce_pcfg(train, unk_threshold=args.unk_threshold)
    t_induce = time.perf_counter() - t0
    g = ind.grammar

    t0 = time.perf_counter()
    compiled = cky_cpp.compile_grammar(g)
    t_compile = time.perf_counter() - t0
    nat = compiled.native
    print(f"train={len(train)} held-out={len(held)} induce={t_induce:.2f}s "
          f"cpp-compile={t_compile:.3f}s")
    print(f"grammar: labels={nat.n_labels} words={nat.n_words} "
          f"binary={nat.n_binary_rules} (pairs={nat.n_binary_pairs}) "
          f"unary={nat.n_unary_rules} lexical={nat.n_lexical_rules}")

    sents: List[List[str]] = []
    for t in held:
        p = preprocess(t)
        if p is not None:
            sents.append([w if w in ind.vocab else UNK for w in p.leaves()])

    rows = []
    for name, lo, hi in BUCKETS:
        bucket = [s for s in sents if lo <= len(s) <= hi]
        if not bucket:
            continue
        cpp_t, cpp_res = [], []
        for s in bucket:
            t0 = time.perf_counter()
            r = cky_cpp.parse(s, g)
            cpp_t.append(time.perf_counter() - t0)
            cpp_res.append((_bracket(r), r.logprob))

        py_t: List[float] = []
        same = 0
        if lo <= args.py_max_len:
            for s, (bc, lc) in zip(bucket, cpp_res):
                if len(s) > args.py_max_len or sum(py_t) >= args.py_budget:
                    break
                t0 = time.perf_counter()
                r = cky.parse(s, g)
                py_t.append(time.perf_counter() - t0)
                same += (_bracket(r) == bc and r.logprob == lc)

        n_py = len(py_t)
        speedup = (sum(py_t) / sum(cpp_t[:n_py])) if n_py else float("nan")
        rows.append((name, len(bucket), statistics.mean(len(s) for s in bucket),
                     statistics.mean(cpp_t), max(cpp_t), n_py,
                     statistics.mean(py_t) if n_py else float("nan"),
                     speedup, same))
        print(f"  {name}: n={len(bucket)} cpp={statistics.mean(cpp_t):.4f}s/sent "
              f"py={rows[-1][6]:.3f}s/sent (n={n_py}) speedup={speedup:.1f}x "
              f"same={same}/{n_py}", flush=True)

    print()
    print("| 길이 | 문장 수 | 평균 길이 | C++ s/문장 | C++ 최대 s | Py 측정 수 | Py s/문장 "
          "| 속도비(같은 문장) | 일치(트리·logprob) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, n, mlen, cm, cmax, n_py, pm, sp, same in rows:
        pm_s = f"{pm:.3f}" if n_py else "—"
        sp_s = f"{sp:.0f}×" if n_py else "—"
        same_s = f"{same}/{n_py}" if n_py else "—"
        print(f"| {name} | {n} | {mlen:.1f} | {cm:.4f} | {cmax:.3f} | {n_py} | {pm_s} "
              f"| {sp_s} | {same_s} |")


if __name__ == "__main__":
    main()
