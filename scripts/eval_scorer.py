"""K5 판별력 점검: dev 문장 vs 형태소 오류를 넣은 문장의 PCFG 점수 비교.

    python -m scripts.eval_scorer --n 3000 --workers 12

오류 종류(각 문장에서 적용 가능한 위치 하나를 무작위로 고름):
- case_swap   : 격조사 대치 JKS ↔ JKO (이형태는 앞 형태소의 받침에 맞춤: 이/가, 을/를)
- particle_del: 어절 안의 조사(JK*, JX) 삭제
- eojeol_swap : 인접한 두 어절의 순서 바꿈
- final_ec    : 종결어미(EF)를 연결어미 '고/EC'로 바꿈
지표: 원문 점수 > 오류 문장 점수인 비율(동점은 0.5). 길이가 바뀌는 particle_del은 형태소당 점수로도 비교.
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

from pcfg.korean import EOJEOL_MARK, mark_eojeol_boundaries  # noqa: E402
from pcfg.score import PCFGScorer  # noqa: E402
from pcfg.tree import from_bracket  # noqa: E402

_SC = None


def to_morphs(tree):
    out = []
    for k, leaf in enumerate(mark_eojeol_boundaries(tree).leaves()):
        space = leaf.startswith(EOJEOL_MARK) and k > 0
        form, tag = leaf.lstrip(EOJEOL_MARK).rsplit("/", 1)
        out.append((form, tag, space))
    return out


def _has_coda(form: str):
    ch = form[-1] if form else ""
    if not ("가" <= ch <= "힣"):
        return None
    return (ord(ch) - 0xAC00) % 28 != 0


def _eojeols(m):
    groups = []
    for x in m:
        if not groups or x[2]:
            groups.append([])
        groups[-1].append(x)
    return groups


def corrupt(m, kind, rng):
    if kind == "case_swap":
        pos = [i for i, x in enumerate(m) if x[1] in ("JKS", "JKO") and i > 0 and _has_coda(m[i - 1][0]) is not None]
        if not pos:
            return None
        i = rng.choice(pos)
        coda = _has_coda(m[i - 1][0])
        new = (("을" if coda else "를"), "JKO") if m[i][1] == "JKS" else (("이" if coda else "가"), "JKS")
        return m[:i] + [(new[0], new[1], m[i][2])] + m[i + 1:]
    if kind == "particle_del":
        pos = [i for i, x in enumerate(m) if (x[1].startswith("JK") or x[1] == "JX") and not x[2] and i > 0]
        if not pos:
            return None
        i = rng.choice(pos)
        return m[:i] + m[i + 1:]
    if kind == "eojeol_swap":
        g = _eojeols(m)
        if len(g) < 2:
            return None
        i = rng.randrange(len(g) - 1)
        g[i], g[i + 1] = g[i + 1], g[i]
        out = []
        for k, e in enumerate(g):
            for j, (f, t, _) in enumerate(e):
                out.append((f, t, j == 0 and k > 0))
        return out
    if kind == "final_ec":
        pos = [i for i, x in enumerate(m) if x[1] == "EF"]
        if not pos:
            return None
        i = pos[-1]
        return m[:i] + [("고", "EC", m[i][2])] + m[i + 1:]
    raise ValueError(kind)


KINDS = ("case_swap", "particle_del", "eojeol_swap", "final_ec")


def _work(item):
    idx, m, variants = item
    t = time.time()
    base = _SC.score(m)
    res = {}
    for kind, v in variants.items():
        r = _SC.score(v)
        res[kind] = (r.logprob, r.parsed, len(v))
    return idx, (base.logprob, base.parsed, len(m)), res, time.time() - t


def main() -> None:
    global _SC
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("data/models/pcfg_best.pkl"))
    ap.add_argument("--dev", type=Path, default=Path("data/treebank/dev.trees"))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--max-len", type=int, default=60)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _SC = PCFGScorer.load(args.model)
    rng = random.Random(args.seed)
    trees = [from_bracket(x) for x in args.dev.read_text(encoding="utf-8").splitlines()]
    trees = [t for t in trees if len(t.leaves()) <= args.max_len]
    trees = rng.sample(trees, min(args.n, len(trees)))
    items = []
    for i, t in enumerate(trees):
        m = to_morphs(t)
        variants = {k: v for k in KINDS if (v := corrupt(m, k, rng)) is not None}
        items.append((i, m, variants))

    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = list(pool.imap_unordered(_work, items, chunksize=8))
    wall = time.time() - t0

    acc = defaultdict(list)
    acc_norm = defaultdict(list)
    parsed = defaultdict(list)
    base_parsed = []
    for _, (blp, bparsed, bn), res, _ in results:
        base_parsed.append(bparsed)
        for kind, (lp, p, n) in res.items():
            acc[kind].append(1.0 if blp > lp else 0.5 if blp == lp else 0.0)
            acc_norm[kind].append(1.0 if blp / bn > lp / n else 0.5 if blp / bn == lp / n else 0.0)
            parsed[kind].append(p)
    summary = {
        "sentences": len(results), "max_len": args.max_len, "wall_sec": round(wall, 1),
        "orig_parse_rate": round(sum(base_parsed) / len(base_parsed), 4),
        "by_kind": {k: {"n": len(v), "acc_raw": round(sum(v) / len(v), 4),
                        "acc_per_morph": round(sum(acc_norm[k]) / len(acc_norm[k]), 4),
                        "corrupt_parse_rate": round(sum(parsed[k]) / len(parsed[k]), 4)}
                    for k, v in acc.items()},
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
