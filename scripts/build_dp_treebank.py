"""K2 한국어 트리뱅크 구축: 모두의 말뭉치 구문 분석(DP) → 형태소 말단 구구조 트리.

    python -m scripts.build_dp_treebank                 # data/NIKL_DP_* → data/treebank/

출력(data/ 아래, git 제외):
- ``data/treebank/{train,dev,test}.trees``  : 한 줄에 브래킷 트리 하나 (TOP 없이)
- ``data/treebank/{train,dev,test}.meta.jsonl`` : 문장 id, 문서 id, 출처 파일, 장르, 어절·형태소 수
- ``data/treebank/stats.json``               : 걸러낸 사유별 개수, 분할 통계

거르는 기준: 루트가 하나가 아님, 비투사, 어절 위치가 문장 form과 어긋남, Kiwi 형태소를 어절에
배정할 수 없음(토큰이 어절 경계를 넘음 등), dev/test 문장 중 train에 똑같은 문장(form)이 있음.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcfg.korean import (  # noqa: E402
    assign_morphemes, dp_files, dp_to_tree, is_projective, iter_dp_sentences,
    single_root, split_of,
)
from pcfg.tree import to_bracket  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/treebank"))
    ap.add_argument("--batch", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="문장 수 제한(시험용, 0=전체)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from kiwipiepy import Kiwi
    kiwi = Kiwi(num_workers=args.workers)

    reasons: Counter = Counter()
    by_split: dict = defaultdict(Counter)
    by_source: dict = defaultdict(Counter)
    train_forms: set = set()
    held: dict = {"dev": [], "test": []}
    f_train = open(args.out / "train.trees", "w", encoding="utf-8")
    m_train = open(args.out / "train.meta.jsonl", "w", encoding="utf-8")
    t0 = time.time()

    def flush(batch: list) -> None:
        results = kiwi.tokenize([s.form for s in batch])
        for s, toks in zip(batch, results):
            morphs = assign_morphemes(s.words, toks)
            if morphs is None:
                reasons["morph_assign_fail"] += 1
                continue
            tree = dp_to_tree(s.heads, s.labels, morphs)
            n_m = sum(len(m) for m in morphs)
            split = split_of(s.doc_id)
            meta = {"sent_id": s.sent_id, "doc_id": s.doc_id, "source": s.source, "genre": s.genre,
                    "n_eojeol": len(s.heads), "n_morph": n_m}
            line = to_bracket(tree)
            if split == "train":
                f_train.write(line + "\n")
                m_train.write(json.dumps(meta, ensure_ascii=False) + "\n")
                train_forms.add(s.form)
                by_split["train"]["sent"] += 1
                by_split["train"]["morph"] += n_m
            else:
                held[split].append((s.form, line, meta))
            reasons["ok"] += 1
            by_source[s.source]["ok"] += 1

    batch: list = []
    seen = 0
    for s in iter_dp_sentences(dp_files(args.data)):
        seen += 1
        by_source[s.source]["seen"] += 1
        if not single_root(s.heads):
            reasons["not_single_root"] += 1
        elif not is_projective(s.heads):
            reasons["non_projective"] += 1
        elif any(s.form[b:e] != w for (b, e), w in zip(s.words, s.word_forms)):
            reasons["offset_mismatch"] += 1
        else:
            batch.append(s)
            if len(batch) >= args.batch:
                flush(batch)
                batch = []
        if args.limit and seen >= args.limit:
            break
    if batch:
        flush(batch)
    f_train.close()
    m_train.close()

    for split in ("dev", "test"):
        with open(args.out / f"{split}.trees", "w", encoding="utf-8") as ft, \
                open(args.out / f"{split}.meta.jsonl", "w", encoding="utf-8") as fm:
            for form, line, meta in held[split]:
                if form in train_forms:
                    reasons[f"{split}_dup_of_train"] += 1
                    continue
                ft.write(line + "\n")
                fm.write(json.dumps(meta, ensure_ascii=False) + "\n")
                by_split[split]["sent"] += 1
                by_split[split]["morph"] += meta["n_morph"]

    stats = {"seen": seen, "reasons": dict(reasons),
             "split": {k: dict(v) for k, v in sorted(by_split.items())},
             "source": {k: dict(v) for k, v in sorted(by_source.items())},
             "seconds": round(time.time() - t0, 1)}
    (args.out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
