#!/usr/bin/env python
"""실제 자연어 문장 + 트리뱅크 → 파스 트리 생성 → 시각화.

파이프라인 (plan.md 전체 흐름):
    트리뱅크 ──[규칙 카운팅·정규화(Phase 3)]──> PCFG 문법
    문장 + 문법 ──[CKY/Viterbi(Phase 2)]──> 최적 파스 트리 ──> PNG

사용 예:
    # 기본: 트리뱅크에서 문법 유도 후, 보류(held-out) 실문장 10개를 파싱·시각화
    python -m scripts.parse_and_visualize

    # 직접 문장 입력 (여러 번 지정 가능)
    python -m scripts.parse_and_visualize \
        --sentence "the company said it will buy the stake" \
        --sentence "investors sold the stock sharply"
"""

from __future__ import annotations

import argparse
import os
from typing import List

from pcfg.data import load_treebank_trees
from pcfg.induce import induce_pcfg, parse_sentence, preprocess
from pcfg.eval import evaluate_pair
from pcfg.draw import save_tree_png
from pcfg.tree import to_bracket


def default_sentences(trees, start_idx: int, count: int, max_len: int) -> List[List[str]]:
    """보류 구간에서 실제 자연어 문장(단어열) count개를 고른다."""
    out: List[List[str]] = []
    for t in trees[start_idx:]:
        words = t.leaves()
        if 3 <= len(words) <= max_len:
            out.append(words)
        if len(out) >= count:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="treebank+문장 → 파스 트리 시각화")
    ap.add_argument("--train-size", type=int, default=3500,
                    help="문법 유도에 쓸 트리뱅크 문장 수")
    ap.add_argument("--unk-threshold", type=int, default=1,
                    help="이 빈도 이하 단어를 <UNK>로 치환")
    ap.add_argument("--max-len", type=int, default=12,
                    help="시각화용 문장 최대 길이")
    ap.add_argument("--count", type=int, default=10, help="생성할 트리 수")
    ap.add_argument("--outdir", default="outputs", help="결과 저장 폴더")
    ap.add_argument("--sentence", action="append", default=[],
                    help="직접 파싱할 문장(공백 분리). 여러 번 지정 가능")
    args = ap.parse_args()

    trees_dir = os.path.join(args.outdir, "trees")
    os.makedirs(trees_dir, exist_ok=True)

    print("트리뱅크 로드 중...")
    trees = load_treebank_trees()
    train = trees[:args.train_size]

    print(f"문법 유도 중 (train={len(train)}, unk_threshold={args.unk_threshold})...")
    ind = induce_pcfg(train, unk_threshold=args.unk_threshold)
    print(f"  비단말 {len(ind.grammar.nonterminals)}개, 어휘 {len(ind.vocab)}개")

    # 입력 문장 결정: 사용자 지정 우선, 없으면 보류 실문장
    if args.sentence:
        sentences = [s.split() for s in args.sentence]
        gold_lookup = {}  # 사용자 문장은 골드 없음
    else:
        sentences = default_sentences(trees, args.train_size, args.count, args.max_len)
        # 골드 트리(정답)로 F1도 함께 보고
        gold_lookup = {" ".join(t.leaves()): t
                       for t in trees[args.train_size:]}

    results_path = os.path.join(args.outdir, "parse_results.txt")
    lines: List[str] = []
    lines.append("문장 + 트리뱅크 유도 문법 → CKY/Viterbi 파스 결과\n")
    lines.append(f"train={len(train)}  unk_threshold={args.unk_threshold}  "
                 f"nonterminals={len(ind.grammar.nonterminals)}  "
                 f"vocab={len(ind.vocab)}\n")
    lines.append("=" * 78 + "\n")

    for i, words in enumerate(sentences, 1):
        pred, fell_back = parse_sentence(words, ind)
        sent = " ".join(words)

        f1_str = ""
        key = sent
        if key in gold_lookup:
            gp = preprocess(gold_lookup[key])
            if gp is not None:
                prf = evaluate_pair(gp, pred)
                f1_str = f"  PARSEVAL-F1={prf.f1:.3f}"

        status = "FALLBACK(우분지)" if fell_back else "PCFG-parse"
        title = f"#{i}  [{status}]{f1_str}\n{sent}"
        img_path = os.path.join(trees_dir, f"tree_{i:02d}.png")
        save_tree_png(pred, img_path, title=title)

        lines.append(f"[{i:02d}] {status}{f1_str}\n")
        lines.append(f"  문장: {sent}\n")
        lines.append(f"  트리: {to_bracket(pred)}\n")
        lines.append(f"  이미지: {img_path}\n")
        lines.append("-" * 78 + "\n")
        print(f"  [{i:02d}] {status}{f1_str}  -> {img_path}")

    with open(results_path, "w") as f:
        f.writelines(lines)
    print(f"\n결과 텍스트: {results_path}")
    print(f"트리 이미지: {trees_dir}/tree_01.png ... tree_{len(sentences):02d}.png")


if __name__ == "__main__":
    main()
