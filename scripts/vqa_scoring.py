#!/usr/bin/env python
"""
Score a VLM's answers on the Shapes-VQA distractor dataset.

1) Run your model on the images and save a CSV with two columns:
       image_id,prediction
       scene_0001_clean,"circle: 1, triangle: 0, square: 2, star: 1"
       ...
2) Score it:
       python vqa_scoring.py --predictions my_model_preds.csv
   (uses shapes_vqa_dataset/metadata.csv if it exists, otherwise downloads the ground
    truth from the team's Hugging Face repo; --hf-repo ORG/NAME picks another repo)
       python vqa_scoring.py --predictions preds.csv --split test --out results.csv

You can also import the helpers in your own code:
       from vqa_scoring import parse_answer, score_answer
       from vqa_scoring import parse_box, score_box          # Lines-VQA ("box: 5" answers)
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

CLASSES = ["circle", "triangle", "square", "star"]   # default; the scorer reads the real ones from the metadata
# image_id prefix -> (local metadata.csv, team Hugging Face repo). The right one is picked
# automatically from your predictions, so the same command works for every dataset.
DATASETS = {
    "scene": ("shapes_vqa_dataset/metadata.csv", "atml-shapes-vqa/shapes-vqa-distractors"),
    "hard": ("shapes_vqa_hard/metadata.csv", "atml-shapes-vqa/shapes-vqa-hard"),
    "obj": ("objects_vqa/metadata.csv", "atml-shapes-vqa/objects-vqa"),
    "line": ("lines_vqa/metadata.csv", "atml-shapes-vqa/lines-vqa"),
}

_WORDS = {"zero": 0, "no": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_NUM = r"(\d+|" + "|".join(_WORDS) + r")"


def _to_int(tok: str) -> int:
    return int(tok) if tok.isdigit() else _WORDS[tok]


def parse_answer(text: str, classes=None) -> dict:
    """Extract per-class counts from free text. Missing classes are None.

    Handles "circle: 2", "Circles = 2", "circle - 2", "**circle**: 2" and "2 circles"/"no stars".
    `classes` defaults to the shape classes; pass e.g. ["apple", "cat", "dog", "car"] for Objects-VQA.
    """
    t = re.sub(r"[*_`]", "", str(text).lower())
    out = {}
    for c in classes or CLASSES:
        name = re.escape(c)
        m = re.search(rf"\b{name}s?\b\s*[:=\-]?\s*{_NUM}\b", t)        # "circle: 2"
        if m is None:
            m = re.search(rf"\b{_NUM}\s+{name}s?\b", t)               # "2 circles"
        out[c] = _to_int(m.group(1)) if m else None
    return out


def classes_from_row(row: dict) -> list:
    return [k[len("count_"):] for k in row if k.startswith("count_") and k != "count_total"]


def counts_from_row(row: dict) -> dict:
    return {c: int(row[f"count_{c}"]) for c in classes_from_row(row)}


def score_answer(prediction: str, truth: dict) -> dict:
    """Compare a prediction string with the true counts ({"circle": 1, ...} or {"apple": 2, ...})."""
    classes = list(truth)
    pred = parse_answer(prediction, classes)
    parsed_all = all(v is not None for v in pred.values())
    filled = {c: (pred[c] if pred[c] is not None else 0) for c in classes}
    true_total, pred_total = sum(truth.values()), sum(filled.values())
    res = {
        "parsed": parsed_all,
        "exact_match": parsed_all and all(pred[c] == truth[c] for c in classes),
        "total_correct": parsed_all and pred_total == true_total,
        "abs_error": sum(abs(filled[c] - truth[c]) for c in classes),
        "total_bias": pred_total - true_total,           # > 0 means the model over-counted
    }
    for c in classes:
        res[f"{c}_correct"] = pred[c] == truth[c]
        res[f"pred_{c}"] = pred[c]
    return res


def parse_box(text: str):
    """Lines-VQA: the box number in an answer ("box: 5", "Box 5", "box #5", "box five"), else the only
    number in the text ("5", "It ends at 5"). None if there is no box number or it is ambiguous."""
    t = re.sub(r"[*_`]", "", str(text).lower())
    m = re.search(rf"\bbox(?:es)?\b\s*(?:number|no\.|#)?\s*[:=\-]?\s*{_NUM}\b", t)
    if m:
        return _to_int(m.group(1))
    nums = set(re.findall(r"\b\d+\b", t))
    return int(nums.pop()) if len(nums) == 1 else None


def _boxes(field: str) -> set:
    return {int(b) for b in str(field).split(",") if b.strip()}


def score_box(prediction: str, row: dict) -> dict:
    """Lines-VQA: compare a prediction with a metadata row (needs target_box, distractor_boxes, crossed_boxes)."""
    pred, truth = parse_box(prediction), int(row["target_box"])
    others, crossed = _boxes(row.get("distractor_boxes", "")), _boxes(row.get("crossed_boxes", ""))
    return {
        "parsed": pred is not None,
        "exact_match": pred == truth,
        "pred_box": pred,
        "other_line": pred in others,          # ended at a box where another line ends
        "jumped": pred in crossed,             # ... a line that crosses the target (switched lines at a crossing)
        "no_line": pred is not None and pred != truth and pred not in others,
    }


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _pct(x):
    return f"{100 * x:5.1f}%"


def _print_count_table(by_version, versions, vw, classes):
    header = (f"{'version':<{vw}} {'n':>4} {'exact':>7} {'total':>7} "
              + " ".join(f"{c:>8}" for c in classes) + f" {'abs_err':>8} {'overcnt':>8} {'parsed':>7}")
    print(header)
    print("-" * len(header))
    for v in versions:
        g = by_version.get(v)
        if not g:
            continue
        n = len(g)
        mean = lambda key: sum(bool(x[key]) for x in g) / n
        print(f"{v:<{vw}} {n:>4} {_pct(mean('exact_match')):>7} {_pct(mean('total_correct')):>7} "
              + " ".join(f"{_pct(mean(f'{c}_correct')):>8}" for c in classes)
              + f" {sum(x['abs_error'] for x in g) / n:>8.2f}"
              + f" {_pct(sum(x['total_bias'] > 0 for x in g) / n):>8} {_pct(mean('parsed')):>7}")
    print(f"\nexact = all {len(classes)} counts right | total = total count right | abs_err = mean summed count error"
          "\novercnt = % of images where the model counted MORE targets than there are")


def _print_trace_table(by_version, versions, vw):
    header = (f"{'version':<{vw}} {'n':>4} {'acc':>7} {'jumped':>7} {'other':>7} {'no_line':>8} {'parsed':>7}"
              f"   {'acc if rank kept':>17} {'acc if changed':>15}")
    print(header)
    print("-" * len(header))
    for v in versions:
        g = by_version.get(v)
        if not g:
            continue
        n = len(g)
        mean = lambda key, rows=g: sum(bool(x[key]) for x in rows) / max(1, len(rows))
        kept = [x for x in g if x["rank_kept"] == "yes"]
        changed = [x for x in g if x["rank_kept"] == "no"]
        split = lambda rows: f"{_pct(mean('exact_match', rows))} (n={len(rows)})" if rows else "-"
        print(f"{v:<{vw}} {n:>4} {_pct(mean('exact_match')):>7} {_pct(mean('jumped')):>7} {_pct(mean('other_line')):>7}"
              f" {_pct(mean('no_line')):>8} {_pct(mean('parsed')):>7}   {split(kept):>17} {split(changed):>15}")
    print("\nacc = right box | jumped = answered the box of a line that crosses the target (switched lines at a crossing)"
          "\nother = box of any other line | no_line = a box no line ends in"
          "\nrank kept = the target ends at the same rank among line ends as it starts among the dots, so")
    print("            'the line goes straight across' would also be right; compare both columns")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, help="CSV with columns image_id,prediction")
    ap.add_argument("--metadata", help="local metadata.csv (default: picked from your image_ids)")
    ap.add_argument("--hf-repo", help="instead of --metadata: fetch metadata.csv from this Hugging Face dataset repo")
    ap.add_argument("--split", choices=["train", "validation", "test", "all"], default="all")
    ap.add_argument("--out", help="optional: save per-image scores to this CSV")
    args = ap.parse_args()

    with open(args.predictions, newline="", encoding="utf-8") as f:
        preds = {r["image_id"]: r["prediction"] for r in csv.DictReader(f)}
    if not preds:
        raise SystemExit("the predictions file is empty")

    if not args.metadata and not args.hf_repo:
        prefix = next(iter(preds)).split("_")[0]
        local, repo = DATASETS.get(prefix, DATASETS["scene"])
        # an unzipped download next to you, or the repo's data/ folder when run from the dataset repo
        found = [p for p in (Path(local), Path(__file__).resolve().parent.parent / "data" / local) if p.exists()]
        if found:
            args.metadata = str(found[0])
        else:
            args.hf_repo = repo
            print(f"Reading the correct answers from Hugging Face ({repo})")
    if args.hf_repo:
        from huggingface_hub import hf_hub_download
        args.metadata = hf_hub_download(args.hf_repo, "metadata.csv", repo_type="dataset")

    with open(args.metadata, newline="", encoding="utf-8") as f:
        meta = {r["image_id"]: r for r in csv.DictReader(f)}
    versions = [v for _, v in sorted({(int(r["difficulty"]), r["version"]) for r in meta.values()})]

    unknown = set(preds) - set(meta)
    if unknown:
        raise SystemExit(f"{len(unknown)} image_ids in predictions are not in the metadata, e.g. {sorted(unknown)[:3]}")
    wanted = [r for r in meta.values() if args.split == "all" or r["split"] == args.split]
    missing = [r["image_id"] for r in wanted if r["image_id"] not in preds]
    if missing:
        print(f"WARNING: {len(missing)} images in split '{args.split}' have no prediction (skipped), e.g. {missing[:3]}")

    trace = "target_box" in next(iter(meta.values()))          # Lines-VQA: one box number per image
    scored = []
    for r in wanted:
        if r["image_id"] in preds:
            s = score_box(preds[r["image_id"]], r) if trace else score_answer(preds[r["image_id"]], counts_from_row(r))
            extra = {"rank_kept": r["rank_kept"]} if trace else {}
            scored.append({"image_id": r["image_id"], "scene_id": r["scene_id"], "version": r["version"],
                           "answer": r["answer"], "prediction": preds[r["image_id"]], **extra, **s})
    if not scored:
        raise SystemExit("nothing to score")

    by_version = defaultdict(list)
    for s in scored:
        by_version[s["version"]].append(s)

    vw = max(12, max(len(v) for v in versions) + 1)
    print(f"\nResults on split '{args.split}'  ({len(scored)} images)\n")
    if trace:
        _print_trace_table(by_version, versions, vw)
    else:
        _print_count_table(by_version, versions, vw, classes_from_row(next(iter(meta.values()))))

    # paired comparison against the clean version of the same scene
    exact = {(s["scene_id"], s["version"]): s["exact_match"] for s in scored}
    print("\nPaired vs clean (same scenes, exact match):")
    for v in versions[1:]:
        pairs = [(exact[(sid, "clean")], ok) for (sid, ver), ok in exact.items() if ver == v and (sid, "clean") in exact]
        if not pairs:
            continue
        b = sum(c and not d for c, d in pairs)      # right on clean, wrong with distractors
        c_ = sum(d and not c for c, d in pairs)     # wrong on clean, right with distractors
        acc_clean = sum(c for c, _ in pairs) / len(pairs)
        acc_v = sum(d for _, d in pairs) / len(pairs)
        print(f"  {v:<{vw}} {_pct(acc_clean)} -> {_pct(acc_v)}  (drop {100 * (acc_clean - acc_v):+.1f} pts)"
              f"  broke={b} fixed={c_}  McNemar p={mcnemar_p(b, c_):.3g}  n={len(pairs)}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(scored[0]))
            w.writeheader()
            w.writerows(scored)
        print(f"\nPer-image scores saved to {args.out}")


if __name__ == "__main__":
    main()
