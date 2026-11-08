#!/usr/bin/env python
"""
Upload the dataset to the Hugging Face Hub so teammates can load it with one line.

One-time setup
    pip install -r requirements.txt
    hf auth login            # paste a token with WRITE access (huggingface.co/settings/tokens)
                             # (older huggingface_hub versions: `huggingface-cli login`)

Upload (private by default)
    python scripts/upload_to_hf.py --repo YOUR_TEAM_ORG/shapes-vqa-distractors --data data/shapes_vqa_dataset

Check everything without uploading
    python scripts/upload_to_hf.py --repo YOUR_TEAM_ORG/shapes-vqa-distractors --data data/shapes_vqa_dataset --dry-run

What gets uploaded
    * train / validation / test splits as Parquet (images embedded) -> `load_dataset(repo)`
    * shapes_vqa_dataset.zip  (the raw images/ folder + metadata.csv, for people who prefer files)
    * preview.png, dataset_info.json and a dataset card (README) with usage instructions
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

# The newer "Xet" upload path stalled at a few KB/s on our connection; classic LFS ran at ~2 MB/s.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

INT_COLUMNS = {"scene_id", "difficulty", "count_circle", "count_triangle", "count_square",
               "count_star", "count_total", "num_distractors", "target_box", "num_crossings"}

CARD = """
# Shapes-VQA Distractors

A synthetic VQA dataset for testing whether **visual distractors** hurt a vision-language
model's ability to count and categorise named shapes.

Every **scene** (a random layout of circles, triangles, squares and stars) is drawn in
**4 versions**. The named shapes are pixel-identical in all 4, so **the correct answer is the
same for all 4 versions** - only the distractors change.

| difficulty | version | what is added |
|---|---|---|
| 0 | `clean` | nothing - only the named shapes |
| 1 | `squiggles` | open scribble lines (waves, zigzags, meanders) |
| 2 | `blobs` | irregular closed blobs that resemble no named shape |
| 3 | `lookalikes` | broken (gap in outline) or wavy-sided versions of named shapes |

![preview](https://huggingface.co/datasets/{repo}/resolve/main/preview.png)

**Size:** {num_scenes} scenes x 4 versions = {num_images} images (512x512 PNG).
**Splits (by scene, so all 4 versions of a scene are in the same split):** {splits}

## Load it

```python
from datasets import load_dataset

ds = load_dataset("{repo}")        # private repo? run `hf auth login` first
ex = ds["test"][0]
ex["image"]          # PIL image
ex["question"]       # the prompt (identical for every image)
ex["answer"]         # e.g. "circle: 1, triangle: 0, square: 2, star: 1"
ex["version"]        # clean / squiggles / blobs / lookalikes
```

Prefer plain files? Download `shapes_vqa_dataset.zip` from the *Files* tab (images/ + metadata.csv).

## Question and answer format

Every image uses the same question:

> {question}

Answers always list all 4 classes in this order, including zeros:
`circle: N, triangle: N, square: N, star: N`.

## Columns

| column | meaning |
|---|---|
| `image` | the image |
| `image_id` | e.g. `scene_0042_blobs` - use this as the key for predictions |
| `scene_id` | scene number; the 4 versions of a scene share it (use it for paired comparisons) |
| `version`, `difficulty` | which distractor set (0 = clean ... 3 = lookalikes) |
| `split` | train / validation / test |
| `question`, `answer` | the VQA pair |
| `count_circle` ... `count_star`, `count_total` | ground-truth counts as integers |
| `num_distractors` | 0 for clean, otherwise 2, 4 or 6 |
| `objects` | JSON string: every object's role (target/distractor), shape, colour, center, radius, bbox |
| `file_name` | path of the image inside the zip |

## Scoring

Use `vqa_scoring.py` from the project repo: save `image_id,prediction` to a CSV and run
`python vqa_scoring.py --predictions preds.csv --hf-repo {repo}`. It reports exact-match,
per-class accuracy, over-counting rate per version, and a paired McNemar test against the
clean version.
"""

CARD_HARD = """
# Shapes-VQA Hard

The harder sibling of `atml-shapes-vqa/shapes-vqa-distractors`. The model must count only the
named shapes (circle, triangle, square, star) **of one colour**, which is named in the question.
The traps are built to exploit two known VLM weaknesses: **colour-shape binding** and
**being misled by text written in the image**.

Every **scene** is drawn in **10 versions**. The target shapes are pixel-identical in all 10,
so **the correct answer is the same for all 10 versions** - only the traps change.

| difficulty | version | what is added to the target-colour shapes |
|---|---|---|
| 0 | `clean` | nothing |
| 1 | `wrong_color` | perfect named shapes in **other** colours |
| 2 | `lookalikes` | broken or wavy shapes in the **target** colour |
| 3 | `text_traps` | misleading words and numbers, e.g. `star: 4`, `red circle`, `5` (count claims are always wrong) |
| 4 | `everything` | all traps from versions 1, 2 and 3 together |
| 5 | `squiggles_2` | scribbles with **2** curves each |
| 6 | `squiggles_4` | the same scribble spots, **4** curves each |
| 7 | `squiggles_8` | the same scribble spots, **8** curves each |
| 8 | `squiggles_16` | the same scribble spots, **16** curves each |
| 9 | `other_shapes` | the same spots, but **nameable shapes that are none of the four types**: heart, moon, cross, lightning bolt |

Versions 1-3 each add the same number of traps (2, 3 or 4); `everything` has exactly their union (6, 9 or 12).
The four `squiggles_*` versions put 2, 3 or 4 scribbles in **the same spots, sizes and colours**
(half of them in the target colour); only the **number of curves** changes, so they measure accuracy vs.
squiggle complexity on the same scenes. A curve is one bend of 30-120 degrees, so total bending roughly
doubles per level. Each squiggle in `objects` has `curves` and `turning_deg` (total bending).
`other_shapes` draws complete outlines of nameable shapes in exactly the squiggle spots, sizes and colours
(different kinds within an image), so comparing it with `squiggles_*` isolates **nameable vs. nameless clutter**.
None of these shapes is or contains a circle, triangle, square or star, so they are never counted.
Colours: red, blue, green, black (120-135 scenes each). Text uses Pillow's built-in open font.
Versions 0-4 are unchanged since v1.0, versions 5-8 since v1.1 (`other_shapes` was added in v1.2).

![preview](https://huggingface.co/datasets/{repo}/resolve/main/preview.png)

**Size:** {num_scenes} scenes x 10 versions = {num_images} images (512x512 PNG).
**Splits (by scene, so all 10 versions of a scene are in the same split):** {splits}

## Load it

```python
from datasets import load_dataset

ds = load_dataset("{repo}")        # private repo? run `hf auth login` first
ex = ds["test"][0]
ex["image"]          # PIL image
ex["target_color"]   # red / blue / green / black
ex["question"]       # the prompt (names the target colour)
ex["answer"]         # e.g. "circle: 1, triangle: 0, square: 2, star: 1"
ex["version"]        # clean / wrong_color / lookalikes / text_traps / everything / squiggles_2 ... squiggles_16 / other_shapes
```

Prefer plain files? Download `{zip_name}` from the *Files* tab (images/ + metadata.csv).

## Question and answer format

The question is the same for all images except for the colour:

> {question}

Answers always list all 4 classes in this order, including zeros:
`circle: N, triangle: N, square: N, star: N`.

## Columns

Same as the first dataset, plus `target_color`. In `objects`, traps have `role = distractor`;
wrong-colour traps have a named `shape`, lookalikes have `shape = lookalike` (+ `lookalike_of`,
`subtype` gap/wavy), text traps have `shape = text` with the written `text` and `subtype`
(`class_word`, `false_count`, `number`, `color_phrase`), squiggles have `shape = squiggle`, and the
`other_shapes` traps have `shape` = `heart` / `moon` / `cross` / `bolt` with `subtype = other_shape`.

## Scoring

Download `vqa_scoring.py` from the *Files* tab and run
`python vqa_scoring.py --predictions preds.csv --split test`. It detects this dataset from the
`hard_...` image ids and fetches the correct answers automatically.
"""

CARD_OBJECTS = """
# Objects-VQA

Count everyday objects - **apples, cats, dogs and cars** - among look-alike distractors.
Same design as `atml-shapes-vqa/shapes-vqa-distractors`, but with real-world categories, so the
distractors are close both visually *and* in meaning (a tiger is not a cat, a tomato is not an apple).

Every **scene** is drawn in **5 versions**. The target objects are pixel-identical in all 5,
so **the correct answer is the same for all 5 versions** - only the distractors change.

| difficulty | version | what is added to the targets |
|---|---|---|
| 0 | `clean` | nothing |
| 1 | `random_objects` | unrelated everyday objects (tree, key, guitar, gift, ...) |
| 2 | `lookalikes` | close relatives: tomato / peach / cherries (apple), tiger / lion / hamster (cat), wolf / fox / bear (dog), bus / truck / tractor (car) |
| 3 | `text_traps` | misleading words and numbers, e.g. `cat: 4`, `3 dogs`, `5` (count claims are always wrong) |
| 4 | `everything` | all distractors from versions 1, 2 and 3 together |

Versions 1-3 each add the same number of distractors (2, 3 or 4); `everything` has exactly their union (6, 9 or 12).
Objects are 48-72 px, rotated up to 15 degrees and randomly mirrored.

![preview](https://huggingface.co/datasets/{repo}/resolve/main/preview.png)

**Size:** {num_scenes} scenes x 5 versions = {num_images} images (512x512 PNG).
**Splits (by scene, so all 5 versions of a scene are in the same split):** {splits}

## Load it

```python
from datasets import load_dataset

ds = load_dataset("{repo}")        # private repo? run `hf auth login` first
ex = ds["test"][0]
ex["image"]          # PIL image
ex["question"]       # the prompt (identical for every image)
ex["answer"]         # e.g. "apple: 2, cat: 1, dog: 0, car: 3"
ex["version"]        # clean / random_objects / lookalikes / text_traps / everything
```

Prefer plain files? Download `{zip_name}` from the *Files* tab (images/ + metadata.csv).

## Question and answer format

Every image uses the same question:

> {question}

Answers always list all 4 classes in this order, including zeros:
`apple: N, cat: N, dog: N, car: N`.

## Columns

Same layout as the shapes datasets: `image_id`, `scene_id`, `version`, `difficulty`, `split`,
`question`, `answer`, `count_apple`, `count_cat`, `count_dog`, `count_car`, `count_total`,
`num_distractors`, `objects`. In `objects`, each entry has `role` (target / distractor), `shape`
(the object, e.g. `tomato`), `subtype` (`random`, `lookalike`, or the text-trap kind),
`lookalike_of`, `text`, `center`, `radius`, `rotation_deg`, `flipped`, `size_px` and `bbox`.

## Scoring

Download `vqa_scoring.py` from the *Files* tab and run
`python vqa_scoring.py --predictions preds.csv --split test`. It detects this dataset from the
`obj_...` image ids and fetches the correct answers automatically.

## Credits

Object artwork: [Noto Color Emoji](https://github.com/googlefonts/noto-emoji) by Google, used under
the SIL Open Font License 1.1. Text uses Pillow's built-in open font.
"""

CARD_LINES = """
# Lines-VQA (follow the line)

Follow one line while other lines cross it. Four dots labelled **A-D** sit on the left, eight
numbered **boxes 1-8** on the right, and smooth lines run from dots to boxes. The question names one
dot; the answer is the box its line ends in. Each crossing is a point where the line the question
asks about (the *target*) and another line are tangled together, so the number of crossings is a
direct dial for how much the task-relevant line is entangled with distractors.

Every **scene** is drawn in **6 versions**. The target line, the dots and the boxes are
pixel-identical in all 6, so **the correct answer is the same for all 6 versions** - only the other
lines change.

| difficulty | version | what is added to the target line |
|---|---|---|
| 0 | `clean` | nothing (all dots and boxes are always drawn) |
| 1 | `crossings_0` | lines from the 3 other dots; none of them crosses the target |
| 2 | `crossings_2` | lines from the 3 other dots that cross the target **2** times in total |
| 3 | `crossings_4` | ... **4** times |
| 4 | `crossings_8` | ... **8** times |
| 5 | `crossings_4_colored` | the same lines as `crossings_4`, but the 3 other lines are coloured (target stays black) |

`clean` vs `crossings_0` measures plain clutter; `crossings_0` -> `_8` measures entanglement; and
`crossings_4` vs `crossings_4_colored` (same geometry, only colour differs) tests whether making the
target easy to separate brings accuracy back.

Rules that keep the answer obvious to a human: every line runs left to right and never doubles back;
every crossing is a clean X of at least 45 degrees and at least 20 px from any other crossing; lines
never touch or run within 14 px of each other except where they cross; the other lines never cross
each other (so every crossing is on the target); each box receives at most one line; all lines have
the same colour and width (except in the coloured version), so nothing shows which line is on top.
The crossing counts are even because of topology (a target from the top dot to the top box has every
other line below it at both ends, so each one crosses it an even number of times). The other lines may
end in different boxes in different versions; the target never changes.

![preview](https://huggingface.co/datasets/{repo}/resolve/main/preview.png)

**Size:** {num_scenes} scenes x 6 versions = {num_images} images (512x512 PNG).
**Splits (by scene, so all 6 versions of a scene are in the same split):** {splits}

## Load it

```python
from datasets import load_dataset

ds = load_dataset("{repo}")        # private repo? run `hf auth login` first
ex = ds["test"][0]
ex["image"]          # PIL image
ex["target_dot"]     # A / B / C / D
ex["question"]       # the prompt (names the dot)
ex["answer"]         # e.g. "box: 5"
ex["version"]        # clean / crossings_0 / crossings_2 / crossings_4 / crossings_8 / crossings_4_colored
```

Prefer plain files? Download `{zip_name}` from the *Files* tab (images/ + metadata.csv).

## Question and answer format

The question is the same for all images except for the dot letter:

> {question}

Answers look like `box: 5`.

## Columns

| column | meaning |
|---|---|
| `image_id` | e.g. `line_0042_crossings_4` - use this as the key for predictions |
| `scene_id`, `version`, `difficulty`, `split`, `question`, `answer` | as in the other datasets |
| `target_dot`, `target_box` | where the target line starts and ends (`target_box` is the answer) |
| `num_crossings` | how often other lines cross the target (0 for clean and crossings_0) |
| `num_distractors` | other lines (0 for clean, otherwise 3) |
| `distractor_boxes` | boxes where the other lines end, e.g. `2,6,7` |
| `crossed_boxes` | the subset of those whose line crosses the target |
| `rank_kept` | `yes` if the target ends at the same rank among the line ends as it starts among the dots (so "the line goes straight across" would also be right), `no` otherwise, empty for clean |
| `objects` | JSON: the dots and boxes (`role = layout`) and every line (`role = target / distractor`) with `from_dot`, `to_box`, `color`, `points` and `bbox`; other lines also list their `crossings` with the target (`at`, `angle_deg`) |

A line's `points` are its control points; the drawn curve is 4 rounds of Chaikin smoothing of them
(`chaikin` in `generate_dataset.py`), which stays within a few pixels of the control polygon.

**About `rank_kept`:** with an even number of crossings the target's rank among the line ends can
only change by an even amount, and a target from a top dot to a top box cannot change rank at all, so
it is kept in about 70% of the crossing images (always in `crossings_0`). Compare accuracy on
`rank_kept = yes` and `no`: a model that really traces should do equally well on both.

## Scoring

Download `vqa_scoring.py` from the *Files* tab and run
`python vqa_scoring.py --predictions preds.csv --split test`. It detects this dataset from the
`line_...` image ids and reports per version: accuracy, how often the model answered the box of a
line that crosses the target (`jumped`), accuracy split by `rank_kept`, and a paired McNemar test
against `clean`. It accepts answers like `box: 5`, `Box 5`, `box five` or just `5`.
"""

CARDS = {"shapes-vqa-distractors": CARD, "shapes-vqa-hard": CARD_HARD, "objects-vqa": CARD_OBJECTS,
         "lines-vqa": CARD_LINES}


def build_dataset_dict(root: Path):
    from datasets import Dataset, DatasetDict, Features, Image, Value

    with open(root / "metadata.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    columns = list(rows[0])
    features = Features({"image": Image(),
                         **{c: Value("int32") if c in INT_COLUMNS else Value("string") for c in columns}})
    splits = {}
    for split in ("train", "validation", "test"):
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        data = {"image": [str(root / r["file_name"]) for r in sub]}
        for c in columns:
            data[c] = [int(r[c]) if c in INT_COLUMNS else r[c] for r in sub]
        splits[split] = Dataset.from_dict(data, features=features)
    return DatasetDict(splits)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="e.g. your-team-org/shapes-vqa-distractors")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data" / "shapes_vqa_dataset"),
                    help="local dataset folder")
    ap.add_argument("--public", action="store_true", help="make the dataset public (default: private)")
    ap.add_argument("--dry-run", action="store_true", help="build and check, but do not upload")
    args = ap.parse_args()

    root = Path(args.data)
    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    if args.repo.split("/")[-1] != info["name"]:
        raise SystemExit(f"Safety check: folder '{root}' holds the dataset '{info['name']}', but the repo is "
                         f"'{args.repo}'. Pass the matching --data folder (or a repo named .../{info['name']}).")
    if not (root / "preview.png").exists():
        raise SystemExit("preview.png missing - run `python scripts/validate_dataset.py` first (it must pass).")

    dd = build_dataset_dict(root)
    print(dd)
    ex = dd["train"][0]
    print("first example:", {k: (v if k != "image" else f"<PIL {v.size} {v.mode}>") for k, v in ex.items() if k != "objects"})
    assert sum(len(d) for d in dd.values()) == info["num_images"], "row count does not match dataset_info.json"

    zip_name = f"{root.resolve().name}.zip"
    question = info.get("question") or info["question_template"].replace("{color}", "<colour>").replace("{dot}", "<letter>")
    card_text = CARDS[info["name"]].format(
        repo=args.repo, num_scenes=info["num_scenes"], num_images=info["num_images"], zip_name=zip_name,
        splits=", ".join(f"{k} = {v} images" for k, v in info["images_per_split"].items()), question=question)
    if args.dry_run:
        print("\nDry run OK - nothing uploaded.")
        return

    from huggingface_hub import HfApi

    print(f"\nUploading to https://huggingface.co/datasets/{args.repo} (private={not args.public}) ...")
    dd.push_to_hub(args.repo, private=not args.public)

    api = HfApi()
    with tempfile.TemporaryDirectory() as tmp:
        root_abs = root.resolve()
        zip_path = shutil.make_archive(str(Path(tmp) / root_abs.name), "zip", root_dir=root_abs.parent, base_dir=root_abs.name)
        scorer = Path(__file__).with_name("vqa_scoring.py")
        for local, remote in [(zip_path, zip_name), (root / "preview.png", "preview.png"),
                              (root / "dataset_info.json", "dataset_info.json"),
                              (root / "metadata.csv", "metadata.csv"),           # used by vqa_scoring.py
                              (scorer, "vqa_scoring.py")]:
            print(f"  uploading {remote} ...")
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=args.repo, repo_type="dataset")

    push_card(args.repo, card_text)
    print(f"Done: https://huggingface.co/datasets/{args.repo}")


def push_card(repo: str, body: str):
    """Replace the card text but keep the YAML header that push_to_hub wrote.
    Written as raw UTF-8 bytes with LF line endings: on Windows, DatasetCard.push_to_hub
    can produce '\\r\\r\\n' line breaks, which makes the Hub ignore the YAML header."""
    import re
    from huggingface_hub import HfApi, hf_hub_download

    path = hf_hub_download(repo, "README.md", repo_type="dataset", force_download=True)
    raw = Path(path).read_bytes().decode("utf-8")
    raw = re.sub(r"\r+\n", "\n", raw).replace("\r", "")
    m = re.match(r"---\n(.*?)\n---\n", raw, re.S)
    if not m:
        raise SystemExit("could not find the YAML header in README.md")
    content = f"---\n{m.group(1)}\n---\n\n{body.strip()}\n"
    HfApi().upload_file(path_or_fileobj=content.encode("utf-8"), path_in_repo="README.md", repo_id=repo,
                        repo_type="dataset", commit_message="Update dataset card")


if __name__ == "__main__":
    main()
