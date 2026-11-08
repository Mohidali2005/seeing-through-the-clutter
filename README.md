# Shapes-VQA Distractors

A synthetic VQA dataset for our ATML project. It tests whether extra visual clutter
(squiggles, blobs, look-alike shapes) makes a vision-language model worse at counting
and categorising named shapes.

![preview](assets/preview.png)

## What's in it

**500 scenes × 4 versions = 2,000 images** (512×512 PNG).

A *scene* is a random layout of 3–7 named shapes (circle, triangle, square, star). Each scene
is drawn 4 times. The named shapes are **pixel-identical** in all 4 versions, so **the correct
answer is the same for all 4**. Only the distractors change:

| difficulty | version | what's added | images |
|---|---|---|---|
| 0 | `clean` | nothing, only the named shapes | 500 |
| 1 | `squiggles` | open scribble lines (waves, zigzags, meanders) | 500 |
| 2 | `blobs` | irregular closed blobs that resemble no named shape | 500 |
| 3 | `lookalikes` | named shapes with a defect: a gap in the outline, or one wavy side | 500 |

In versions 1–3, the distractors sit in **the same spots, with the same sizes and colours**.
Only their form changes. So the difference between `squiggles` and `lookalikes` is purely
*how much the clutter resembles the real shapes*, not how much clutter there is.

Every image has 2, 4 or 6 distractors (0 for clean), balanced across scenes.

**Splits are by scene** (all 4 versions of a scene are always in the same split):

| split | scenes | images |
|---|---|---|
| train | 350 | 1,400 |
| validation | 75 | 300 |
| test | 75 | 300 |

## Second dataset: Shapes-VQA Hard

`atml-shapes-vqa/shapes-vqa-hard` uses the same idea but much nastier traps. The model must
count only the named shapes **of one colour**, named in the question (*"... that are red ..."*).
**500 scenes × 10 versions = 5,000 images**, with the same correct answer in all 10 versions:

| difficulty | version | what's added to the target-colour shapes | trap |
|---|---|---|---|
| 0 | `clean` | nothing | baseline |
| 1 | `wrong_color` | perfect named shapes in *other* colours | colour-shape binding |
| 2 | `lookalikes` | broken / wavy shapes in the *target* colour | shape similarity |
| 3 | `text_traps` | misleading text: `star: 4`, `red circle`, `5` (count claims are always wrong) | reading text |
| 4 | `everything` | all traps from 1, 2 and 3 together | worst case |
| 5–8 | `squiggles_2`, `_4`, `_8`, `_16` | scribbles with 2, 4, 8 or 16 curves | squiggle complexity |
| 9 | `other_shapes` | heart, moon, cross, lightning bolt in the squiggle spots | nameable vs. nameless clutter |

Versions 1–3 each add 2, 3 or 4 traps; `everything` contains exactly their union (6, 9 or 12).
The four squiggle versions use the same spots, sizes and colours for their scribbles; only the
number of curves changes (one curve = one bend of 30–120°), so they give accuracy vs. squiggle
complexity on the same scenes. Each squiggle in `objects` records `curves` and `turning_deg`.
`other_shapes` puts nameable shapes that are none of the four types (and contain none of them)
in exactly those spots, sizes and colours, so it differs from the squiggle versions only in whether
the clutter has a name. In `objects` they have `shape` = `heart` / `moon` / `cross` / `bolt`.
(Squiggle versions were added in v1.1 and `other_shapes` in v1.2; earlier versions are unchanged.)
Colours are red, blue, green and black (120–135 scenes each). Loading, columns and scoring work
the same way. There is one extra column, `target_color`, and image ids start with `hard_`, so
`vqa_scoring.py` finds the right answers by itself. Build it with
`python scripts/generate_hard_dataset.py` and check it with `python scripts/validate_dataset.py --data data/shapes_vqa_hard`.

## Third dataset: Objects-VQA

`atml-shapes-vqa/objects-vqa` moves from shapes to everyday objects. The model counts
**apples, cats, dogs and cars**, and the distractors are close in both looks and meaning.
**500 scenes × 5 versions = 2,500 images**, with the same correct answer in all 5 versions:

| difficulty | version | what's added to the targets |
|---|---|---|
| 0 | `clean` | nothing |
| 1 | `random_objects` | unrelated objects (tree, key, guitar, gift, ...) |
| 2 | `lookalikes` | tomato/peach/cherries, tiger/lion/hamster, wolf/fox/bear, bus/truck/tractor |
| 3 | `text_traps` | misleading text: `cat: 4`, `3 dogs`, `5` (count claims are always wrong) |
| 4 | `everything` | all distractors from 1, 2 and 3 together |

Answers look like `apple: 2, cat: 1, dog: 0, car: 3`. Image ids start with `obj_`, and the same
`vqa_scoring.py` handles it automatically. Objects are drawn with Google's
[Noto Color Emoji](https://github.com/googlefonts/noto-emoji) font (SIL Open Font License 1.1),
kept in `assets/noto-emoji/` together with its licence. Build it with
`python scripts/generate_objects_dataset.py` and check it with `python scripts/validate_dataset.py --data data/objects_vqa`.

## Fourth dataset: Lines-VQA (follow the line)

`atml-shapes-vqa/lines-vqa` tests a different skill: **tracing**. Four dots labelled A-D sit on the
left, eight numbered boxes on the right, and smooth lines run from dots to boxes. The question names
a dot (*"Follow the line that starts at the dot labelled B. Which numbered box does it end in?"*) and
the answer is a box, e.g. `box: 5`. Every crossing is a point where the asked-about (target) line is
tangled with a distractor line, so the number of crossings is a direct dial for entanglement.
**500 scenes × 6 versions = 3,000 images**, with the same target line and answer in all 6:

| difficulty | version | what's added to the target line |
|---|---|---|
| 0 | `clean` | nothing (all dots and boxes are always drawn) |
| 1 | `crossings_0` | lines from the 3 other dots; none crosses the target |
| 2 | `crossings_2` | lines from the 3 other dots that cross the target 2 times in total |
| 3 | `crossings_4` | ... 4 times |
| 4 | `crossings_8` | ... 8 times |
| 5 | `crossings_4_colored` | the lines of `crossings_4`, but the 3 other lines are coloured |

`clean` vs `crossings_0` measures plain clutter, `crossings_0` → `_8` measures entanglement, and
`crossings_4` vs `crossings_4_colored` (identical geometry) tests whether making the target easy to
separate brings accuracy back. Crossing counts are even because of topology: a target from the top
dot to the top box has every other line below it at both ends, so each one crosses it an even
number of times.

Every crossing is a clean X (at least 45°, at least 20 px from other crossings), lines never
touch or run within 14 px of each other except where they cross, the other lines never cross each
other, and all lines share one colour and width, so nothing shows which line is on top.
One known shortcut: in ~70% of the crossing images (and all of `crossings_0`) the target ends at
the same rank among the line ends as it starts among the dots, so "the line goes straight across"
would be right. The `rank_kept` column marks these images, and `vqa_scoring.py` reports accuracy
on both groups; a model that really traces does equally well on both.

Extra columns: `target_dot`, `target_box`, `num_crossings`, `distractor_boxes`, `crossed_boxes`,
`rank_kept`. Each line in `objects` has `from_dot`, `to_box`, `color` and its control `points`
(the drawn curve is `chaikin(points, 4)` from `generate_dataset.py`). For this dataset the scorer
reports accuracy, `jumped` (answered the box of a line that crosses the target) and the
`rank_kept` split. Build it with `python scripts/generate_lines_dataset.py` (~5 min, uses several CPU
cores) and check it with `python scripts/validate_dataset.py --data data/lines_vqa` (~8 min).

## The question and answer

Every image uses the **same question**:

> Count the shapes in this image. Only count complete, regular shapes of these four types:
> circle, triangle, square, star. Do not count anything else, such as scribbles, irregular
> blobs, or broken or wavy versions of these shapes. Answer in exactly this format:
> circle: \<number\>, triangle: \<number\>, square: \<number\>, star: \<number\>

Answers always list all 4 classes in this order, including zeros:

```
circle: 1, triangle: 0, square: 2, star: 1
```

## Load it

### Option A: Hugging Face (recommended)

```python
from datasets import load_dataset

ds = load_dataset("atml-shapes-vqa/shapes-vqa-distractors")   # private repo: run `hf auth login` once
ex = ds["test"][0]
ex["image"]      # PIL image
ex["question"]   # the prompt
ex["answer"]     # "circle: 1, triangle: 0, square: 2, star: 1"
ex["version"]    # clean / squiggles / blobs / lookalikes

lookalikes_test = ds["test"].filter(lambda e: e["version"] == "lookalikes")
```

### Option B: local folder

Download `shapes_vqa_dataset.zip` from the Hugging Face *Files* tab and unzip it, or
rebuild it yourself (see below). Inside:

```
shapes_vqa_dataset/
├── images/                 scene_0001_clean.png, scene_0001_squiggles.png, ...
├── metadata.csv            one row per image (all the columns below)
├── dataset_info.json       generation settings + seed
└── preview.png             a few scenes side by side
```

```python
import csv
from PIL import Image

rows = list(csv.DictReader(open("shapes_vqa_dataset/metadata.csv", encoding="utf-8")))
r = rows[0]
img = Image.open("shapes_vqa_dataset/" + r["file_name"])
print(r["question"], r["answer"])
```

### Columns

| column | meaning |
|---|---|
| `image_id` | e.g. `scene_0042_blobs`. Use this as the key for predictions |
| `file_name` | path of the image, relative to the dataset folder |
| `scene_id` | scene number. The 4 versions of a scene share it (use it for paired comparisons) |
| `version`, `difficulty` | which distractor set (0 = clean … 3 = lookalikes) |
| `split` | train / validation / test |
| `question`, `answer` | the VQA pair |
| `count_circle`, `count_triangle`, `count_square`, `count_star`, `count_total` | ground-truth counts (integers) |
| `num_distractors` | 0 for clean, otherwise 2, 4 or 6 |
| `objects` | JSON list of every object in the image (see below) |

`objects` (parse with `json.loads`) gives, for every object: `role` (`target` or
`distractor`), `shape` (`circle`/`triangle`/`square`/`star` for targets;
`squiggle`/`blob`/`lookalike` for distractors), `subtype` (e.g. `gap`/`wavy` for
lookalikes, `wave`/`zigzag`/`scribble` for squiggles), `lookalike_of` (which named shape a
lookalike imitates), `color`, `center` [x, y], `radius`, `rotation_deg` and a pixel `bbox`
[x0, y0, x1, y1]. This is useful for the disentanglement part, e.g. finding which image
patches contain distractors.

## Run a VLM and score it

1. Run your model and save a CSV with `image_id,prediction`:

```python
import csv
from datasets import load_dataset

ds = load_dataset("atml-shapes-vqa/shapes-vqa-distractors", split="test")

def ask_model(image, question):
    ...   # <- your VLM call here (Qwen2-VL, LLaVA, InternVL, ...), return its text answer

with open("preds_mymodel.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image_id", "prediction"])
    for ex in ds:
        w.writerow([ex["image_id"], ask_model(ex["image"], ex["question"])])
```

2. Score it:

```bash
python scripts/vqa_scoring.py --predictions preds_mymodel.csv --split test --hf-repo atml-shapes-vqa/shapes-vqa-distractors
# or, with the local data/ folder:
python scripts/vqa_scoring.py --predictions preds_mymodel.csv --split test
```

You get a table per version: exact match (all 4 counts right), total-count accuracy,
per-class accuracy, mean count error, **over-count rate** (did the model count distractors
as real shapes?) and parse rate. Below it is a **paired comparison against clean** (same
scenes) with a McNemar significance test.

The parser is forgiving: `circle: 2`, `Circles = 2`, `**Circle**: two`, `2 circles` and
`no stars` all work. To score inside a notebook:

```python
from vqa_scoring import score_answer
score_answer("circle: 1, triangle: 0, square: 2, star: 1",
             {"circle": 1, "triangle": 0, "square": 2, "star": 1})
# {'parsed': True, 'exact_match': True, 'total_correct': True, 'abs_error': 0, 'total_bias': 0, ...}
```

## How we know the labels are right

The labels come straight from how each image was drawn. `validate_dataset.py` then
re-checks everything, including against the **actual pixels**:

- every scene has all 4 versions, with the same split, question and answer
- answer string = count columns = the target objects listed in `objects`
- named shapes are identical in all 4 versions, and distractor spots are identical in versions 1–3
- no two objects touch (at least 16 px apart) and everything is inside the image
- **pixels:** outside the distractor areas, every distracted image is byte-identical to its clean image
- **pixels:** there is no ink anywhere except where a listed object is, and every listed object is visible

It was also tested on purpose-broken data: it catches a wrong answer, a stray dot and a stray line.

Design rules that prevent shortcuts:

- **Same style for everything:** same 4 px outline, same 8-colour palette, same size range for
  targets and distractors. A model can't tell them apart by colour or thickness.
- **No ambiguous classes:** no rectangles, ellipses or pentagons, so a distorted square can't
  be mistaken for a legitimately named shape. Squares are tilted at most 15° (never a "diamond").
- **Clear distractors:** blobs are non-convex and have no rotational symmetry (never a rounded
  square/triangle/star). Scribbles never curl into a C (broken-circle look) or cross
  themselves (digit/letter look). Lookalike defects are always clearly visible: a 15–25% gap,
  or a full wavy side.

**Before big experiments:** have each teammate answer ~20 `lookalikes` images by hand
(`preview.png` is a good start). Humans should get essentially 100%.

## Rebuild, check, upload (maintainers)

```bash
pip install -r requirements.txt

python scripts/generate_dataset.py         # -> data/shapes_vqa_dataset/  (seed 42, ~3-5 min)
python scripts/validate_dataset.py         # must print ALL CHECKS PASSED; writes preview.png

hf auth login                              # once; token with WRITE access
python scripts/upload_to_hf.py --repo atml-shapes-vqa/shapes-vqa-distractors --data data/shapes_vqa_dataset --dry-run
python scripts/upload_to_hf.py --repo atml-shapes-vqa/shapes-vqa-distractors --data data/shapes_vqa_dataset
```

The same seed gives the same dataset. Useful options: `--scenes 20 --out test_run` for a quick
try, `--seed 7` for a fresh dataset, `--overwrite` to replace an existing folder. The scripts
find `data/`, `assets/` and `docs/` on their own, so they work from any folder.

To share on Hugging Face, create an **organization** for the team
(huggingface.co → New → Organization), add your teammates, and upload to `ORG_NAME/...`.
Everyone in the org can then load the private dataset.

## Folder layout

```
ATML-Dataset/
├── scripts/       all Python code (below)
├── data/          the 4 built datasets: shapes_vqa_dataset/, shapes_vqa_hard/, objects_vqa/, lines_vqa/
│                  (not tracked; the copies on Hugging Face are the reference)
├── docs/          ATML_Study3_Methodology_Predictions.pdf (teammate's experiment plan)
│   └── guides/    one teammate guide PDF per dataset
├── results/       Study 3 result reports (model runs on the datasets)
├── src/tangle/    entanglement study code (Part 1), tests/ and kaggle/kernel/ next to it
├── runs/          downloaded Kaggle outputs (not tracked)
├── assets/        noto-emoji/ (font + licence for Objects-VQA), preview.png (for this README)
└── README.md, requirements.txt
```

| script (in `scripts/`) | what it does |
|---|---|
| `generate_dataset.py` | draws all images and writes `metadata.csv` + `dataset_info.json` |
| `generate_hard_dataset.py` | the same for Shapes-VQA Hard (reuses the shape drawing code) |
| `generate_objects_dataset.py` | the same for Objects-VQA (needs `assets/noto-emoji/NotoColorEmoji.ttf`) |
| `generate_lines_dataset.py` | the same for Lines-VQA (follow the line) |
| `make_lines_guide.py` | builds `docs/guides/Lines-VQA_Teammate_Guide.pdf` from `data/lines_vqa/` |
| `make_hard_guide.py` | builds `docs/guides/Shapes-VQA_Hard_Teammate_Guide.pdf` from `data/shapes_vqa_hard/` |
| `validate_dataset.py` | checks labels against metadata and pixels for every dataset; writes `preview.png` |
| `vqa_scoring.py` | parses model answers and reports accuracy per version (every dataset) |
| `upload_to_hf.py` | pushes the dataset (+ zip, preview, card) to the Hugging Face Hub |

## Entanglement study (Part 1)

Does clutter leak into the model's notes about the real objects, and does that leak cause wrong answers?
The code is in `src/tangle/`. It runs Qwen3-VL-4B (and Qwen3-VL-8B as a check) on Kaggle with the same prompt and
settings as Study 3, and keeps the notes of every image square at every layer.

| part | what it does |
|---|---|
| `src/tangle/notes.py` | records the notes of every square for the clean and cluttered version of each scene (leak and probe tests) |
| `src/tangle/swap.py` | answers every cluttered picture again with some squares given their clean notes (swap test) |
| `src/tangle/run_all.py` | runs every step on Kaggle in priority order, one worker per GPU, and can resume |
| `src/tangle/analysis/` | leak, probe, link and swap statistics with scene bootstrap ranges, plus figures |
| `kaggle/kernel/` | the Kaggle script that clones this repo at a pinned commit and starts the run |
| `tests/` | CPU tests with the real processor and a tiny random model |

```bash
pip install -e .
python -X utf8 -m pytest -p no:warnings        # CPU tests
python -m tangle.main                          # checks the local data against the pinned hashes
```
