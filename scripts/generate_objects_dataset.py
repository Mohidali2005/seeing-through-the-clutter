#!/usr/bin/env python
"""
Generate Objects-VQA: count everyday objects among look-alike distractors.

Task: count the apples, cats, dogs and cars in the image.

Every scene is drawn in 5 versions. The target objects are pixel-identical in all 5,
so the correct answer is the same for all 5 - only the distractors change:

    difficulty 0  clean           only the target objects
    difficulty 1  random_objects  + unrelated objects (tree, key, guitar, ...)
    difficulty 2  lookalikes      + close relatives of the targets (tomato, tiger, wolf, bus, ...)
    difficulty 3  text_traps      + misleading words and numbers ("cat: 4", "3 dogs", "5")
    difficulty 4  everything      + all distractors from versions 1, 2 and 3 together

Objects are drawn with Google's Noto Color Emoji font (SIL Open Font License 1.1).
Get it once (10.7 MB) into assets/noto-emoji/:
    https://github.com/googlefonts/noto-emoji/raw/main/2D/fonts/NotoColorEmoji.ttf

Usage
    python scripts/generate_objects_dataset.py                          # 500 scenes -> data/objects_vqa/
    python scripts/generate_objects_dataset.py --scenes 10 --out test_objects
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATASET_NAME = "objects-vqa"
DATASET_VERSION = "1.0"
ID_PREFIX = "obj"
ROOT = Path(__file__).resolve().parent.parent    # repo root (this file lives in scripts/)
DEFAULT_FONT = ROOT / "assets" / "noto-emoji" / "NotoColorEmoji.ttf"
FONT_URL = "https://github.com/googlefonts/noto-emoji/raw/main/2D/fonts/NotoColorEmoji.ttf"

IMG_SIZE = 512
CLASSES = ["apple", "cat", "dog", "car"]
VERSIONS = ["clean", "random_objects", "lookalikes", "text_traps", "everything"]
TRAP_GROUPS = ["random_objects", "lookalikes", "text_traps"]
TARGET_COUNTS = [3, 4, 5, 6, 7]
GROUP_SIZES = [2, 3, 4]           # distractors per trap type; "everything" has 3x this
MAX_PER_CLASS = 3

# Unicode code points (Noto Color Emoji)
EMOJI = {
    "apple": 0x1F34E, "cat": 0x1F431, "dog": 0x1F436, "car": 0x1F697,
    # look-alikes: clearly NOT the target to a human, but visually and semantically close
    "tomato": 0x1F345, "peach": 0x1F351, "cherries": 0x1F352,
    "tiger": 0x1F42F, "lion": 0x1F981, "hamster": 0x1F439,
    "wolf": 0x1F43A, "fox": 0x1F98A, "bear": 0x1F43B,
    "bus": 0x1F68C, "truck": 0x1F69A, "tractor": 0x1F69C,
    # unrelated objects (nothing round-and-red, no animals, fruit or vehicles)
    "tree": 0x1F333, "house": 0x1F3E0, "key": 0x1F511, "books": 0x1F4DA, "guitar": 0x1F3B8,
    "light bulb": 0x1F4A1, "gift": 0x1F381, "bell": 0x1F514, "anchor": 0x2693, "chair": 0x1FA91,
    "kite": 0x1FA81, "umbrella": 0x1F302,
}
LOOKALIKES = {"apple": ["tomato", "peach", "cherries"], "cat": ["tiger", "lion", "hamster"],
              "dog": ["wolf", "fox", "bear"], "car": ["bus", "truck", "tractor"]}
RANDOM_OBJECTS = ["tree", "house", "key", "books", "guitar", "light bulb", "gift", "bell",
                  "anchor", "chair", "kite", "umbrella"]
PLURAL = {"apple": "apples", "cat": "cats", "dog": "dogs", "car": "cars"}

SIZE_RANGE = (48.0, 72.0)         # longest side of an object, in pixels
MAX_ROTATION = 15.0               # degrees
MIN_GAP = 11.0                    # min empty pixels between two objects
EDGE_MARGIN = 8.0
TEXT_PX = (18, 30)                # font size range for text traps
TEXT_MAX_RADIUS = 58.0
TEXT_COLORS = {"black": (20, 20, 20), "red": (214, 39, 40), "blue": (31, 119, 180), "green": (44, 160, 44)}
SPLIT_PATTERN = ["train", "train", "validation", "train", "train", "test", "train", "train",
                 "train", "validation", "train", "test", "train", "train", "train", "validation",
                 "train", "test", "train", "train"]

QUESTION = (
    "Count the objects in this image. Only count these four kinds of objects: apple, cat, dog, car. "
    "Do not count anything else: similar-looking things (for example other fruit, other animals or "
    "other vehicles) do not count, and neither do any words or numbers written in the image. "
    "Answer in exactly this format: apple: <number>, cat: <number>, dog: <number>, car: <number>"
)

VERSION_DESCRIPTIONS = {
    "clean": "Only the target objects (apple, cat, dog, car).",
    "random_objects": "+ unrelated everyday objects (tree, key, guitar, ...).",
    "lookalikes": "+ close relatives of the targets: tomato/peach/cherries, tiger/lion/hamster, "
                  "wolf/fox/bear, bus/truck/tractor.",
    "text_traps": "+ misleading words and numbers written in the image (count claims are always wrong).",
    "everything": "+ all distractors from random_objects, lookalikes and text_traps together.",
}


def format_answer(counts):
    return ", ".join(f"{c}: {counts[c]}" for c in CLASSES)


# --------------------------------------------------------------------------- #
# Sprites (an RGBA image + the radius of the circle that contains every visible pixel)
# --------------------------------------------------------------------------- #
class Sprites:
    def __init__(self, font_path: Path):
        if not font_path.exists():
            raise SystemExit(f"Emoji font not found: {font_path}\nDownload it (10.7 MB) from:\n  {FONT_URL}")
        self.font = ImageFont.truetype(str(font_path), 109)     # Noto Color Emoji is a 109 px bitmap font
        self._base: dict[str, Image.Image] = {}
        self._text_fonts: dict[int, ImageFont.FreeTypeFont] = {}

    def base(self, name):
        if name not in self._base:
            tile = Image.new("RGBA", (180, 180), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((90, 90), chr(EMOJI[name]), font=self.font, embedded_color=True, anchor="mm")
            box = tile.getbbox()
            if box is None:
                raise RuntimeError(f"emoji '{name}' rendered empty")
            self._base[name] = tile.crop(box)
        return self._base[name]

    def emoji(self, name, rng):
        im = self.base(name).convert("RGBa")                    # premultiplied alpha -> clean edges
        size = float(rng.uniform(*SIZE_RANGE))
        scale = size / max(im.size)
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
        flipped = bool(rng.random() < 0.5)
        if flipped:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        rot = float(rng.uniform(-MAX_ROTATION, MAX_ROTATION))
        im = im.rotate(rot, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0)).convert("RGBA")
        return _finish(im) | dict(rotation_deg=round(rot, 1), flipped=flipped, size_px=round(size, 1))

    def text(self, text, px, rgb):
        f = self._text_fonts.setdefault(px, ImageFont.load_default(size=px))
        x0, y0, x1, y1 = f.getbbox(text)
        im = Image.new("RGBA", (x1 - x0 + 4, y1 - y0 + 4), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((2 - x0, 2 - y0), text, font=f, fill=rgb + (255,))
        return _finish(im) | dict(rotation_deg=None, flipped=None, size_px=px)


def _finish(im):
    im = im.crop(im.getbbox())
    alpha = np.asarray(im)[:, :, 3]
    ys, xs = np.nonzero(alpha > 0)
    radius = float(np.sqrt((xs + 0.5 - im.width / 2) ** 2 + (ys + 0.5 - im.height / 2) ** 2).max()) + 1.0
    return dict(sprite=im, radius=radius)


def make_text_trap(sprites, counts, rng):
    for _ in range(200):
        kind = ["class_word", "false_count", "number"][rng.integers(3)]
        cls = CLASSES[rng.integers(len(CLASSES))]
        if kind == "class_word":
            text = cls if rng.random() < 0.5 else PLURAL[cls]
        elif kind == "false_count":
            n = int(rng.choice([k for k in range(1, 10) if k != counts[cls]]))
            text = f"{cls}: {n}" if rng.random() < 0.5 else (f"{n} {PLURAL[cls]}" if n > 1 else f"1 {cls}")
        else:
            text = str(int(rng.choice([k for k in range(1, 10) if k != sum(counts.values())])))
        color = list(TEXT_COLORS)[rng.integers(len(TEXT_COLORS))]
        for px in range(TEXT_PX[1], TEXT_PX[0] - 1, -1):
            s = sprites.text(text, px, TEXT_COLORS[color])
            if s["radius"] <= TEXT_MAX_RADIUS:
                return s | dict(shape="text", subtype=kind, lookalike_of=None, text=text, color=color)
    raise RuntimeError("could not make a text trap")


# --------------------------------------------------------------------------- #
# Scene
# --------------------------------------------------------------------------- #
def sample_target_classes(n, rng):
    counts = {c: 0 for c in CLASSES}
    out = []
    for _ in range(n):
        options = [c for c in CLASSES if counts[c] < MAX_PER_CLASS]
        c = options[rng.integers(len(options))]
        counts[c] += 1
        out.append(c)
    return out


def place(radii, rng, tries_per_slot=500):
    order = sorted(range(len(radii)), key=lambda i: -radii[i])
    placed = {}
    for i in order:
        r = radii[i]
        lo, hi = r + EDGE_MARGIN, IMG_SIZE - r - EDGE_MARGIN
        for _ in range(tries_per_slot):
            x, y = rng.uniform(lo, hi), rng.uniform(lo, hi)
            # +1.5 px: objects are later snapped to whole pixels (each can move up to ~0.7 px)
            if all(math.hypot(x - px, y - py) >= r + pr + MIN_GAP + 1.5 for px, py, pr in placed.values()):
                placed[i] = (x, y, r)
                break
        else:
            return None
    return [placed[i][:2] for i in range(len(radii))]


def make_scene(scene_id, n_targets, k, split, seed, sprites):
    rng = np.random.default_rng([seed, scene_id, 3])
    names = sample_target_classes(n_targets, rng)
    counts = {c: names.count(c) for c in CLASSES}

    for _attempt in range(200):
        targets = [sprites.emoji(n, rng) | dict(shape=n, subtype=None, lookalike_of=None, text=None, color=None)
                   for n in names]
        groups = {"random_objects": [], "lookalikes": [], "text_traps": []}
        for _ in range(k):
            name = RANDOM_OBJECTS[rng.integers(len(RANDOM_OBJECTS))]
            groups["random_objects"].append(sprites.emoji(name, rng) | dict(
                shape=name, subtype="random", lookalike_of=None, text=None, color=None))
            of = CLASSES[rng.integers(len(CLASSES))]
            name = LOOKALIKES[of][rng.integers(len(LOOKALIKES[of]))]
            groups["lookalikes"].append(sprites.emoji(name, rng) | dict(
                shape=name, subtype="lookalike", lookalike_of=of, text=None, color=None))
            groups["text_traps"].append(make_text_trap(sprites, counts, rng))
        objs = targets + [o for g in TRAP_GROUPS for o in groups[g]]
        for _ in range(20):
            centers = place([o["radius"] for o in objs], rng)
            if centers is not None:
                break
        if centers is not None:
            break
    else:
        raise RuntimeError(f"scene {scene_id}: could not place objects")

    for i, (o, (cx, cy)) in enumerate(zip(objs, centers)):
        w, h = o["sprite"].size
        o["x0"], o["y0"] = int(round(cx - w / 2)), int(round(cy - h / 2))
        o["cx"], o["cy"] = o["x0"] + w / 2, o["y0"] + h / 2
        o["id"] = i
        o["role"] = "target" if i < n_targets else "distractor"
    return dict(scene_id=scene_id, split=split, counts=counts, targets=targets, groups=groups)


def distractors_for(scene, version):
    if version == "clean":
        return []
    if version == "everything":
        return [o for g in TRAP_GROUPS for o in scene["groups"][g]]
    return scene["groups"][version]


def public_object(o):
    w, h = o["sprite"].size
    return {
        "id": o["id"], "role": o["role"], "shape": o["shape"], "subtype": o["subtype"],
        "lookalike_of": o["lookalike_of"], "text": o["text"], "color": o["color"],
        "center": [round(o["cx"], 1), round(o["cy"], 1)], "radius": round(o["radius"], 1),
        "rotation_deg": o["rotation_deg"], "flipped": o["flipped"], "size_px": o["size_px"],
        "bbox": [o["x0"], o["y0"], o["x0"] + w, o["y0"] + h],
    }


def render_scene(scene, image_dir):
    layer = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (255, 255, 255, 255))
    for o in scene["targets"]:
        layer.alpha_composite(o["sprite"], (o["x0"], o["y0"]))

    rows, sid, c = [], scene["scene_id"], scene["counts"]
    for difficulty, version in enumerate(VERSIONS):
        img = layer.copy()
        dists = distractors_for(scene, version)
        for o in dists:
            img.alpha_composite(o["sprite"], (o["x0"], o["y0"]))
        image_id = f"{ID_PREFIX}_{sid:04d}_{version}"
        img.convert("RGB").save(image_dir / f"{image_id}.png", optimize=True)
        rows.append({
            "image_id": image_id,
            "file_name": f"images/{image_id}.png",
            "scene_id": sid,
            "version": version,
            "difficulty": difficulty,
            "split": scene["split"],
            "question": QUESTION,
            "answer": format_answer(c),
            **{f"count_{k}": c[k] for k in CLASSES},
            "count_total": sum(c.values()),
            "num_distractors": len(dists),
            "objects": json.dumps([public_object(o) for o in scene["targets"] + dists]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, default=500, help="number of scenes (images = 5 x scenes)")
    ap.add_argument("--out", default=str(ROOT / "data" / "objects_vqa"), help="output folder")
    ap.add_argument("--seed", type=int, default=42, help="random seed (same seed -> same dataset)")
    ap.add_argument("--font", default=str(DEFAULT_FONT), help="path to NotoColorEmoji.ttf")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output folder")
    args = ap.parse_args()

    sprites = Sprites(Path(args.font))
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"'{out}' already exists. Re-run with --overwrite to replace it.")
        shutil.rmtree(out)
    image_dir = out / "images"
    image_dir.mkdir(parents=True)

    nt, nk = len(TARGET_COUNTS), len(GROUP_SIZES)
    design = [(TARGET_COUNTS[i % nt], GROUP_SIZES[(i // nt) % nk], SPLIT_PATTERN[i % len(SPLIT_PATTERN)])
              for i in range(args.scenes)]
    order = np.random.default_rng([args.seed, 3]).permutation(args.scenes)

    rows = []
    for scene_id, idx in enumerate(order, start=1):
        n_t, k, split = design[idx]
        rows += render_scene(make_scene(scene_id, n_t, k, split, args.seed, sprites), image_dir)
        if scene_id % 50 == 0 or scene_id == args.scenes:
            print(f"  {scene_id}/{args.scenes} scenes  ({len(rows)} images)")

    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    split_counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")}
    all_lookalikes = [n for v in LOOKALIKES.values() for n in v]
    info = {
        "name": DATASET_NAME,
        "title": "Objects-VQA",
        "version": DATASET_VERSION,
        "created": date.today().isoformat(),
        "summary": ("Count apples, cats, dogs and cars. Each scene is drawn in 5 versions with the same "
                    "correct answer; versions add unrelated objects, look-alike objects, misleading text, "
                    "or all of them."),
        "num_scenes": args.scenes,
        "num_images": len(rows),
        "images_per_split": split_counts,
        "versions": {v: {"difficulty": i, "description": VERSION_DESCRIPTIONS[v]} for i, v in enumerate(VERSIONS)},
        "question": QUESTION,
        "answer_format": format_answer({c: "<number>" for c in CLASSES}),
        "artwork": "Noto Color Emoji by Google (SIL Open Font License 1.1), " + FONT_URL,
        "config": {
            "seed": args.seed, "id_prefix": ID_PREFIX, "image_size": IMG_SIZE, "stroke_px": 0,
            "classes": CLASSES, "versions": VERSIONS, "target_counts": TARGET_COUNTS,
            "group_sizes": GROUP_SIZES, "max_per_class": MAX_PER_CLASS, "size_range_px": list(SIZE_RANGE),
            "max_rotation_deg": MAX_ROTATION, "min_gap_px": MIN_GAP, "edge_margin_px": EDGE_MARGIN,
            "lookalikes": LOOKALIKES, "random_objects": RANDOM_OBJECTS,
            "emoji_codepoints": {k: f"U+{v:04X}" for k, v in EMOJI.items()},
            # used by validate_dataset.py
            "distractor_shapes": {"random_objects": RANDOM_OBJECTS, "lookalikes": all_lookalikes,
                                  "text_traps": ["text"], "everything": RANDOM_OBJECTS + all_lookalikes + ["text"]},
            "distractors_per_version": {"random_objects": GROUP_SIZES, "lookalikes": GROUP_SIZES,
                                        "text_traps": GROUP_SIZES, "everything": [3 * g for g in GROUP_SIZES]},
            "same_slot_versions": [],
            "union_versions": {"everything": TRAP_GROUPS},
        },
    }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Done: {len(rows)} images in '{out}'  {split_counts}")
    print(f"Next: python validate_dataset.py --data {out}")


if __name__ == "__main__":
    main()
