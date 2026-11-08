#!/usr/bin/env python
"""
Generate Shapes-VQA HARD.

Task: count only the named shapes (circle, triangle, square, star) of ONE colour,
named in the question (e.g. "Count only the red shapes ...").

Every scene is drawn in 10 versions. The target shapes are pixel-identical in all 10,
so the correct answer is the same for all 10 - only the traps change:

    difficulty 0  clean        only the target-colour named shapes
    difficulty 1  wrong_color  + perfect named shapes in OTHER colours
    difficulty 2  lookalikes   + broken / wavy shapes in the TARGET colour
    difficulty 3  text_traps   + misleading words and numbers ("star: 4", "red circle", "5")
    difficulty 4  everything   + all of the traps from versions 1, 2 and 3 together
    difficulty 5-8  squiggles_2 / _4 / _8 / _16
                               + scribbles with 2, 4, 8 or 16 curves. Same spots, sizes and colours
                                 in all four; only the number of curves (complexity) changes.
    difficulty 9  other_shapes + nameable shapes that are none of the four types (heart, moon,
                                 cross, lightning bolt), in the squiggle spots, sizes and colours.
                                 Compare with squiggles_*: nameable vs. nameless clutter, same places.

Each trap version adds the same number of objects (2, 3 or 4); "everything" contains
exactly the union of versions 1-3 (6, 9 or 12 traps). Versions 0-4 are identical to v1.0,
versions 5-8 to v1.1.

Usage
    python scripts/generate_hard_dataset.py                          # 500 scenes -> data/shapes_vqa_hard/
    python scripts/generate_hard_dataset.py --scenes 10 --out test_hard
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

import generate_dataset as base     # shared shape geometry + drawing (identical style to v1)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATASET_NAME = "shapes-vqa-hard"
DATASET_VERSION = "1.2"             # 1.1 = adds squiggles_* (0-4 unchanged); 1.2 = adds other_shapes (0-8 unchanged)
ID_PREFIX = "hard"

IMG_SIZE = base.IMG_SIZE            # 512
SS = base.SUPERSAMPLE               # draw at 4x, then shrink
STROKE = base.STROKE                # 4 px outlines

CLASSES = base.CLASSES              # circle, triangle, square, star
SQUIGGLE_CURVES = [2, 4, 8, 16]     # squiggle complexity = number of curves (bends) in each squiggle
SQUIGGLE_VERSIONS = [f"squiggles_{n}" for n in SQUIGGLE_CURVES]
# nameable shapes that are none of the four types and contain none of them (no arrow: its head is a triangle)
OTHER_SHAPES = ["heart", "moon", "cross", "bolt"]
# degrees; the moon always opens to the left (opening right, it reads like the letter C)
OTHER_ROTATION = {"heart": (-15, 15), "moon": (150, 210), "cross": (-15, 15), "bolt": (-15, 15)}
VERSIONS = ["clean", "wrong_color", "lookalikes", "text_traps", "everything"] + SQUIGGLE_VERSIONS + ["other_shapes"]
TRAP_GROUPS = ["wrong_color", "lookalikes", "text_traps"]
TARGET_COUNTS = [3, 4, 5, 6, 7]     # target shapes per scene (balanced)
GROUP_SIZES = [2, 3, 4]             # traps per trap type (balanced); "everything" has 3x this
MAX_PER_CLASS = 3

SHAPE_RADIUS = (24.0, 38.0)         # every shape fits in a slot circle of this radius
SQUIGGLE_RADIUS = (30.0, 44.0)      # squiggle slots (a squiggle always fills its slot)
TEXT_RADIUS = (46.0, 58.0)          # text slots are larger (words are wide)
MIN_GAP = 11.0                      # min empty pixels between two slots
EDGE_MARGIN = 8.0
SPLIT_PATTERN = base.SPLIT_PATTERN  # 70 / 15 / 15 by scene

# 4 colours that nobody confuses with each other
PALETTE = {
    "red": (214, 39, 40),
    "blue": (31, 119, 180),
    "green": (44, 160, 44),
    "black": (20, 20, 20),
}
COLORS = list(PALETTE)

QUESTION_TEMPLATE = (
    "Count the shapes in this image. Only count complete, regular shapes that are {color} "
    "and are one of these four types: circle, triangle, square, star. Do not count shapes of "
    "any other colour, broken or wavy shapes, or any words or numbers written in the image. "
    "Answer in exactly this format: circle: <number>, triangle: <number>, square: <number>, "
    "star: <number>"
)

VERSION_DESCRIPTIONS = {
    "clean": "Only the target-colour named shapes.",
    "wrong_color": "+ perfect named shapes in other colours (tests colour-shape binding).",
    "lookalikes": "+ broken or wavy shapes in the target colour (tests shape similarity).",
    "text_traps": "+ misleading words and numbers written in the image (tests text distraction).",
    "everything": "+ all traps from wrong_color, lookalikes and text_traps together.",
    **{f"squiggles_{n}": f"+ scribbles with {n} curves each (same spots, sizes and colours in every "
                         f"squiggles_* version; only the number of curves changes)." for n in SQUIGGLE_CURVES},
    "other_shapes": "+ nameable shapes that are none of the four types (heart, moon, cross, lightning bolt), "
                    "in the same spots, sizes and colours as the squiggles.",
}

TEXT_MAX_PX, TEXT_MIN_PX = 30, 18   # font size range in final pixels (always easy to read)
_FONTS: dict[int, ImageFont.FreeTypeFont] = {}


def font(px_final: int):
    """Pillow's bundled open font (Aileron), at supersampled size. Same on every OS."""
    if px_final not in _FONTS:
        _FONTS[px_final] = ImageFont.load_default(size=px_final * SS)
    return _FONTS[px_final]


# --------------------------------------------------------------------------- #
# Traps
# --------------------------------------------------------------------------- #
def make_wrong_color(cx, cy, R, target_color, rng):
    shape = CLASSES[rng.integers(len(CLASSES))]
    path, rot = base.named_shape_outline(shape, cx, cy, R, rng)
    others = [c for c in COLORS if c != target_color]
    return dict(shape=shape, subtype="wrong_color", lookalike_of=None, rotation_deg=rot,
                color=others[rng.integers(len(others))], path=path, text=None)


def make_lookalike(cx, cy, R, target_color, rng):
    d = base.make_lookalike(cx, cy, R, rng)
    d.update(color=target_color, text=None)
    return d


def _text_candidates(target_color, counts, rng):
    """A misleading text. Count claims are always WRONG for the target colour."""
    kind = ["class_word", "false_count", "number", "color_phrase"][rng.integers(4)]
    cls = CLASSES[rng.integers(len(CLASSES))]
    if kind == "class_word":
        return kind, cls if rng.random() < 0.5 else cls + "s"
    if kind == "false_count":
        n = int(rng.choice([k for k in range(1, 10) if k != counts[cls]]))
        if rng.random() < 0.5:
            return kind, f"{cls}: {n}"
        return kind, f"{n} {cls}" + ("s" if n > 1 else "")
    if kind == "number":
        total = sum(counts.values())
        return kind, str(int(rng.choice([k for k in range(1, 10) if k != total])))
    return kind, f"{target_color} {cls}"


def make_text(cx, cy, R, target_color, counts, rng):
    for _ in range(100):
        kind, text = _text_candidates(target_color, counts, rng)
        for px in range(TEXT_MAX_PX, TEXT_MIN_PX - 1, -1):
            x0, y0, x1, y1 = font(px).getbbox(text, anchor="mm")
            half_diag = math.hypot(max(-x0, x1), max(-y0, y1)) / SS
            if half_diag <= 0.95 * R:
                break
        else:
            continue                                     # too long for this slot, pick another
        color = target_color if rng.random() < 0.5 else COLORS[rng.integers(len(COLORS))]
        return dict(shape="text", subtype=kind, lookalike_of=None, rotation_deg=None,
                    color=color, text=text, font_px=px,
                    bbox_rel=(x0 / SS, y0 / SS, x1 / SS, y1 / SS), path=None)
    raise RuntimeError("could not fit a text trap")


MIN_BEND_DEG = 30.0                  # every bend of a squiggle turns at least this much (so it is visible)
MAX_BEND_DEG = 120.0                 # ... and at most this much (hairpins make letter/digit-like strokes)


def _squiggle_points(n_bends, rng):
    """n_bends + 2 points in the unit disc: every step >= 0.6 long and every bend between
    MIN_BEND_DEG and MAX_BEND_DEG."""
    min_bend, max_bend = np.deg2rad(MIN_BEND_DEG), np.deg2rad(MAX_BEND_DEG)
    for _ in range(10000):
        pts = [_disc_point(rng)]
        for _ in range(n_bends + 1):
            for _ in range(300):
                q = _disc_point(rng)
                if np.hypot(*(q - pts[-1])) < 0.6:
                    continue
                if len(pts) >= 2:
                    a, b = pts[-1] - pts[-2], q - pts[-1]
                    if not min_bend <= abs(math.atan2(a[0] * b[1] - a[1] * b[0], a @ b)) <= max_bend:
                        continue
                pts.append(q)
                break
            else:
                break
        if len(pts) == n_bends + 2 and np.hypot(*(pts[-1] - pts[0])) >= 0.8:   # open ends
            return np.array(pts)
    raise RuntimeError("could not sample a squiggle")


def _disc_point(rng):
    r, a = 0.9 * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
    return np.array([r * math.cos(a), r * math.sin(a)])


def make_squiggle(cx, cy, R, n_curves, rng):
    """A smooth scribble with exactly `n_curves` bends (each at least MIN_BEND_DEG), through
    n_curves + 2 random points in the slot, smoothed. More curves -> a more tangled scribble
    (it may cross itself)."""
    ctrl = _squiggle_points(n_curves, rng)
    path = base.chaikin(ctrl, 4)                               # smooth curve, unit scale
    path -= (path.min(0) + path.max(0)) / 2                    # centre it, then make it fill its slot,
    path *= 0.95 * R / np.hypot(path[:, 0], path[:, 1]).max()  # so every complexity level is the same size
    path += np.array([cx, cy])
    seg = np.diff(path, axis=0)
    turn = np.arctan2(seg[:-1, 0] * seg[1:, 1] - seg[:-1, 1] * seg[1:, 0], (seg[:-1] * seg[1:]).sum(1))
    return dict(shape="squiggle", subtype=f"{n_curves}_curves", lookalike_of=None, rotation_deg=None,
                path=path, text=None, curves=n_curves, turning_deg=round(float(np.degrees(np.abs(turn).sum())), 1))


def _other_outline(kind):
    """Closed outline of one of OTHER_SHAPES in unit coordinates (y points down, like the image)."""
    if kind == "heart":                          # classic heart curve: lobes up, tip down
        t = np.linspace(0, 2 * np.pi, 160, endpoint=False)
        pts = np.stack([16 * np.sin(t) ** 3,
                        -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))], axis=1)
    elif kind == "moon":                         # crescent: unit circle minus an inner circle shifted right
        r2, d = 0.9, 0.4
        x = (d * d + 1 - r2 * r2) / (2 * d)      # the two circles meet at (x, +-y)
        y = math.sqrt(1 - x * x)
        a0, b0 = math.atan2(y, x), math.atan2(y, x - d)
        outer = np.linspace(a0, 2 * np.pi - a0, 120)          # round back of the moon (through the left)
        inner = np.linspace(2 * np.pi - b0, b0, 100)[1:-1]    # inner edge, also through the left
        pts = np.vstack([np.stack([np.cos(outer), np.sin(outer)], axis=1),
                         np.stack([d + r2 * np.cos(inner), r2 * np.sin(inner)], axis=1)])
    elif kind == "cross":                        # plus-shaped cross with equal arms
        w = 0.3
        pts = np.array([(w, -1), (w, -w), (1, -w), (1, w), (w, w), (w, 1),
                        (-w, 1), (-w, w), (-1, w), (-1, -w), (-w, -w), (-w, -1)], dtype=float)
    elif kind == "bolt":                         # lightning bolt
        pts = np.array([(0, -1), (-0.6, 0.15), (-0.05, 0.15), (-0.35, 1), (0.6, -0.15), (0.15, -0.15),
                        (0.6, -1)], dtype=float)
    else:
        raise ValueError(kind)
    return base._close(pts)


def make_other_shape(cx, cy, R, kind, rng):
    """A complete outline of a nameable non-target shape. Sized like a squiggle: centred, and
    scaled so its farthest point is 0.95 R from the slot centre."""
    rot = rng.uniform(*OTHER_ROTATION[kind])
    a = math.radians(rot)
    path = _other_outline(kind) @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    path -= (path.min(0) + path.max(0)) / 2
    path *= 0.95 * R / np.hypot(path[:, 0], path[:, 1]).max()
    path += np.array([cx, cy])
    return dict(shape=kind, subtype="other_shape", lookalike_of=None, rotation_deg=rot, path=path, text=None)


# --------------------------------------------------------------------------- #
# Scene
# --------------------------------------------------------------------------- #
def place_extra(fixed, radii, rng, tries_per_slot=2000):
    """Place new circles that avoid the fixed ones (x, y, r) and each other."""
    placed = list(fixed)
    out = []
    for r in radii:
        lo, hi = r + EDGE_MARGIN, IMG_SIZE - r - EDGE_MARGIN
        for _ in range(tries_per_slot):
            x, y = rng.uniform(lo, hi), rng.uniform(lo, hi)
            if all(math.hypot(x - px, y - py) >= r + pr + MIN_GAP for px, py, pr in placed):
                placed.append((x, y, r))
                out.append((x, y, r))
                break
        else:
            return None
    return out


def place(radii, rng, tries_per_slot=500):
    order = sorted(range(len(radii)), key=lambda i: -radii[i])
    placed = {}
    for i in order:
        r = radii[i]
        lo, hi = r + EDGE_MARGIN, IMG_SIZE - r - EDGE_MARGIN
        for _ in range(tries_per_slot):
            x, y = rng.uniform(lo, hi), rng.uniform(lo, hi)
            if all(math.hypot(x - px, y - py) >= r + pr + MIN_GAP for px, py, pr in placed.values()):
                placed[i] = (x, y, r)
                break
        else:
            return None
    return [placed[i] for i in range(len(radii))]


def make_scene(scene_id, n_targets, k, target_color, split, seed):
    rng = np.random.default_rng([seed, scene_id, 2])
    shapes = base.sample_target_classes(n_targets, rng)
    counts = {c: shapes.count(c) for c in CLASSES}

    n_shape_slots = n_targets + 2 * k             # targets + wrong-colour + lookalikes
    slots = None
    for _ in range(2000):
        radii = list(rng.uniform(*SHAPE_RADIUS, n_shape_slots)) + list(rng.uniform(*TEXT_RADIUS, k))
        slots = place(radii, rng)
        if slots is not None:
            break
    if slots is None:
        raise RuntimeError(f"scene {scene_id}: could not place objects")
    shape_slots = [slots[i] for i in rng.permutation(n_shape_slots)]   # role independent of size
    text_slots = slots[n_shape_slots:]

    targets = []
    for shape, (cx, cy, r) in zip(shapes, shape_slots[:n_targets]):
        path, rot = base.named_shape_outline(shape, cx, cy, r, rng)
        targets.append(dict(role="target", shape=shape, subtype=None, lookalike_of=None, rotation_deg=rot,
                            color=target_color, cx=cx, cy=cy, radius=r, path=path, text=None))

    groups = {
        "wrong_color": [make_wrong_color(cx, cy, r, target_color, rng) | dict(cx=cx, cy=cy, radius=r)
                        for cx, cy, r in shape_slots[n_targets:n_targets + k]],
        "lookalikes": [make_lookalike(cx, cy, r, target_color, rng) | dict(cx=cx, cy=cy, radius=r)
                       for cx, cy, r in shape_slots[n_targets + k:]],
        "text_traps": [make_text(cx, cy, r, target_color, counts, rng) | dict(cx=cx, cy=cy, radius=r)
                       for cx, cy, r in text_slots],
    }
    next_id = n_targets
    for g in TRAP_GROUPS:
        for o in groups[g]:
            o.update(role="distractor", id=next_id)
            next_id += 1
    for i, o in enumerate(targets):
        o["id"] = i

    # Squiggle versions (added in v1.1). They use their OWN random stream and are drawn after
    # everything above, so versions 0-4 are exactly the same as in v1.0. The k squiggle slots
    # avoid the targets; every squiggles_* version uses the same slots and colours.
    rng_sq = np.random.default_rng([seed, scene_id, 2, 7])
    sq_slots = None
    for _ in range(2000):
        sq_slots = place_extra([(o["cx"], o["cy"], o["radius"]) for o in targets],
                               list(rng_sq.uniform(*SQUIGGLE_RADIUS, k)), rng_sq)
        if sq_slots is not None:
            break
    if sq_slots is None:
        raise RuntimeError(f"scene {scene_id}: could not place squiggles")
    others = [c for c in COLORS if c != target_color]
    sq_colors = [target_color if rng_sq.random() < 0.5 else others[rng_sq.integers(len(others))] for _ in sq_slots]
    for n in SQUIGGLE_CURVES:
        groups[f"squiggles_{n}"] = [
            make_squiggle(cx, cy, r, n, rng_sq) | dict(cx=cx, cy=cy, radius=r, color=col, role="distractor", id=next_id + j)
            for j, ((cx, cy, r), col) in enumerate(zip(sq_slots, sq_colors))
        ]

    # Other-shapes version (added in v1.2): nameable shapes in the squiggle slots with the squiggle
    # colours, so it differs from squiggles_* only in what is drawn. Own random stream, drawn after
    # everything above, so versions 0-8 are exactly the same as in v1.1. Different kinds within an image.
    rng_os = np.random.default_rng([seed, scene_id, 2, 8])
    kinds = [OTHER_SHAPES[i] for i in rng_os.permutation(len(OTHER_SHAPES))[:len(sq_slots)]]
    groups["other_shapes"] = [
        make_other_shape(cx, cy, r, kind, rng_os) | dict(cx=cx, cy=cy, radius=r, color=col, role="distractor", id=next_id + j)
        for j, ((cx, cy, r), col, kind) in enumerate(zip(sq_slots, sq_colors, kinds))
    ]
    return dict(scene_id=scene_id, split=split, target_color=target_color, counts=counts,
                targets=targets, groups=groups, k=k)


def distractors_for(scene, version):
    if version == "clean":
        return []
    if version == "everything":
        return [o for g in TRAP_GROUPS for o in scene["groups"][g]]
    return scene["groups"][version]


# --------------------------------------------------------------------------- #
# Rendering + metadata
# --------------------------------------------------------------------------- #
def draw_object(draw, o):
    rgb = PALETTE[o["color"]]
    if o["shape"] == "text":
        draw.text((o["cx"] * SS, o["cy"] * SS), o["text"], font=font(o["font_px"]), fill=rgb, anchor="mm")
    else:
        base.draw_path(draw, o["path"], rgb)


def public_object(o):
    if o["shape"] == "text":
        x0, y0, x1, y1 = o["bbox_rel"]
        xs, ys = (o["cx"] + x0, o["cx"] + x1), (o["cy"] + y0, o["cy"] + y1)
        pad = 1
    else:
        xs, ys = (o["path"][:, 0].min(), o["path"][:, 0].max()), (o["path"][:, 1].min(), o["path"][:, 1].max())
        pad = STROKE / 2 + 1
    bbox = [max(0, math.floor(xs[0] - pad)), max(0, math.floor(ys[0] - pad)),
            min(IMG_SIZE, math.ceil(xs[1] + pad)), min(IMG_SIZE, math.ceil(ys[1] + pad))]
    d = {
        "id": o["id"], "role": o["role"], "shape": o["shape"], "subtype": o["subtype"],
        "lookalike_of": o["lookalike_of"], "text": o["text"], "color": o["color"],
        "center": [round(o["cx"], 1), round(o["cy"], 1)], "radius": round(o["radius"], 1),
        "rotation_deg": None if o["rotation_deg"] is None else round(float(o["rotation_deg"]), 1),
        "bbox": bbox,
    }
    if o["shape"] == "squiggle":            # only squiggles carry these keys (keeps v1.0 rows identical)
        d["curves"], d["turning_deg"] = o["curves"], o["turning_deg"]
    return d


def render_scene(scene, image_dir):
    big = IMG_SIZE * SS
    layer = Image.new("RGB", (big, big), (255, 255, 255))
    d = ImageDraw.Draw(layer)
    for o in scene["targets"]:
        draw_object(d, o)

    rows, sid, c = [], scene["scene_id"], scene["counts"]
    for difficulty, version in enumerate(VERSIONS):
        img = layer.copy()
        d = ImageDraw.Draw(img)
        dists = distractors_for(scene, version)
        for o in dists:
            draw_object(d, o)
        image_id = f"{ID_PREFIX}_{sid:04d}_{version}"
        img.reduce(SS).save(image_dir / f"{image_id}.png", optimize=True)
        rows.append({
            "image_id": image_id,
            "file_name": f"images/{image_id}.png",
            "scene_id": sid,
            "version": version,
            "difficulty": difficulty,
            "split": scene["split"],
            "target_color": scene["target_color"],
            "question": QUESTION_TEMPLATE.format(color=scene["target_color"]),
            "answer": base.format_answer(c),
            **{f"count_{k}": c[k] for k in CLASSES},
            "count_total": sum(c.values()),
            "num_distractors": len(dists),
            "objects": json.dumps([public_object(o) for o in scene["targets"] + dists]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, default=500, help=f"number of scenes (images = {len(VERSIONS)} x scenes)")
    ap.add_argument("--out", default=str(base.ROOT / "data" / "shapes_vqa_hard"), help="output folder")
    ap.add_argument("--seed", type=int, default=42, help="random seed (same seed -> same dataset)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output folder")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"'{out}' already exists. Re-run with --overwrite to replace it.")
        shutil.rmtree(out)
    image_dir = out / "images"
    image_dir.mkdir(parents=True)

    # balanced design over (targets, traps per type, target colour) and split
    nt, nk, nc = len(TARGET_COUNTS), len(GROUP_SIZES), len(COLORS)
    design = [(TARGET_COUNTS[i % nt], GROUP_SIZES[(i // nt) % nk], COLORS[(i // (nt * nk)) % nc],
               SPLIT_PATTERN[i % len(SPLIT_PATTERN)]) for i in range(args.scenes)]
    order = np.random.default_rng([args.seed, 2]).permutation(args.scenes)

    rows = []
    for scene_id, idx in enumerate(order, start=1):
        n_t, k, color, split = design[idx]
        rows += render_scene(make_scene(scene_id, n_t, k, color, split, args.seed), image_dir)
        if scene_id % 50 == 0 or scene_id == args.scenes:
            print(f"  {scene_id}/{args.scenes} scenes  ({len(rows)} images)")

    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    split_counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")}
    info = {
        "name": DATASET_NAME,
        "title": "Shapes-VQA Hard",
        "version": DATASET_VERSION,
        "created": date.today().isoformat(),
        "summary": ("Count only the named shapes of one colour (named in the question). Each scene is drawn "
                    "in 10 versions with the same correct answer; versions add wrong-colour shapes, "
                    "target-colour lookalikes, misleading text, all of them, scribbles of increasing "
                    "complexity (2, 4, 8, 16 curves), or nameable non-target shapes in the scribble spots."),
        "num_scenes": args.scenes,
        "num_images": len(rows),
        "images_per_split": split_counts,
        "versions": {v: {"difficulty": i, "description": VERSION_DESCRIPTIONS[v]} for i, v in enumerate(VERSIONS)},
        "question_template": QUESTION_TEMPLATE,
        "answer_format": base.format_answer({c: "<number>" for c in CLASSES}),
        "config": {
            "seed": args.seed, "id_prefix": ID_PREFIX, "image_size": IMG_SIZE, "stroke_px": STROKE,
            "classes": CLASSES, "versions": VERSIONS, "colors": COLORS, "palette": PALETTE,
            "target_counts": TARGET_COUNTS, "group_sizes": GROUP_SIZES, "max_per_class": MAX_PER_CLASS,
            "shape_radius": list(SHAPE_RADIUS), "text_radius": list(TEXT_RADIUS),
            "squiggle_radius": list(SQUIGGLE_RADIUS), "squiggle_bend_deg": [MIN_BEND_DEG, MAX_BEND_DEG],
            "min_gap_px": MIN_GAP, "edge_margin_px": EDGE_MARGIN, "text_font": "Pillow built-in (Aileron Regular)",
            # used by validate_dataset.py
            "other_shape_rotation_deg": OTHER_ROTATION,
            "squiggle_curves": SQUIGGLE_CURVES,
            "distractor_shapes": {"wrong_color": CLASSES, "lookalikes": ["lookalike"], "text_traps": ["text"],
                                  "everything": CLASSES + ["lookalike", "text"],
                                  **{v: ["squiggle"] for v in SQUIGGLE_VERSIONS},
                                  "other_shapes": OTHER_SHAPES},
            "distractors_per_version": {"wrong_color": GROUP_SIZES, "lookalikes": GROUP_SIZES,
                                        "text_traps": GROUP_SIZES, "everything": [3 * g for g in GROUP_SIZES],
                                        **{v: GROUP_SIZES for v in SQUIGGLE_VERSIONS + ["other_shapes"]}},
            "same_slot_versions": SQUIGGLE_VERSIONS + ["other_shapes"],
            "union_versions": {"everything": TRAP_GROUPS},
        },
    }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Done: {len(rows)} images in '{out}'  {split_counts}")
    print(f"Next: python validate_dataset.py --data {out}")


if __name__ == "__main__":
    main()
