#!/usr/bin/env python
"""
Generate the Shapes-VQA distractor dataset.

Every *scene* is a random layout of named shapes (circle, triangle, square, star).
Each scene is drawn in 4 versions that differ ONLY in the extra distractors:

    difficulty 0  clean       only the named shapes
    difficulty 1  squiggles   + open scribble lines
    difficulty 2  blobs       + irregular closed blobs
    difficulty 3  lookalikes  + broken / wavy versions of the named shapes

The named shapes (position, size, colour, rotation) are pixel-identical in all 4
versions, and the distractors sit in the same spots with the same colours in
versions 1-3. So the correct answer is the same for all 4 versions of a scene.

Usage
    python scripts/generate_dataset.py                          # 500 scenes -> data/shapes_vqa_dataset/
    python scripts/generate_dataset.py --scenes 10 --out test_run
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
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent    # repo root (this file lives in scripts/)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATASET_NAME = "shapes-vqa-distractors"
DATASET_VERSION = "1.0"

IMG_SIZE = 512        # final image size in pixels
SUPERSAMPLE = 4       # draw at 4x size, then shrink -> smooth anti-aliased lines
STROKE = 4            # outline thickness in final pixels (identical for every object)

CLASSES = ["circle", "triangle", "square", "star"]          # the "named" shapes
VERSIONS = ["clean", "squiggles", "blobs", "lookalikes"]    # list index = difficulty
TARGET_COUNTS = [3, 4, 5, 6, 7]     # named shapes per scene (balanced across scenes)
DISTRACTOR_COUNTS = [2, 4, 6]       # distractors per scene (balanced across scenes)
MAX_PER_CLASS = 3                   # at most 3 of any one named shape per image

RADIUS_RANGE = (28.0, 46.0)  # every object fits inside a circle ("slot") of this radius
MIN_GAP = 16.0               # min empty pixels between two slots -> nothing touches
EDGE_MARGIN = 10.0           # min pixels between a slot and the image border

# 70 / 15 / 15 split, assigned per scene (all 4 versions share the split).
# Repeating block of 20 scenes: 14 train, 3 validation, 3 test (interleaved).
SPLIT_PATTERN = ["train", "train", "validation", "train", "train", "test", "train", "train",
                 "train", "validation", "train", "test", "train", "train", "train", "validation",
                 "train", "test", "train", "train"]

# Same palette for named shapes and distractors, so colour is never a shortcut.
PALETTE = {
    "black": (20, 20, 20),
    "red": (214, 39, 40),
    "blue": (31, 119, 180),
    "green": (44, 160, 44),
    "orange": (255, 127, 14),
    "purple": (148, 103, 189),
    "brown": (140, 86, 75),
    "pink": (227, 119, 194),
}
COLOR_NAMES = list(PALETTE)

# One fixed question for every image, so versions are directly comparable.
QUESTION = (
    "Count the shapes in this image. Only count complete, regular shapes of these "
    "four types: circle, triangle, square, star. Do not count anything else, such as "
    "scribbles, irregular blobs, or broken or wavy versions of these shapes. "
    "Answer in exactly this format: circle: <number>, triangle: <number>, "
    "square: <number>, star: <number>"
)

VERSION_DESCRIPTIONS = {
    "clean": "Only the named shapes.",
    "squiggles": "Named shapes + open scribble lines (waves, zigzags, scribbles).",
    "blobs": "Named shapes + irregular closed blobs that resemble no named shape.",
    "lookalikes": "Named shapes + broken (gap in outline) or wavy-sided versions of named shapes.",
}


def format_answer(counts: dict) -> str:
    return ", ".join(f"{c}: {counts[c]}" for c in CLASSES)


# --------------------------------------------------------------------------- #
# Geometry helpers (all coordinates are in final-image pixels, float)
# --------------------------------------------------------------------------- #
def _ring(cx, cy, radii, angles):
    return np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1)


def _close(path):
    return np.vstack([path, path[:1]])


def densify(path, step=1.0):
    """Insert points so consecutive points are <= `step` px apart (keeps corners)."""
    out = [path[:1]]
    for p, q in zip(path[:-1], path[1:]):
        n = max(1, int(math.ceil(np.hypot(*(q - p)) / step)))
        t = (np.arange(1, n + 1) / n)[:, None]
        out.append(p + (q - p) * t)
    return np.vstack(out)


def arclength(path):
    seg = np.hypot(*np.diff(path, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(seg)])


def scale_about(path, cx, cy, factor):
    return np.array([cx, cy]) + (path - np.array([cx, cy])) * factor


def max_radius(path, cx, cy):
    return float(np.hypot(path[:, 0] - cx, path[:, 1] - cy).max())


def polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def convex_hull(p):
    pts = sorted(map(tuple, p))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def _self_crossing(path):
    """True if any two non-adjacent segments of an open polyline cross."""
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    n = len(path) - 1
    for i in range(n):
        for j in range(i + 2, n):
            a, b, c, d = path[i], path[i + 1], path[j], path[j + 1]
            if orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0:
                return True
    return False


def chaikin(path, iterations=4):
    """Smooth an open polyline (endpoints are kept, curve stays inside the control hull)."""
    for _ in range(iterations):
        q = 0.75 * path[:-1] + 0.25 * path[1:]
        r = 0.25 * path[:-1] + 0.75 * path[1:]
        mid = np.empty((2 * len(q), 2))
        mid[0::2], mid[1::2] = q, r
        path = np.vstack([path[:1], mid, path[-1:]])
    return path


# --------------------------------------------------------------------------- #
# Named shapes (targets)
# --------------------------------------------------------------------------- #
def named_shape_outline(shape, cx, cy, R, rng):
    """Closed outline of a perfect named shape that fits in a slot of radius R.
    Returns (path, rotation_degrees)."""
    if shape == "circle":
        a = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        return _close(_ring(cx, cy, np.full(120, 0.9 * R), a)), 0.0
    if shape == "triangle":                      # equilateral, any orientation
        rot = rng.uniform(0, 120)
        a = np.deg2rad(rot) - np.pi / 2 + 2 * np.pi * np.arange(3) / 3
        return _close(_ring(cx, cy, np.full(3, R), a)), rot
    if shape == "square":                        # tilted at most 15 deg (never a "diamond")
        rot = rng.uniform(-15, 15)
        a = np.deg2rad(rot) + np.pi / 4 + np.pi / 2 * np.arange(4)
        return _close(_ring(cx, cy, np.full(4, R), a)), rot
    if shape == "star":                          # regular 5-pointed star
        rot = rng.uniform(0, 72)
        k = np.arange(10)
        a = np.deg2rad(rot) - np.pi / 2 + np.pi * k / 5
        r = np.where(k % 2 == 0, R, 0.45 * R)
        return _close(_ring(cx, cy, r, a)), rot
    raise ValueError(shape)


# --------------------------------------------------------------------------- #
# Distractors
# --------------------------------------------------------------------------- #
def make_squiggle(cx, cy, R, rng):
    """Open line: a sine wave, a zigzag, or a smooth random scribble."""
    style = ["wave", "zigzag", "scribble"][rng.integers(3)]
    if style == "wave":
        x = np.linspace(-1, 1, 160)
        amp, periods, phase = rng.uniform(0.2, 0.3), rng.uniform(2.0, 3.0), rng.uniform(0, 2 * np.pi)
        pts = np.stack([x, amp * np.sin(np.pi * periods * (x + 1) + phase)], axis=1)
    elif style == "zigzag":                     # regular /\/\/\ (no letter-like Z or N)
        n_seg = int(rng.integers(6, 10))
        x = np.linspace(-1, 1, n_seg + 1)
        y = np.where(np.arange(n_seg + 1) % 2 == 0, 1.0, -1.0) * rng.uniform(0.18, 0.28)
        pts = np.stack([x, y], axis=1)
    else:
        # Smooth irregular meander: segments of random length whose turns alternate
        # left/right. So it never curls into a C/U arc (which would look like a broken
        # circle) and never loops/crosses itself (which looks like a digit or letter).
        for _ in range(1000):
            n_seg = int(rng.integers(4, 7))
            lengths = rng.uniform(0.6, 1.0, n_seg)
            turns = np.deg2rad(rng.uniform(40, 75, n_seg - 1)) * rng.choice([-1.0, 1.0])
            turns[1::2] *= -1                              # alternate turn direction
            heading = np.concatenate([[0.0], np.cumsum(turns)])
            steps = np.stack([lengths * np.cos(heading), lengths * np.sin(heading)], axis=1)
            ctrl = np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])
            ctrl -= (ctrl.min(0) + ctrl.max(0)) / 2
            if not _self_crossing(ctrl):
                break
        else:
            raise RuntimeError("could not sample a scribble")
        pts = chaikin(ctrl, 4)
    theta = rng.uniform(0, np.pi)
    rotm = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = pts @ rotm.T
    pts = pts / np.hypot(pts[:, 0], pts[:, 1]).max() * R * rng.uniform(0.85, 1.0)
    return dict(shape="squiggle", subtype=style, lookalike_of=None, rotation_deg=None,
                path=pts + np.array([cx, cy]))


def make_blob(cx, cy, R, rng):
    """Irregular closed blob: smooth, clearly non-convex, and with no rotational
    symmetry (2/3/4/5-fold symmetry would make it look like an oval, rounded
    triangle, rounded square or star)."""
    n = 160
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = np.arange(1, 6)
    for _ in range(100000):
        ca = rng.normal(0, 1, 5) * 0.32 / k ** 0.6
        cb = rng.normal(0, 1, 5) * 0.32 / k ** 0.6
        r = 1 + (ca[:, None] * np.cos(k[:, None] * a) + cb[:, None] * np.sin(k[:, None] * a)).sum(0)
        if r.min() < 0.35 * r.max():                       # too pinched / spiky
            continue
        # n-fold rotational self-correlation of the outline (1 = perfectly symmetric)
        e = (ca ** 2 + cb ** 2)[1:]                         # harmonics 2..5 (1 is just a shift)
        symmetry = max((e * np.cos(2 * np.pi * k[1:] / m)).sum() / e.sum() for m in (2, 3, 4, 5))
        if symmetry > 0.2:
            continue
        unit = _ring(0, 0, r / r.max(), a)
        solidity = polygon_area(unit) / polygon_area(convex_hull(unit))
        if solidity > 0.88:                                 # too convex -> could read as a circle
            continue
        break
    else:
        raise RuntimeError("could not sample a blob")
    rot = rng.uniform(0, 2 * np.pi)
    pts = _ring(cx, cy, r / r.max() * R * rng.uniform(0.85, 1.0), a + rot)
    return dict(shape="blob", subtype=None, lookalike_of=None, rotation_deg=None, path=_close(pts))


def with_gap(closed, frac, rng):
    """Remove a contiguous `frac` of the outline -> one open path."""
    dense = densify(closed, 1.0)
    s = arclength(dense)
    perimeter = s[-1]
    pts, s = dense[:-1], s[:-1]
    t = (s - rng.uniform(0, perimeter)) % perimeter
    keep = t >= frac * perimeter
    return pts[keep][np.argsort(t[keep])]


def wavy_amplitude(R, edge_len=None):
    amp = 0.15 * R if edge_len is None else min(0.15 * R, 0.25 * edge_len)
    return max(4.0, amp)


def with_wavy_edges(closed, edges, R):
    """Polygon with the chosen edges replaced by waves perpendicular to the edge.
    Waves are zero at both ends, so every corner stays exactly where it was."""
    verts = closed[:-1]
    n = len(verts)
    parts = []
    for i in range(n):
        p, q = verts[i], verts[(i + 1) % n]
        if i not in edges:
            parts.append(p[None])
            continue
        length = float(np.hypot(*(q - p)))
        half_waves = max(2, int(round(length / 13.0)))     # ~26 px per full wiggle
        t = np.linspace(0, 1, int(math.ceil(length)) + 1)[:-1]
        normal = np.array([-(q - p)[1], (q - p)[0]]) / length
        offset = wavy_amplitude(R, length) * np.sin(np.pi * half_waves * t)
        parts.append(p + (q - p) * t[:, None] + normal * offset[:, None])
    return _close(np.vstack(parts))


def with_wavy_arc(closed, cx, cy, frac, amp, rng):
    """Circle only: replace a contiguous `frac` of the outline with a wavy section."""
    dense = densify(closed, 1.0)
    s = arclength(dense)
    perimeter = s[-1]
    pts, s = dense[:-1].copy(), s[:-1]
    window = frac * perimeter
    t = (s - rng.uniform(0, perimeter)) % perimeter
    inside = t < window
    periods = max(2, int(round(window / 26.0)))           # ~26 px per wiggle
    d = amp * np.sin(2 * np.pi * periods * t[inside] / window)   # 0 at both ends -> continuous
    v = pts[inside] - np.array([cx, cy])
    r = np.hypot(v[:, 0], v[:, 1])
    pts[inside] = np.array([cx, cy]) + v * (np.maximum(r + d, 0.3 * r) / r)[:, None]
    return _close(pts)


def make_lookalike(cx, cy, R, rng):
    """A named shape (same size/rotation rules as real ones) with a clearly visible defect:
    'gap'  -> 15-25% of the outline is missing
    'wavy' -> one side is a wiggly line (triangle/square: one side, star: both edges of
              one point, circle: 30-45% of the circumference)"""
    of = CLASSES[rng.integers(len(CLASSES))]
    closed, rot = named_shape_outline(of, cx, cy, R, rng)
    if rng.random() < 0.5:
        defect, path = "gap", with_gap(closed, rng.uniform(0.15, 0.25), rng)
    else:
        defect = "wavy"
        if of == "circle":
            path = with_wavy_arc(closed, cx, cy, rng.uniform(0.30, 0.45), wavy_amplitude(R), rng)
        elif of == "star":                       # outer points are at even vertex indices
            j = int(rng.integers(5))
            path = with_wavy_edges(closed, {(2 * j - 1) % 10, 2 * j}, R)
        else:
            path = with_wavy_edges(closed, {int(rng.integers(len(closed) - 1))}, R)
    mr = max_radius(path, cx, cy)
    if mr > R:                                              # keep it inside its slot
        path = scale_about(path, cx, cy, R / mr)
    return dict(shape="lookalike", subtype=defect, lookalike_of=of, rotation_deg=rot, path=path)


DISTRACTOR_MAKERS = {"squiggles": make_squiggle, "blobs": make_blob, "lookalikes": make_lookalike}


# --------------------------------------------------------------------------- #
# Scene layout
# --------------------------------------------------------------------------- #
def sample_target_classes(n, rng):
    counts = {c: 0 for c in CLASSES}
    shapes = []
    for _ in range(n):
        options = [c for c in CLASSES if counts[c] < MAX_PER_CLASS]
        c = options[rng.integers(len(options))]
        counts[c] += 1
        shapes.append(c)
    return shapes


def place_slots(radii, rng, tries_per_slot=400):
    """Non-overlapping circles inside the image (largest first). None if it fails."""
    placed = []
    for r in sorted(radii, reverse=True):
        lo, hi = r + EDGE_MARGIN, IMG_SIZE - r - EDGE_MARGIN
        for _ in range(tries_per_slot):
            x, y = rng.uniform(lo, hi), rng.uniform(lo, hi)
            if all(math.hypot(x - px, y - py) >= r + pr + MIN_GAP for px, py, pr in placed):
                placed.append((x, y, r))
                break
        else:
            return None
    return placed


def make_scene(scene_id, n_targets, n_distractors, split, seed):
    rng = np.random.default_rng([seed, scene_id])
    shapes = sample_target_classes(n_targets, rng)

    slots = None
    for _ in range(1000):
        slots = place_slots(rng.uniform(*RADIUS_RANGE, size=n_targets + n_distractors), rng)
        if slots is not None:
            break
    if slots is None:
        raise RuntimeError(f"scene {scene_id}: could not place objects")
    slots = [slots[i] for i in rng.permutation(len(slots))]   # role independent of size
    t_slots, d_slots = slots[:n_targets], slots[n_targets:]

    targets = []
    for shape, (cx, cy, r) in zip(shapes, t_slots):
        path, rot = named_shape_outline(shape, cx, cy, r, rng)
        targets.append(dict(role="target", shape=shape, subtype=None, lookalike_of=None,
                            rotation_deg=rot, color=COLOR_NAMES[rng.integers(len(COLOR_NAMES))],
                            cx=cx, cy=cy, radius=r, path=path))

    d_colors = [COLOR_NAMES[rng.integers(len(COLOR_NAMES))] for _ in d_slots]
    distractors = {"clean": []}
    for version in VERSIONS[1:]:
        maker = DISTRACTOR_MAKERS[version]
        distractors[version] = [
            dict(role="distractor", color=col, cx=cx, cy=cy, radius=r, **maker(cx, cy, r, rng))
            for (cx, cy, r), col in zip(d_slots, d_colors)
        ]
    counts = {c: shapes.count(c) for c in CLASSES}
    return dict(scene_id=scene_id, split=split, targets=targets, distractors=distractors,
                counts=counts, n_distractors=n_distractors)


# --------------------------------------------------------------------------- #
# Rendering + metadata
# --------------------------------------------------------------------------- #
def draw_path(draw, path, rgb):
    """Thick polyline with round joints and caps (identical style for every object)."""
    w = STROKE * SUPERSAMPLE
    h = w / 2
    pts = [tuple(p) for p in (path * SUPERSAMPLE).tolist()]
    for a, b in zip(pts[:-1], pts[1:]):
        draw.line([a, b], fill=rgb, width=w)
    for x, y in pts:
        draw.ellipse([x - h, y - h, x + h, y + h], fill=rgb)


def public_object(idx, o):
    pad = STROKE / 2 + 1
    p = o["path"]
    bbox = [max(0, math.floor(p[:, 0].min() - pad)), max(0, math.floor(p[:, 1].min() - pad)),
            min(IMG_SIZE, math.ceil(p[:, 0].max() + pad)), min(IMG_SIZE, math.ceil(p[:, 1].max() + pad))]
    return {
        "id": idx,
        "role": o["role"],
        "shape": o["shape"],
        "subtype": o["subtype"],
        "lookalike_of": o["lookalike_of"],
        "color": o["color"],
        "center": [round(o["cx"], 1), round(o["cy"], 1)],
        "radius": round(o["radius"], 1),
        "rotation_deg": None if o["rotation_deg"] is None else round(float(o["rotation_deg"]), 1),
        "bbox": bbox,
    }


def render_scene(scene, image_dir):
    big = IMG_SIZE * SUPERSAMPLE
    base = Image.new("RGB", (big, big), (255, 255, 255))
    draw = ImageDraw.Draw(base)
    for o in scene["targets"]:
        draw_path(draw, o["path"], PALETTE[o["color"]])

    rows = []
    sid = scene["scene_id"]
    for difficulty, version in enumerate(VERSIONS):
        img = base.copy()
        draw = ImageDraw.Draw(img)
        for o in scene["distractors"][version]:
            draw_path(draw, o["path"], PALETTE[o["color"]])
        image_id = f"scene_{sid:04d}_{version}"
        img.reduce(SUPERSAMPLE).save(image_dir / f"{image_id}.png", optimize=True)

        objs = scene["targets"] + scene["distractors"][version]
        c = scene["counts"]
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
            "num_distractors": len(scene["distractors"][version]),
            "objects": json.dumps([public_object(i, o) for i, o in enumerate(objs)]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, default=500, help="number of scenes (images = 4 x scenes)")
    ap.add_argument("--out", default=str(ROOT / "data" / "shapes_vqa_dataset"), help="output folder")
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

    # Balanced design: every (targets, distractors) combo and split appears equally often,
    # then scenes are shuffled so scene ids carry no pattern.
    design = [(TARGET_COUNTS[i % len(TARGET_COUNTS)],
               DISTRACTOR_COUNTS[(i // len(TARGET_COUNTS)) % len(DISTRACTOR_COUNTS)],
               SPLIT_PATTERN[i % len(SPLIT_PATTERN)]) for i in range(args.scenes)]
    order = np.random.default_rng(args.seed).permutation(args.scenes)

    rows = []
    for scene_id, idx in enumerate(order, start=1):
        n_t, n_d, split = design[idx]
        rows += render_scene(make_scene(scene_id, n_t, n_d, split, args.seed), image_dir)
        if scene_id % 50 == 0 or scene_id == args.scenes:
            print(f"  {scene_id}/{args.scenes} scenes  ({len(rows)} images)")

    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    split_counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")}
    info = {
        "name": DATASET_NAME,
        "version": DATASET_VERSION,
        "created": date.today().isoformat(),
        "num_scenes": args.scenes,
        "num_images": len(rows),
        "images_per_split": split_counts,
        "versions": {v: {"difficulty": i, "description": VERSION_DESCRIPTIONS[v]} for i, v in enumerate(VERSIONS)},
        "question": QUESTION,
        "answer_format": format_answer({c: "<number>" for c in CLASSES}),
        "config": {
            "seed": args.seed, "image_size": IMG_SIZE, "stroke_px": STROKE, "classes": CLASSES,
            "versions": VERSIONS, "target_counts": TARGET_COUNTS, "distractor_counts": DISTRACTOR_COUNTS,
            "max_per_class": MAX_PER_CLASS, "radius_range": list(RADIUS_RANGE), "min_gap_px": MIN_GAP,
            "edge_margin_px": EDGE_MARGIN, "palette": PALETTE,
        },
    }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Done: {len(rows)} images in '{out}'  {split_counts}")
    print(f"Next: python validate_dataset.py --data {out}")


if __name__ == "__main__":
    main()
