#!/usr/bin/env python
"""
Generate Lines-VQA: follow one line while other lines cross it.

Four dots labelled A-D sit on the left, eight numbered boxes on the right, and smooth lines run
from dots to boxes. The question names one dot; the answer is the box its line ends in.

Every scene is drawn in 6 versions. The asked-about (target) line, the dots and the boxes are
pixel-identical in all 6, so the correct answer is the same for all 6. Only the other lines change:

    difficulty 0  clean                only the target line
    difficulty 1  crossings_0          + lines from the 3 other dots; none of them crosses the target
    difficulty 2  crossings_2          + lines from the 3 other dots that cross the target 2 times in total
    difficulty 3  crossings_4          ... 4 times
    difficulty 4  crossings_8          ... 8 times
    difficulty 5  crossings_4_colored  the same lines as crossings_4, but the other 3 lines are coloured

Rules that keep the answer obvious to a human (validate_dataset.py checks them):
  * every line runs left to right and never doubles back; lines never touch or run close
    together except where they cross
  * every crossing is a clean X (at least 45 degrees) and at least 20 px from any other crossing
  * the other lines never cross each other, so every crossing in the image is on the target line
  * each box receives at most one line, and every line is the same colour and width (except in
    crossings_4_colored), so nothing shows which line lies on top at a crossing

The crossing levels are even because of topology: if the target starts at the top dot and ends in
the top box, every other line is below it at both ends and must cross it an even number of times.
The other lines may end in different boxes in different versions (the target never changes).

Usage
    python scripts/generate_lines_dataset.py                          # 500 scenes -> data/lines_vqa/
    python scripts/generate_lines_dataset.py --scenes 20 --out test_lines
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from datetime import date
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from generate_dataset import PALETTE, ROOT, SPLIT_PATTERN, STROKE, SUPERSAMPLE, chaikin, draw_path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATASET_NAME = "lines-vqa"
DATASET_VERSION = "1.0"
ID_PREFIX = "line"
RNG_TAG = 4                       # per-scene stream: default_rng([seed, scene_id, 4])

IMG_SIZE = 512
DOTS = ["A", "B", "C", "D"]
NUM_BOXES = 8
N_DISTRACTORS = 3                 # one line from each of the other dots
VERSIONS = ["clean", "crossings_0", "crossings_2", "crossings_4", "crossings_8", "crossings_4_colored"]
CROSSINGS = {"clean": 0, "crossings_0": 0, "crossings_2": 2, "crossings_4": 4, "crossings_8": 8,
             "crossings_4_colored": 4}
DRAWN_VERSIONS = ["crossings_0", "crossings_2", "crossings_4", "crossings_8"]   # each gets its own lines
RECOLORED = {"crossings_4_colored": "crossings_4"}                               # same lines, coloured
LINE_COLOR = "black"
DISTRACTOR_COLORS = ["red", "blue", "green", "orange", "purple"]
INK = PALETTE["black"]
WHITE = (255, 255, 255)

# Layout (final-image pixels). Dots on the left, boxes on the right.
DOT_X, DOT_R = 60.0, 7.0
DOT_Y = [64.0, 192.0, 320.0, 448.0]
DOT_JITTER = 12.0
LABEL_X, LABEL_PX = 30.0, 26
BOX_X0, BOX_X1, BOX_H, BOX_BORDER, BOX_PX = 452.0, 488.0, 32.0, 3, 20
BOX_Y = [32.0 + 64.0 * j for j in range(NUM_BOXES)]

# Line geometry. Every line is a Chaikin-smoothed polyline through control points that share the
# same x columns, so lines never double back and two lines can be compared column by column.
LEAD = 22.0                       # straight run out of the dot and into the box
N_COLS = 18                       # control columns between the two straight runs
CHAIKIN_ITERS = 4
XS = np.round(np.concatenate([[DOT_X], np.linspace(DOT_X + LEAD, BOX_X0 - LEAD, N_COLS), [BOX_X0]]), 2)
FREE_COLS = range(2, len(XS) - 2)          # columns not pinned to a dot or a box
CROSS_INTERVALS = list(range(3, 16))       # a crossing happens between column c and c+1 (away from dots/boxes)

Y_MIN, Y_MAX = 12.0, 500.0
TARGET_WOBBLE = (45.0, 25.0, 12.0)         # amplitude of the target's 1st/2nd/3rd sine bend
OTHER_WOBBLE = (45.0, 22.0)
MAX_TARGET_SLOPE = 0.9                     # keeps the target fairly flat, so crossings are steep
GAP_TO_TARGET = 22.0                       # min vertical gap to the target at a control column
CROSS_OFFSET = (24.0, 32.0)                # distance from the target just before/after crossing it
APPROACH = 28.0                            # extra distance allowed per column away from a crossing
GAP_BETWEEN = 30.0                         # min vertical gap between two other lines at a column
PREF_GAP = 45.0                            # preferred gap to the target / next line away from crossings
REACH_SLOPE = 2.5                          # how fast a line may move away from its dot / toward its box
REL_SLOPE = 2.8                            # typical slope of a crossing line relative to the target
PLAN_ANGLE = 45.0                          # crossings are only planned where they come out this steep
SAME_LINE_GAP = 2                          # columns between two crossings of the same line (no narrow lenses)
SMOOTH_PASSES = 4                          # smoothing rounds on the other lines' control points

# Rules checked on the final curves (also by validate_dataset.py)
MIN_ANGLE = 45.0                           # crossing angle, degrees
MIN_CROSS_DIST = 20.0                      # px between two crossings
MIN_SEP = 14.0                             # px between two lines, away from their crossings
CROSS_EXCL = 20.0                          # radius around a crossing that the MIN_SEP rule ignores
MAX_SLOPE = 5.0                            # |dy/dx| anywhere on a line
MAX_TURN = 55.0                            # degrees of turning within ~8 px of a line (no kinks)

QUESTION_TEMPLATE = (
    "Follow the line that starts at the dot labelled {dot}. Which numbered box does it end in? "
    "Lines may cross each other, but a line never turns onto another line at a crossing. "
    "Answer in exactly this format: box: <number>"
)

VERSION_DESCRIPTIONS = {
    "clean": "Only the target line (all dots and boxes are always drawn).",
    "crossings_0": "+ lines from the 3 other dots; none of them crosses the target line.",
    "crossings_2": "+ lines from the 3 other dots that cross the target line 2 times in total.",
    "crossings_4": "+ lines from the 3 other dots that cross the target line 4 times in total.",
    "crossings_8": "+ lines from the 3 other dots that cross the target line 8 times in total.",
    "crossings_4_colored": "The same lines as crossings_4, but the 3 other lines are coloured (target stays black).",
}

# (target dot, target box) pairs. crossings_0 needs the lines above the target at the dots to end
# in boxes above it (nothing crosses it), so dot d can reach boxes d+1 .. d+5. A target that has to
# drop more than MAX_TARGET_DROP px is too steep to be crossed 8 times at clean angles, so those
# two pairs (A->5, D->4) are left out. 18 pairs remain, symmetric top-to-bottom.
MAX_TARGET_DROP = 170.0
PAIRS = [(d, b) for d in range(len(DOTS)) for b in range(d + 1, d + NUM_BOXES - N_DISTRACTORS + 1)
         if abs(BOX_Y[b - 1] - DOT_Y[d]) <= MAX_TARGET_DROP]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def curve(ys):
    """Dense drawn path of a line from its control y values (one per column of XS)."""
    return chaikin(np.stack([XS, ys], axis=1), CHAIKIN_ITERS)


def _u():
    return np.clip((XS - XS[1]) / (XS[-2] - XS[1]), 0.0, 1.0)


def wobbly_path(y0, y1, amplitudes, rng):
    """Control ys from y0 (at the dot) to y1 (at the box): a straight line plus sine bends."""
    u = _u()
    a = rng.uniform(-1.0, 1.0, len(amplitudes)) * np.array(amplitudes)
    y = y0 + (y1 - y0) * u + sum(a[k] * np.sin((k + 1) * np.pi * u) for k in range(len(a)))
    y[:2], y[-2:] = y0, y1
    return y


def resample(path, step=1.0):
    """Same points as generate_dataset.densify (consecutive points <= step apart), vectorised."""
    seg = np.diff(path, axis=0)
    n = np.maximum(1, np.ceil(np.hypot(seg[:, 0], seg[:, 1]) / step)).astype(int)
    idx = np.repeat(np.arange(len(seg)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 1) / np.repeat(n, n)
    return np.vstack([path[:1], path[idx] + seg[idx] * t[:, None]])


def max_turn(path):
    p = resample(path)
    h = np.degrees(np.arctan2(np.diff(p[:, 1]), np.diff(p[:, 0])))
    w = 8
    return float(np.abs(h[w:] - h[:-w]).max()) if len(h) > w else 0.0


def crossings_between(p, q):
    """Crossings of two dense paths that share x samples: list of (point, angle_deg).
    None if they touch without crossing."""
    if not np.array_equal(p[:, 0], q[:, 0]):
        raise ValueError("paths must share x samples")
    diff = q[:, 1] - p[:, 1]
    if (diff == 0).any():
        return None
    out = []
    for i in np.nonzero(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]:
        t = diff[i] / (diff[i] - diff[i + 1])
        pt = p[i] + t * (p[i + 1] - p[i])
        a, b = max(0, i - 3), min(len(p) - 1, i + 4)
        vp, vq = p[b] - p[a], q[b] - q[a]
        ang = abs(math.degrees(math.atan2(vp[1], vp[0]) - math.atan2(vq[1], vq[0])))
        out.append((pt, min(ang, 180.0 - ang)))
    return out


def min_separation(a, b, exclude, limit=MIN_SEP):
    """Smallest distance between two resampled left-to-right paths (capped at `limit`), ignoring
    points within CROSS_EXCL of the points in `exclude`. Only nearby x values are compared."""
    if exclude:
        e = np.array(exclude)
        a = a[np.sqrt(((a[:, None] - e[None]) ** 2).sum(-1)).min(1) > CROSS_EXCL]
        b = b[np.sqrt(((b[:, None] - e[None]) ** 2).sum(-1)).min(1) > CROSS_EXCL]
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    lo = np.searchsorted(b[:, 0], a[:, 0] - limit, side="left")
    hi = np.searchsorted(b[:, 0], a[:, 0] + limit, side="right")
    width = int((hi - lo).max())
    if width == 0:
        return float(limit)
    idx = lo[:, None] + np.arange(width)[None, :]
    valid = idx < hi[:, None]
    d2 = ((a[:, None, :] - b[np.minimum(idx, len(b) - 1)]) ** 2).sum(-1)
    return float(min(limit, np.sqrt(np.where(valid, d2, np.inf).min())))


def line_ok(path):
    slope = np.abs(np.diff(path[:, 1]) / np.diff(path[:, 0])).max()
    return (np.diff(path[:, 0]) > 0).all() and slope <= MAX_SLOPE and max_turn(path) <= MAX_TURN


def check_lines(target, others, expected):
    """All human-legibility rules for one version. Returns each other line's crossings with the
    target as [(point, angle)], or None if a rule is broken."""
    for p in others:
        if not line_ok(p):
            return None
    for p, q in combinations(others, 2):                  # other lines never cross each other
        c = crossings_between(p, q)
        if c is None or c:
            return None
    cross = []
    for p, n in zip(others, expected):
        c = crossings_between(target, p)
        if c is None or len(c) != n or any(a < MIN_ANGLE for _, a in c):
            return None
        cross.append(c)
    pts = [pt for c in cross for pt, _ in c]
    if any(math.dist(a, b) < MIN_CROSS_DIST for a, b in combinations(pts, 2)):
        return None
    t1, o1 = resample(target), [resample(p) for p in others]
    for p, c in zip(o1, cross):
        if min_separation(t1, p, [pt for pt, _ in c]) < MIN_SEP:
            return None
    for p, q in combinations(o1, 2):
        if min_separation(p, q, []) < MIN_SEP:
            return None
    return cross


# --------------------------------------------------------------------------- #
# Scene
# --------------------------------------------------------------------------- #
def make_target(y0, y1, rng):
    for _ in range(5000):
        ys = np.round(wobbly_path(y0, y1, TARGET_WOBBLE, rng), 2)
        if ys.min() < 30 or ys.max() > IMG_SIZE - 30:
            continue
        if np.abs(np.diff(ys) / np.diff(XS)).max() > MAX_TARGET_SLOPE:
            continue
        if line_ok(curve(ys)):
            return ys
    raise RuntimeError("could not sample a target line")


def reach_band(y0, y1):
    """Per column: the lowest and highest y a line from y0 (dot) to y1 (box) can reach."""
    from_dot = np.maximum(XS - XS[1], 0.0) * REACH_SLOPE
    to_box = np.maximum(XS[-2] - XS, 0.0) * REACH_SLOPE
    lo = np.maximum.reduce([y0 - from_dot, y1 - to_box, np.full(len(XS), Y_MIN)])
    hi = np.minimum.reduce([y0 + from_dot, y1 + to_box, np.full(len(XS), Y_MAX)])
    return lo, hi


def _planned_angle(target_slope, d):
    """Rough crossing angle when the target passes a line downward (d=+1) or upward (d=-1)."""
    other = target_slope - d * REL_SLOPE
    a = abs(math.degrees(math.atan(target_slope) - math.atan(other)))
    return min(a, 180.0 - a)


def _fits(lo, hi, a, b):
    return max(lo, a) <= min(hi, b)


def sample_walk(k, start, end, ty, bands, rng):
    """How many other lines are above the target at each column ("gap"); it changes by one at each
    crossing. Picks uniformly among all walks with exactly k crossings that fit the geometry.
    Returns (gaps, steps) with steps = [(column, crossed line index)], or None if there is none."""
    t_slope = (np.diff(ty) / np.diff(XS)).tolist()
    first, last = 1, len(XS) - 2                   # columns where the straight runs end
    ty_l = ty.tolist()
    lo = [b[0].tolist() for b in bands]
    hi = [b[1].tolist() for b in bands]
    gaps_all = range(N_DISTRACTORS + 1)
    col_ok = {(c, g): all(_side_ok(j, c, g, ty_l[c], (lo[j], hi[j])) for j in range(N_DISTRACTORS))
              for c in FREE_COLS for g in gaps_all}
    near_a, near_b = GAP_TO_TARGET, CROSS_OFFSET[1]
    crossings = {}                                 # (c, g) -> [(next gap, crossed line)]
    for c in CROSS_INTERVALS:
        for g in gaps_all:
            opts = []
            for d in (1, -1):
                j = g if d == 1 else g - 1         # the line the target passes
                if not 0 <= g + d <= N_DISTRACTORS or _planned_angle(t_slope[c], d) < PLAN_ANGLE:
                    continue
                # line j is below the target before and above after (d=+1), or the reverse
                before = (ty_l[c] + near_a, ty_l[c] + near_b) if d == 1 else (ty_l[c] - near_b, ty_l[c] - near_a)
                after = (ty_l[c + 1] - near_b, ty_l[c + 1] - near_a) if d == 1 else (ty_l[c + 1] + near_a, ty_l[c + 1] + near_b)
                if _fits(lo[j][c], hi[j][c], *before) and _fits(lo[j][c + 1], hi[j][c + 1], *after):
                    opts.append((g + d, j))
            crossings[(c, g)] = opts

    def moves(c, g, recent):
        """(next gap, crossed line or None) for the step from column c to c+1. `recent` holds the
        lines crossed in the last SAME_LINE_GAP-1 steps; they may not be crossed again yet."""
        return [(g, None)] + [(ng, j) for ng, j in crossings.get((c, g), ()) if j not in recent]

    memo = {}

    def count(c, g, recent, rem):
        key = (c, g, recent, rem)
        if key not in memo:
            if rem < 0 or not col_ok.get((c, g), True):
                memo[key] = 0
            elif c == last:
                memo[key] = int(rem == 0 and g == end)
            else:
                memo[key] = sum(count(c + 1, ng, (j,) + recent[:-1], rem - (j is not None))
                                for ng, j in moves(c, g, recent))
        return memo[key]

    recent = (None,) * (SAME_LINE_GAP - 1)
    if count(first, start, recent, k) == 0:
        return None
    gaps = np.full(len(XS), start)
    steps, g, rem = [], start, k
    for c in range(first, last):
        opts = moves(c, g, recent)
        w = np.array([count(c + 1, ng, (j,) + recent[:-1], rem - (j is not None)) for ng, j in opts], float)
        ng, j = opts[rng.choice(len(opts), p=w / w.sum())]
        if j is not None:
            steps.append((c, j))
            rem -= 1
        g, recent = ng, (j,) + recent[:-1]
        gaps[c + 1:] = g
    return gaps, steps


def _side_ok(j, c, g, t, band):
    """Can line j sit on its side of the target at column c, with the lines between them?"""
    lo, hi = band
    if j < g:
        return lo[c] <= t - GAP_TO_TARGET - (g - 1 - j) * GAP_BETWEEN
    return hi[c] >= t + GAP_TO_TARGET + (j - g) * GAP_BETWEEN


def place_others(ty, gaps, steps, dot_ys, box_ys, bands, rng):
    """Control ys of the other lines: they follow their own wobbly path but keep to the side of the
    target given by `gaps`, stay apart, come close to the target where they cross it, and stay
    inside their reach band (so they can still get from their dot to their box)."""
    n = len(XS)
    homes = np.array([np.clip(wobbly_path(dot_ys[j], box_ys[j], OTHER_WOBBLE, rng), *bands[j])
                      for j in range(N_DISTRACTORS)])
    close = rng.uniform(*CROSS_OFFSET, size=(N_DISTRACTORS, n))
    near = np.full((N_DISTRACTORS, n), np.inf)      # columns to this line's nearest crossing
    cols = np.arange(n)
    pinned = np.zeros((N_DISTRACTORS, n), bool)    # the two columns around each crossing
    for c, j in steps:
        near[j] = np.minimum(near[j], np.where(cols <= c, c - cols, cols - c - 1))
        pinned[j, c] = pinned[j, c + 1] = True
    lo = np.array([b[0] for b in bands])
    hi = np.array([b[1] for b in bands])

    def project(pref):
        """Move preferred positions to the nearest positions that obey every placement rule."""
        ys = pref.copy()
        for c in FREE_COLS:
            t, g = ty[c], gaps[c]
            lim, soft = t - GAP_TO_TARGET, t - PREF_GAP
            for j in range(g - 1, -1, -1):          # above the target, nearest first
                # don't go so far up that the lines further out leave their reach band
                floor = max([lo[j, c]] + [lo[i, c] + (j - i) * GAP_BETWEEN for i in range(j)])
                y = pref[j, c] if pinned[j, c] else min(pref[j, c], soft)
                if np.isfinite(near[j, c]):
                    y = max(y, t - close[j, c] - APPROACH * near[j, c])
                ys[j, c] = y = min(min(max(y, floor), hi[j, c]), lim)
                lim, soft = y - GAP_BETWEEN, y - PREF_GAP
            lim, soft = t + GAP_TO_TARGET, t + PREF_GAP
            for j in range(g, N_DISTRACTORS):       # below the target, nearest first
                ceil = min([hi[j, c]] + [hi[i, c] - (i - j) * GAP_BETWEEN for i in range(j + 1, N_DISTRACTORS)])
                y = pref[j, c] if pinned[j, c] else max(pref[j, c], soft)
                if np.isfinite(near[j, c]):
                    y = min(y, t + close[j, c] + APPROACH * near[j, c])
                ys[j, c] = y = max(max(min(y, ceil), lo[j, c]), lim)
                lim, soft = y + GAP_BETWEEN, y + PREF_GAP
        return ys

    ys = project(homes)
    for _ in range(SMOOTH_PASSES):                  # smooth out bumps, keeping the crossings in place
        smooth = ys.copy()
        smooth[:, 2:-2] = 0.25 * ys[:, 1:-3] + 0.5 * ys[:, 2:-2] + 0.25 * ys[:, 3:-1]
        smooth[pinned] = ys[pinned]
        ys = project(smooth)
    if (ys < lo - 0.5).any() or (ys > hi + 0.5).any():
        return None
    return np.round(ys, 2)


def make_version(k, ty, target_dot, target_box, dot_ys, rng, tries=300, max_no_walk=15):
    """The 3 other lines for one version, crossing the target exactly k times in total.
    None if it fails (then the caller draws a new target line)."""
    other_dots = [d for d in range(len(DOTS)) if d != target_dot]
    above_boxes, below_boxes = list(range(1, target_box)), list(range(target_box + 1, NUM_BOXES + 1))
    # lines above the target at the boxes: must fit the boxes, and k crossings change it by an even step
    ends = [r for r in range(N_DISTRACTORS + 1) if r <= len(above_boxes) and N_DISTRACTORS - r <= len(below_boxes)
            and abs(r - target_dot) <= k and (r - target_dot) % 2 == k % 2]
    target = curve(ty)
    no_walk = 0
    for _ in range(tries):
        r = ends[rng.integers(len(ends))]
        boxes = sorted(rng.choice(above_boxes, size=r, replace=False).tolist()) if r else []
        boxes += sorted(rng.choice(below_boxes, size=N_DISTRACTORS - r, replace=False).tolist()) \
            if r < N_DISTRACTORS else []
        dys, bys = [dot_ys[d] for d in other_dots], [BOX_Y[b - 1] for b in boxes]
        bands = [reach_band(dys[j], bys[j]) for j in range(N_DISTRACTORS)]
        walk = sample_walk(k, target_dot, r, ty, bands, rng)
        if walk is None:
            no_walk += 1
            if no_walk >= max_no_walk:             # this target line can't take k crossings
                return None
            continue
        gaps, steps = walk
        ys = place_others(ty, gaps, steps, dys, bys, bands, rng)
        if ys is None:
            continue
        expected = [sum(j == i for _, j in steps) for i in range(N_DISTRACTORS)]
        cross = check_lines(target, [curve(y) for y in ys], expected)
        if cross is None:
            continue
        return [dict(from_dot=other_dots[i], to_box=boxes[i], ys=ys[i], crossings=cross[i])
                for i in range(N_DISTRACTORS)]
    return None


def make_scene(scene_id, target_dot, target_box, split, seed):
    rng = np.random.default_rng([seed, scene_id, RNG_TAG])
    dot_ys = np.round(np.array(DOT_Y) + rng.uniform(-DOT_JITTER, DOT_JITTER, len(DOTS)), 2)
    colors = [str(c) for c in rng.choice(DISTRACTOR_COLORS, size=N_DISTRACTORS, replace=False)]
    for _attempt in range(500):
        ty = make_target(dot_ys[target_dot], BOX_Y[target_box - 1], rng)
        others = {}
        for v in sorted(DRAWN_VERSIONS, key=lambda v: -CROSSINGS[v]):     # hardest first: fail fast
            lines = make_version(CROSSINGS[v], ty, target_dot, target_box, dot_ys, rng)
            if lines is None:
                break
            others[v] = [dict(o, color=LINE_COLOR) for o in lines]
        else:
            for v, base in RECOLORED.items():
                others[v] = [dict(o, color=c) for o, c in zip(others[base], colors)]
            return dict(scene_id=scene_id, split=split, dot_ys=dot_ys, target_dot=target_dot,
                        target_box=target_box, target_ys=ty, others=others)
    raise RuntimeError(f"scene {scene_id}: could not draw all versions")


# --------------------------------------------------------------------------- #
# Rendering + metadata
# --------------------------------------------------------------------------- #
_FONTS: dict[int, ImageFont.FreeTypeFont] = {}


def _font(px):
    return _FONTS.setdefault(px, ImageFont.load_default(size=px))


def layout_objects(dot_ys):
    S = SUPERSAMPLE
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    objs = []
    for i, name in enumerate(DOTS):
        x0, y0, x1, y1 = probe.textbbox((LABEL_X * S, dot_ys[i] * S), name, font=_font(LABEL_PX * S), anchor="mm")
        objs.append({"id": i, "role": "layout", "shape": "dot", "label": name,
                     "center": [DOT_X, float(dot_ys[i])], "radius": DOT_R,
                     "label_bbox": [x0 / S, y0 / S, x1 / S, y1 / S]})
    for j, y in enumerate(BOX_Y):
        objs.append({"id": len(DOTS) + j, "role": "layout", "shape": "box", "label": str(j + 1),
                     "center": [(BOX_X0 + BOX_X1) / 2, y], "bbox": [BOX_X0, y - BOX_H / 2, BOX_X1, y + BOX_H / 2]})
    return objs


def draw_layout(draw, dot_ys):
    S = SUPERSAMPLE
    for i, name in enumerate(DOTS):
        x, y, r = DOT_X * S, dot_ys[i] * S, DOT_R * S
        draw.ellipse([x - r, y - r, x + r, y + r], fill=INK)
        draw.text((LABEL_X * S, y), name, font=_font(LABEL_PX * S), fill=INK, anchor="mm")
    for j, y in enumerate(BOX_Y):
        draw.rectangle([BOX_X0 * S, (y - BOX_H / 2) * S, BOX_X1 * S, (y + BOX_H / 2) * S],
                       fill=WHITE, outline=INK, width=BOX_BORDER * S)
        draw.text(((BOX_X0 + BOX_X1) / 2 * S, y * S), str(j + 1), font=_font(BOX_PX * S), fill=INK, anchor="mm")


def line_object(idx, role, from_dot, to_box, ys, color, crossings=None):
    path = curve(ys)
    pad = STROKE / 2 + 1
    o = {"id": idx, "role": role, "shape": "line", "from_dot": DOTS[from_dot], "to_box": int(to_box),
         "color": color, "points": [[float(x), float(y)] for x, y in zip(XS, ys)],
         "bbox": [max(0, math.floor(path[:, 0].min() - pad)), max(0, math.floor(path[:, 1].min() - pad)),
                  min(IMG_SIZE, math.ceil(path[:, 0].max() + pad)), min(IMG_SIZE, math.ceil(path[:, 1].max() + pad))]}
    if crossings is not None:
        o["num_crossings"] = len(crossings)
        o["crossings"] = [{"at": [round(float(p[0]), 1), round(float(p[1]), 1)], "angle_deg": round(a, 1)}
                          for p, a in crossings]
    return o


def render_scene(scene, image_dir):
    big = IMG_SIZE * SUPERSAMPLE
    base = Image.new("RGB", (big, big), WHITE)
    draw_path(ImageDraw.Draw(base), curve(scene["target_ys"]), PALETTE[LINE_COLOR])
    sid, td, tb = scene["scene_id"], scene["target_dot"], scene["target_box"]
    layout = layout_objects(scene["dot_ys"])
    target = line_object(len(layout), "target", td, tb, scene["target_ys"], LINE_COLOR)

    rows = []
    for difficulty, version in enumerate(VERSIONS):
        img = base.copy()
        draw = ImageDraw.Draw(img)
        others = scene["others"].get(version, [])
        for o in others:
            draw_path(draw, curve(o["ys"]), PALETTE[o["color"]])
        draw_layout(draw, scene["dot_ys"])
        image_id = f"{ID_PREFIX}_{sid:04d}_{version}"
        img.reduce(SUPERSAMPLE).save(image_dir / f"{image_id}.png", optimize=True)

        objs = [line_object(target["id"] + 1 + i, "distractor", o["from_dot"], o["to_box"], o["ys"],
                            o["color"], o["crossings"]) for i, o in enumerate(others)]
        rows.append({
            "image_id": image_id,
            "file_name": f"images/{image_id}.png",
            "scene_id": sid,
            "version": version,
            "difficulty": difficulty,
            "split": scene["split"],
            "question": QUESTION_TEMPLATE.format(dot=DOTS[td]),
            "answer": f"box: {tb}",
            "target_dot": DOTS[td],
            "target_box": tb,
            "num_crossings": sum(o["num_crossings"] for o in objs),
            "num_distractors": len(objs),
            "distractor_boxes": ",".join(str(b) for b in sorted(o["to_box"] for o in objs)),
            "crossed_boxes": ",".join(str(b) for b in sorted(o["to_box"] for o in objs if o["num_crossings"])),
            # does the target end at the same rank among the line ends as it starts among the dots?
            # (if yes, "the line goes straight across" guesses right without tracing)
            "rank_kept": "" if not objs else ("yes" if sum(o["to_box"] < tb for o in objs) == td else "no"),
            "objects": json.dumps(layout + [target] + objs),
        })
    return rows


def build_scene(task):
    """One scene -> its metadata rows (runs in a worker process)."""
    scene_id, td, tb, split, seed, image_dir = task
    return render_scene(make_scene(scene_id, td, tb, split, seed), Path(image_dir))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", type=int, default=500, help="number of scenes (images = 6 x scenes)")
    ap.add_argument("--out", default=str(ROOT / "data" / "lines_vqa"), help="output folder")
    ap.add_argument("--seed", type=int, default=42, help="random seed (same seed -> same dataset)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="parallel processes (the output does not depend on this)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output folder")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"'{out}' already exists. Re-run with --overwrite to replace it.")
        shutil.rmtree(out)
    image_dir = out / "images"
    image_dir.mkdir(parents=True)

    # Balanced design: every (target dot, target box) pair equally often, and splits exactly
    # 70/15/15 (the usual 20-scene pattern), shuffled so that no pair is tied to a split.
    design_rng = np.random.default_rng([args.seed, RNG_TAG])
    splits = [SPLIT_PATTERN[i % len(SPLIT_PATTERN)] for i in range(args.scenes)]
    splits = [splits[i] for i in design_rng.permutation(args.scenes)]
    design = [(PAIRS[i % len(PAIRS)], splits[i]) for i in range(args.scenes)]
    order = design_rng.permutation(args.scenes)

    tasks = [(scene_id, *design[idx][0], design[idx][1], args.seed, str(image_dir))
             for scene_id, idx in enumerate(order, start=1)]
    rows = []
    with Pool(args.workers) as pool:             # imap keeps scene order, so the output is deterministic
        for n, scene_rows in enumerate(pool.imap(build_scene, tasks), start=1):
            rows += scene_rows
            if n % 50 == 0 or n == args.scenes:
                print(f"  {n}/{args.scenes} scenes  ({len(rows)} images)", flush=True)

    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    split_counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")}
    info = {
        "name": DATASET_NAME,
        "title": "Lines-VQA",
        "version": DATASET_VERSION,
        "created": date.today().isoformat(),
        "summary": ("Follow the line from a labelled dot to a numbered box. Each scene is drawn in 6 versions "
                    "with the same correct answer; versions add other lines that cross the target line "
                    "0, 2, 4 or 8 times, or colour those lines."),
        "num_scenes": args.scenes,
        "num_images": len(rows),
        "images_per_split": split_counts,
        "versions": {v: {"difficulty": i, "description": VERSION_DESCRIPTIONS[v]} for i, v in enumerate(VERSIONS)},
        "question_template": QUESTION_TEMPLATE,
        "answer_format": "box: <number>",
        "config": {
            # used by validate_dataset.py (task "trace" selects the line checks)
            "task": "trace", "seed": args.seed, "id_prefix": ID_PREFIX, "image_size": IMG_SIZE,
            "stroke_px": STROKE, "versions": VERSIONS, "dots": DOTS, "num_boxes": NUM_BOXES,
            "crossings_per_version": CROSSINGS, "distractors_per_version": {v: (0 if v == "clean" else N_DISTRACTORS) for v in VERSIONS},
            "recolored_versions": RECOLORED, "line_color": LINE_COLOR, "distractor_colors": DISTRACTOR_COLORS,
            "palette": PALETTE, "chaikin_iterations": CHAIKIN_ITERS, "columns_x": XS.tolist(),
            "min_crossing_angle_deg": MIN_ANGLE, "min_crossing_distance_px": MIN_CROSS_DIST,
            "min_line_separation_px": MIN_SEP, "crossing_exclusion_px": CROSS_EXCL,
            "max_slope": MAX_SLOPE, "max_turn_deg": MAX_TURN, "y_range": [Y_MIN, Y_MAX],
            # layout
            "dot_x": DOT_X, "dot_radius": DOT_R, "box_x": [BOX_X0, BOX_X1], "box_height": BOX_H,
            "box_y": BOX_Y, "label_px": LABEL_PX, "box_number_px": BOX_PX,
        },
    }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Done: {len(rows)} images in '{out}'  {split_counts}")
    print(f"Next: python validate_dataset.py --data {out}")


if __name__ == "__main__":
    main()
