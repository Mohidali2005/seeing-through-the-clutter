#!/usr/bin/env python
"""
Check that the generated dataset is correct, then save a preview sheet.

Works for every dataset; the rules come from dataset_info.json. Lines-VQA (config.task == "trace") has its
own checks in validate_trace(): crossings, angles, spacing and separation recomputed from each line's points.
Labels are checked against the metadata AND against the actual image pixels:
  * every scene has all versions, with the same split / question / answer
  * answer string == per-class counts == the target objects listed in `objects`
  * target shapes are identical in all versions (original: distractor slots identical in versions 1-3;
    hard: "everything" contains exactly the traps of the other versions)
  * hard: targets have the target colour, no perfect target-colour shape is a distractor, and no text
    trap states the true count
  * no two objects touch; every object is inside the image
  * pixels: outside the distractor areas, each distracted image is IDENTICAL to its clean image
  * pixels: there is no ink anywhere except where an object is listed, and every listed object is visible

Usage
    python scripts/validate_dataset.py                               # checks data/shapes_vqa_dataset/
    python scripts/validate_dataset.py --data data/shapes_vqa_hard   # checks the hard dataset
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DISTRACTOR_SHAPE = {"squiggles": "squiggle", "blobs": "blob", "lookalikes": "lookalike"}
INK_MARGIN = 2.0     # anti-aliasing allowance (px) beyond the stroke
MIN_VISIBLE_PX = 30  # every object must have at least this many clearly coloured pixels


def _count_claim(text, classes):
    """Parse a text trap that claims a count: 'star: 4', '3 stars' -> ('star', 4); '5' -> (None, 5)."""
    t = text.strip().lower()
    if t.isdigit():
        return None, int(t)
    for c in classes:
        m = re.fullmatch(rf"{c}s?:\s*(\d+)", t) or re.fullmatch(rf"(\d+)\s+{c}s?", t)
        if m:
            return c, int(m.group(1))
    return None


class Report:
    def __init__(self):
        self.errors = []

    def check(self, ok, msg):
        if not ok:
            self.errors.append(msg)
        return ok


def disc_window(obj, extra, size):
    """(row slice, col slice, local boolean disc mask) for an object's slot circle."""
    cx, cy = obj["center"]
    r = obj["radius"] + extra
    x0, x1 = max(0, math.floor(cx - r)), min(size, math.ceil(cx + r) + 1)
    y0, y1 = max(0, math.floor(cy - r)), min(size, math.ceil(cy + r) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= r * r
    return slice(y0, y1), slice(x0, x1), mask


def union_mask(objects, extra, size):
    m = np.zeros((size, size), bool)
    for o in objects:
        ys, xs, local = disc_window(o, extra, size)
        m[ys, xs] |= local
    return m


def visible_px(arr, obj, extra, size):
    ys, xs, local = disc_window(obj, extra, size)
    dark = arr[ys, xs].min(axis=2) < 200
    return int((dark & local).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data" / "shapes_vqa_dataset"))
    ap.add_argument("--preview-scenes", type=int, default=6, help="scenes shown in preview.png (0 = none)")
    ap.add_argument("--skip-pixels", action="store_true", help="only check metadata (fast)")
    args = ap.parse_args()

    root = Path(args.data)
    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    cfg = info["config"]
    if cfg.get("task") == "trace":                 # Lines-VQA: follow-the-line, not counting
        return validate_trace(args, root, info)
    classes, versions, size = cfg["classes"], cfg["versions"], cfg["image_size"]
    extra = cfg["stroke_px"] / 2 + INK_MARGIN
    # dataset-specific rules (defaults = the original Shapes-VQA dataset)
    id_prefix = cfg.get("id_prefix", "scene")
    allowed_shapes = cfg.get("distractor_shapes") or {v: [DISTRACTOR_SHAPE[v]] for v in versions[1:]}
    allowed_counts = cfg.get("distractors_per_version") or {v: cfg["distractor_counts"] for v in versions[1:]}
    same_slot_versions = cfg.get("same_slot_versions", versions[1:])
    union_versions = cfg.get("union_versions", {})

    def expected_question(row):
        if "question_template" in info:
            return info["question_template"].format(color=row["target_color"])
        return info["question"]

    with open(root / "metadata.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rep = Report()

    # ---------------- metadata checks ----------------
    rep.check(len(rows) == info["num_images"], f"metadata has {len(rows)} rows, dataset_info says {info['num_images']}")
    ids = [r["image_id"] for r in rows]
    rep.check(len(ids) == len(set(ids)), "duplicate image_id values")

    scenes = defaultdict(dict)
    for r in rows:
        r["objects"] = json.loads(r["objects"])
        scenes[int(r["scene_id"])][r["version"]] = r
    rep.check(len(scenes) == info["num_scenes"], f"{len(scenes)} scenes found, expected {info['num_scenes']}")

    for sid, vers in sorted(scenes.items()):
        tag = f"scene {sid}"
        if not rep.check(sorted(vers) == sorted(versions), f"{tag}: versions {sorted(vers)}"):
            continue
        clean = vers["clean"]
        for d, v in enumerate(versions):
            r = vers[v]
            rep.check(int(r["difficulty"]) == d, f"{tag}/{v}: difficulty {r['difficulty']} != {d}")
            rep.check(r["image_id"] == f"{id_prefix}_{sid:04d}_{v}", f"{tag}/{v}: image_id {r['image_id']}")
            rep.check(r["file_name"] == f"images/{r['image_id']}.png", f"{tag}/{v}: file_name {r['file_name']}")
            rep.check(r["question"] == expected_question(r), f"{tag}/{v}: question text differs")
            same_keys = ["split", "answer", "count_total", *[f"count_{c}" for c in classes]]
            if "target_color" in r:
                same_keys.append("target_color")
            for key in same_keys:
                rep.check(r[key] == clean[key], f"{tag}/{v}: {key} differs from clean version")

            # answer <-> counts <-> objects
            targets = [o for o in r["objects"] if o["role"] == "target"]
            dists = [o for o in r["objects"] if o["role"] == "distractor"]
            recount = Counter(o["shape"] for o in targets)
            rep.check(set(recount) <= set(classes), f"{tag}/{v}: unknown target shape {set(recount)}")
            for c in classes:
                rep.check(int(r[f"count_{c}"]) == recount[c], f"{tag}/{v}: count_{c}={r[f'count_{c}']} but {recount[c]} targets")
                rep.check(int(r[f"count_{c}"]) <= cfg["max_per_class"], f"{tag}/{v}: count_{c} > max_per_class")
            rep.check(int(r["count_total"]) == len(targets), f"{tag}/{v}: count_total mismatch")
            rep.check(len(targets) in cfg["target_counts"], f"{tag}/{v}: {len(targets)} targets not allowed")
            expected_answer = ", ".join(f"{c}: {recount[c]}" for c in classes)
            rep.check(r["answer"] == expected_answer, f"{tag}/{v}: answer '{r['answer']}' != '{expected_answer}'")
            rep.check(int(r["num_distractors"]) == len(dists), f"{tag}/{v}: num_distractors mismatch")

            if v == "clean":
                rep.check(len(dists) == 0, f"{tag}/clean: has distractors")
            else:
                rep.check(len(dists) in allowed_counts[v], f"{tag}/{v}: {len(dists)} distractors not allowed")
                rep.check(all(o["shape"] in allowed_shapes[v] for o in dists), f"{tag}/{v}: wrong distractor type")
            for o in dists:
                if o["shape"] == "lookalike":
                    rep.check(o["lookalike_of"] in classes and o["subtype"] in ("gap", "wavy"),
                              f"{tag}/{v}: bad lookalike fields on object {o['id']}")

            # colour task: targets are the target colour, and a perfect named shape that is a
            # distractor must NOT be the target colour. Other tasks: a distractor is never a target class.
            if "target_color" in r:
                tc = r["target_color"]
                rep.check(all(o["color"] == tc for o in targets), f"{tag}/{v}: target with wrong colour")
                rep.check(tc in r["question"], f"{tag}/{v}: target colour missing from question")
                for o in dists:
                    if o["shape"] in classes:
                        rep.check(o["color"] != tc, f"{tag}/{v}: distractor {o['id']} is a perfect {tc} {o['shape']}")
            else:
                for o in dists:
                    rep.check(o["shape"] not in classes, f"{tag}/{v}: distractor {o['id']} is a real {o['shape']}")
            # look-alikes must imitate a real class (and be a listed relative of it, if the dataset lists them)
            for o in dists:
                if o.get("subtype") == "lookalike":
                    of = o.get("lookalike_of")
                    rep.check(of in classes, f"{tag}/{v}: lookalike {o['id']} imitates unknown class {of}")
                    if "lookalikes" in cfg and of in cfg["lookalikes"]:
                        rep.check(o["shape"] in cfg["lookalikes"][of], f"{tag}/{v}: {o['shape']} is not a listed look-alike of {of}")
            # a text trap must never state the true count
            for o in dists:
                if o["shape"] == "text":
                    claim = _count_claim(o["text"], classes)
                    if claim is not None:
                        cls, n = claim
                        true_n = recount[cls] if cls else len(targets)
                        rep.check(n != true_n, f"{tag}/{v}: text '{o['text']}' states the true count")

            # same named shapes in every version
            rep.check(json.dumps(targets, sort_keys=True) ==
                      json.dumps([o for o in clean["objects"] if o["role"] == "target"], sort_keys=True),
                      f"{tag}/{v}: named shapes differ from clean version")

            # geometry: inside image, nothing touching
            for o in r["objects"]:
                (cx, cy), rad = o["center"], o["radius"]
                rep.check(rad <= cx <= size - rad and rad <= cy <= size - rad, f"{tag}/{v}: object {o['id']} outside image")
                x0, y0, x1, y1 = o["bbox"]
                rep.check(0 <= x0 < x1 <= size and 0 <= y0 < y1 <= size, f"{tag}/{v}: object {o['id']} bad bbox")
            for a, b in combinations(r["objects"], 2):
                gap = math.dist(a["center"], b["center"]) - a["radius"] - b["radius"]
                rep.check(gap >= cfg["min_gap_px"] - 0.25, f"{tag}/{v}: objects {a['id']},{b['id']} too close ({gap:.1f}px)")

        # distractors sit in the same slots with the same colours (original dataset, versions 1-3)
        if len(same_slot_versions) > 1:
            slot_sets = [[(o["center"], o["radius"], o["color"]) for o in vers[v]["objects"] if o["role"] == "distractor"]
                         for v in same_slot_versions]
            rep.check(all(s == slot_sets[0] for s in slot_sets), f"{tag}: distractor slots differ between versions")
        # a combined version must contain exactly the distractors of its component versions
        for v, parts in union_versions.items():
            got = sorted(json.dumps(o, sort_keys=True) for o in vers[v]["objects"] if o["role"] == "distractor")
            want = sorted(json.dumps(o, sort_keys=True) for p in parts for o in vers[p]["objects"] if o["role"] == "distractor")
            rep.check(got == want, f"{tag}/{v}: distractors are not exactly those of {parts}")

    # split hygiene
    split_of_scene = {sid: vers["clean"]["split"] for sid, vers in scenes.items() if "clean" in vers}
    rep.check(set(split_of_scene.values()) <= {"train", "validation", "test"}, "unknown split name")

    # ---------------- pixel checks ----------------
    if not args.skip_pixels:
        for n, (sid, vers) in enumerate(sorted(scenes.items()), start=1):
            tag = f"scene {sid}"
            arrays = {}
            for v, r in vers.items():
                path = root / r["file_name"]
                if not rep.check(path.exists(), f"{tag}/{v}: missing file {path}"):
                    continue
                with Image.open(path) as im:
                    rep.check(im.size == (size, size), f"{tag}/{v}: size {im.size}")
                    rep.check(im.mode == "RGB", f"{tag}/{v}: mode {im.mode}")
                    arrays[v] = np.asarray(im.convert("RGB"))
            if len(arrays) != len(versions):
                continue
            clean_arr = arrays["clean"]
            targets = [o for o in vers["clean"]["objects"] if o["role"] == "target"]
            ink = (clean_arr != 255).any(axis=2)
            rep.check(not ink[~union_mask(targets, extra, size)].any(), f"{tag}/clean: ink outside the named shapes")
            for o in targets:
                rep.check(visible_px(clean_arr, o, extra, size) >= MIN_VISIBLE_PX, f"{tag}/clean: target {o['id']} not visible")
            for v in versions[1:]:
                arr = arrays[v]
                dists = [o for o in vers[v]["objects"] if o["role"] == "distractor"]
                dmask = union_mask(dists, extra, size)
                rep.check(np.array_equal(arr[~dmask], clean_arr[~dmask]),
                          f"{tag}/{v}: pixels differ from clean image outside distractor areas")
                for o in dists:
                    rep.check(visible_px(arr, o, extra, size) >= MIN_VISIBLE_PX, f"{tag}/{v}: distractor {o['id']} not visible")
            if n % 100 == 0:
                print(f"  pixel-checked {n}/{len(scenes)} scenes")

    # ---------------- summary ----------------
    print(f"\nDataset: {root.resolve()}")
    print(f"  {len(scenes)} scenes, {len(rows)} images")
    split_counts = Counter(split_of_scene.values())
    print("  scenes per split: " + ", ".join(f"{s}={split_counts[s]}" for s in ("train", "validation", "test")))
    print("  images per version: " + ", ".join(f"{v}={sum(r['version'] == v for r in rows)}" for v in versions))
    cleans = [vers["clean"] for vers in scenes.values() if "clean" in vers]
    print("  targets per image: " + str(dict(sorted(Counter(int(r["count_total"]) for r in cleans).items()))))
    for v in versions[1:]:
        dist = Counter(int(vers[v]["num_distractors"]) for vers in scenes.values() if v in vers)
        print(f"  distractors per image ({v}): " + str(dict(sorted(dist.items()))))
    print("  total targets by class: " + ", ".join(f"{c}={sum(int(r[f'count_{c}']) for r in cleans)}" for c in classes))
    if cleans and "target_color" in cleans[0]:
        print("  scenes per target colour: " + str(dict(sorted(Counter(r["target_color"] for r in cleans).items()))))
    if "lookalikes" in versions:
        looks = [o for vers in scenes.values() if "lookalikes" in vers for o in vers["lookalikes"]["objects"] if o["role"] == "distractor"]
        print("  lookalikes imitate: " + str(dict(Counter(o["lookalike_of"] for o in looks))) +
              "  kinds: " + str(dict(Counter(o["subtype"] if o["shape"] == "lookalike" else o["shape"] for o in looks))))
    if "text_traps" in versions:
        texts = [o for vers in scenes.values() if "text_traps" in vers for o in vers["text_traps"]["objects"] if o["role"] == "distractor"]
        print("  text traps by kind: " + str(dict(Counter(o["subtype"] for o in texts))))
    if "other_shapes" in versions:
        others = [o for vers in scenes.values() if "other_shapes" in vers for o in vers["other_shapes"]["objects"] if o["role"] == "distractor"]
        print("  other shapes by kind: " + str(dict(Counter(o["shape"] for o in others))))

    if args.preview_scenes > 0:
        out = save_preview(root, scenes, versions, size, args.preview_scenes)
        print(f"  preview saved: {out}")

    if rep.errors:
        print(f"\nFAILED: {len(rep.errors)} problem(s). First ones:")
        for e in rep.errors[:30]:
            print("  - " + e)
        sys.exit(1)
    print("\nALL CHECKS PASSED" + (" (metadata only)" if args.skip_pixels else ""))


# --------------------------------------------------------------------------- #
# Lines-VQA ("trace" task): follow a line from a dot to a box
# --------------------------------------------------------------------------- #
def _resample(path, step=1.0):
    """Insert points so consecutive points are <= step px apart."""
    seg = np.diff(path, axis=0)
    n = np.maximum(1, np.ceil(np.hypot(seg[:, 0], seg[:, 1]) / step)).astype(int)
    idx = np.repeat(np.arange(len(seg)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 1) / np.repeat(n, n)
    return np.vstack([path[:1], path[idx] + seg[idx] * t[:, None]])


def _line_crossings(p, q):
    """Crossings of two drawn paths that share x samples: [(point, angle_deg)], None if they touch."""
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


def _min_distance(a, b, exclude, radius, limit):
    """Smallest distance (capped at `limit`) between two resampled left-to-right paths, ignoring
    points within `radius` of the points in `exclude`."""
    if exclude:
        e = np.array(exclude)
        a = a[np.sqrt(((a[:, None] - e[None]) ** 2).sum(-1)).min(1) > radius]
        b = b[np.sqrt(((b[:, None] - e[None]) ** 2).sum(-1)).min(1) > radius]
    if len(a) == 0 or len(b) == 0:
        return float(limit)
    lo = np.searchsorted(b[:, 0], a[:, 0] - limit, side="left")
    hi = np.searchsorted(b[:, 0], a[:, 0] + limit, side="right")
    width = int((hi - lo).max())
    if width == 0:
        return float(limit)
    idx = lo[:, None] + np.arange(width)[None, :]
    d2 = ((a[:, None, :] - b[np.minimum(idx, len(b) - 1)]) ** 2).sum(-1)
    return float(min(limit, np.sqrt(np.where(idx < hi[:, None], d2, np.inf).min())))


def _channel_min(arr):
    """Per-pixel min over R, G, B (much faster than arr.min(axis=2) on HxWx3 uint8)."""
    return np.minimum(np.minimum(arr[..., 0], arr[..., 1]), arr[..., 2])


def _stroke_mask(path, half_width, size, ss=2):
    """Pixels within ~half_width of a polyline (drawn at ss x, any coverage counts)."""
    im = Image.new("L", (size * ss, size * ss), 0)
    d = ImageDraw.Draw(im)
    pts = [tuple(p) for p in (path * ss).tolist()]
    r = half_width * ss
    d.line(pts, fill=255, width=max(1, round(2 * r)))
    for x, y in pts:
        d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    return np.asarray(im.reduce(ss)) > 0            # mean > 0  <=>  any covered sub-pixel


def _layout_mask(layout, extra, size):
    """Pixels of the dots (discs), their letters and the boxes (rectangles), plus `extra` px."""
    m = np.zeros((size, size), bool)
    for o in layout:
        rects = [o["label_bbox"]] if o["shape"] == "dot" else [o["bbox"]]
        if o["shape"] == "dot":
            obj = dict(center=o["center"], radius=o["radius"])
            ys, xs, local = disc_window(obj, extra, size)
            m[ys, xs] |= local
        for x0, y0, x1, y1 in rects:
            m[max(0, math.floor(y0 - extra)):min(size, math.ceil(y1 + extra) + 1),
              max(0, math.floor(x0 - extra)):min(size, math.ceil(x1 + extra) + 1)] = True
    return m


def validate_trace(args, root, info):
    from generate_dataset import chaikin                    # the drawn curve = chaikin(points)

    cfg = info["config"]
    versions, size, dots, nbox = cfg["versions"], cfg["image_size"], cfg["dots"], cfg["num_boxes"]
    cols_x = cfg["columns_x"]
    extra = cfg["stroke_px"] / 2 + INK_MARGIN
    recolored = cfg["recolored_versions"]
    palette = {k: tuple(v) for k, v in cfg["palette"].items()}

    def drawn(o):
        return chaikin(np.array(o["points"], float), cfg["chaikin_iterations"])

    with open(root / "metadata.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rep = Report()
    rep.check(len(rows) == info["num_images"], f"metadata has {len(rows)} rows, dataset_info says {info['num_images']}")
    ids = [r["image_id"] for r in rows]
    rep.check(len(ids) == len(set(ids)), "duplicate image_id values")
    scenes = defaultdict(dict)
    for r in rows:
        r["objects"] = json.loads(r["objects"])
        scenes[int(r["scene_id"])][r["version"]] = r
    rep.check(len(scenes) == info["num_scenes"], f"{len(scenes)} scenes found, expected {info['num_scenes']}")

    stats = defaultdict(list)
    for sid, vers in sorted(scenes.items()):
        tag = f"scene {sid}"
        if not rep.check(sorted(vers) == sorted(versions), f"{tag}: versions {sorted(vers)}"):
            continue
        clean = vers["clean"]
        clean_layout = [o for o in clean["objects"] if o["role"] == "layout"]
        clean_target = [o for o in clean["objects"] if o["role"] == "target"]
        # layout: dots A-D at their x, boxes 1..N at their slots
        ldots = [o for o in clean_layout if o["shape"] == "dot"]
        lboxes = [o for o in clean_layout if o["shape"] == "box"]
        rep.check([o["label"] for o in ldots] == dots, f"{tag}: dot labels {[o['label'] for o in ldots]}")
        rep.check([o["label"] for o in lboxes] == [str(i + 1) for i in range(nbox)], f"{tag}: box labels")
        rep.check(all(o["center"][0] == cfg["dot_x"] for o in ldots), f"{tag}: dot not at dot_x")
        for o, y in zip(lboxes, cfg["box_y"]):
            rep.check(o["bbox"] == [cfg["box_x"][0], y - cfg["box_height"] / 2, cfg["box_x"][1], y + cfg["box_height"] / 2],
                      f"{tag}: box {o['label']} misplaced")
        dot_y = {o["label"]: o["center"][1] for o in ldots}
        if not rep.check(len(clean_target) == 1, f"{tag}: {len(clean_target)} target lines"):
            continue
        target = clean_target[0]
        t_path = drawn(target)

        for d, v in enumerate(versions):
            r = vers[v]
            vt = f"{tag}/{v}"
            rep.check(int(r["difficulty"]) == d, f"{vt}: difficulty {r['difficulty']} != {d}")
            rep.check(r["image_id"] == f"{cfg['id_prefix']}_{sid:04d}_{v}", f"{vt}: image_id {r['image_id']}")
            rep.check(r["file_name"] == f"images/{r['image_id']}.png", f"{vt}: file_name {r['file_name']}")
            rep.check(r["question"] == info["question_template"].format(dot=r["target_dot"]), f"{vt}: question text differs")
            for key in ("split", "answer", "target_dot", "target_box"):
                rep.check(r[key] == clean[key], f"{vt}: {key} differs from clean version")
            tb = int(r["target_box"])
            rep.check(r["answer"] == f"box: {tb}", f"{vt}: answer '{r['answer']}' != 'box: {tb}'")
            rep.check(r["target_dot"] in dots and 1 <= tb <= nbox, f"{vt}: bad target dot/box")
            # the target line and the layout are identical in every version
            rep.check(json.dumps([o for o in r["objects"] if o["role"] == "layout"], sort_keys=True) ==
                      json.dumps(clean_layout, sort_keys=True), f"{vt}: layout differs from clean version")
            tl = [o for o in r["objects"] if o["role"] == "target"]
            rep.check(json.dumps(tl, sort_keys=True) == json.dumps(clean_target, sort_keys=True),
                      f"{vt}: target line differs from clean version")
            rep.check(target["shape"] == "line" and target["from_dot"] == r["target_dot"] and target["to_box"] == tb
                      and target["color"] == cfg["line_color"], f"{vt}: target line does not match the answer")

            others = [o for o in r["objects"] if o["role"] == "distractor"]
            rep.check(int(r["num_distractors"]) == len(others) == cfg["distractors_per_version"][v],
                      f"{vt}: {len(others)} other lines, expected {cfg['distractors_per_version'][v]}")
            rep.check(sorted(o["from_dot"] for o in others) == ([] if not others else [x for x in dots if x != r["target_dot"]]),
                      f"{vt}: other lines must start at the other dots, once each")
            boxes = [o["to_box"] for o in others]
            rep.check(len(set(boxes)) == len(boxes) and tb not in boxes and all(1 <= b <= nbox for b in boxes),
                      f"{vt}: other lines end in bad boxes {boxes}")
            rep.check(r["distractor_boxes"] == ",".join(map(str, sorted(boxes))), f"{vt}: distractor_boxes column")
            rep.check(r["crossed_boxes"] == ",".join(str(b) for b in sorted(o["to_box"] for o in others if o["num_crossings"])),
                      f"{vt}: crossed_boxes column")
            if v in recolored:
                base = [o for o in vers[recolored[v]]["objects"] if o["role"] == "distractor"]
                rep.check([(o["points"], o["from_dot"], o["to_box"]) for o in others] ==
                          [(o["points"], o["from_dot"], o["to_box"]) for o in base], f"{vt}: lines differ from {recolored[v]}")
                cols = [o["color"] for o in others]
                rep.check(len(set(cols)) == len(cols) and all(c in cfg["distractor_colors"] for c in cols),
                          f"{vt}: bad colours {cols}")
            else:
                rep.check(all(o["color"] == cfg["line_color"] for o in others), f"{vt}: other lines must be {cfg['line_color']}")

            # geometry of every line, recomputed from its control points
            paths = []                                  # drawn paths of the other lines
            for o in ([target] if d == 0 else []) + others:
                pts = np.array(o["points"], float)
                p = drawn(o)
                if o["role"] == "distractor":
                    paths.append(p)
                ok = (pts[:, 0].tolist() == cols_x and pts[0].tolist() == [cfg["dot_x"], dot_y[o["from_dot"]]]
                      and pts[-1].tolist() == [cfg["box_x"][0], cfg["box_y"][o["to_box"] - 1]]
                      and pts[1, 1] == pts[0, 1] and pts[-2, 1] == pts[-1, 1])
                rep.check(ok, f"{vt}: line from {o['from_dot']} has wrong columns or end points")
                dx = np.diff(p[:, 0])
                rep.check((dx > 0).all(), f"{vt}: line from {o['from_dot']} doubles back")
                rep.check(np.abs(np.diff(p[:, 1]) / dx).max() <= cfg["max_slope"] + 1e-6, f"{vt}: line from {o['from_dot']} too steep")
                rep.check(cfg["y_range"][0] - 1e-6 <= p[:, 1].min() and p[:, 1].max() <= cfg["y_range"][1] + 1e-6,
                          f"{vt}: line from {o['from_dot']} leaves the drawing area")
                q = _resample(p)
                h = np.degrees(np.arctan2(np.diff(q[:, 1]), np.diff(q[:, 0])))
                rep.check(np.abs(h[8:] - h[:-8]).max() <= cfg["max_turn_deg"] + 1e-6, f"{vt}: line from {o['from_dot']} has a kink")
            all_pts = []
            for o, p in zip(others, paths):
                c = _line_crossings(t_path, p)
                if not rep.check(c is not None, f"{vt}: line from {o['from_dot']} touches the target"):
                    continue
                rep.check(len(c) == o["num_crossings"] == len(o["crossings"]), f"{vt}: line from {o['from_dot']} crosses "
                          f"{len(c)} times, metadata says {o['num_crossings']}")
                for (pt, ang), m in zip(c, o["crossings"]):
                    rep.check(ang >= cfg["min_crossing_angle_deg"] - 1e-6, f"{vt}: crossing at {m['at']} only {ang:.1f} deg")
                    rep.check(math.dist(pt, m["at"]) <= 1.0 and abs(ang - m["angle_deg"]) <= 0.2,
                              f"{vt}: crossing position/angle differs from metadata")
                    stats["angle"].append(ang)
                all_pts += [pt for pt, _ in c]
                rep.check(_min_distance(_resample(t_path), _resample(p), [pt for pt, _ in c], cfg["crossing_exclusion_px"],
                                        cfg["min_line_separation_px"]) >= cfg["min_line_separation_px"] - 0.05,
                          f"{vt}: line from {o['from_dot']} runs too close to the target")
            rep.check(len(all_pts) == int(r["num_crossings"]) == cfg["crossings_per_version"][v],
                      f"{vt}: {len(all_pts)} crossings, num_crossings={r['num_crossings']}, expected {cfg['crossings_per_version'][v]}")
            for a, b in combinations(all_pts, 2):
                rep.check(math.dist(a, b) >= cfg["min_crossing_distance_px"] - 1e-6, f"{vt}: two crossings {math.dist(a, b):.1f}px apart")
            for (oa, pa), (ob, pb) in combinations(list(zip(others, paths)), 2):
                c = _line_crossings(pa, pb)
                rep.check(c == [], f"{vt}: lines from {oa['from_dot']} and {ob['from_dot']} cross or touch")
                rep.check(_min_distance(_resample(pa), _resample(pb), [], 0, cfg["min_line_separation_px"])
                          >= cfg["min_line_separation_px"] - 0.05, f"{vt}: lines from {oa['from_dot']} and {ob['from_dot']} too close")
            if others:
                left = dots.index(r["target_dot"])                       # lines above the target at the dots
                right = sum(b < tb for b in boxes)                       # ... and at the boxes
                stats[f"rank_kept/{v}"].append(left == right)
            rep.check(r["rank_kept"] == ("" if not others else ("yes" if left == right else "no")),
                      f"{vt}: rank_kept column is wrong")

    split_of_scene = {sid: vers["clean"]["split"] for sid, vers in scenes.items() if "clean" in vers}
    rep.check(set(split_of_scene.values()) <= {"train", "validation", "test"}, "unknown split name")

    # ---------------- pixel checks ----------------
    if not args.skip_pixels:
        for n, (sid, vers) in enumerate(sorted(scenes.items()), start=1):
            tag = f"scene {sid}"
            arrays = {}
            for v, r in vers.items():
                path = root / r["file_name"]
                if not rep.check(path.exists(), f"{tag}/{v}: missing file {path}"):
                    continue
                with Image.open(path) as im:
                    rep.check(im.size == (size, size) and im.mode == "RGB", f"{tag}/{v}: size {im.size} mode {im.mode}")
                    arrays[v] = np.asarray(im.convert("RGB"))
            if len(arrays) != len(versions):
                continue
            clean = vers["clean"]
            layout = _layout_mask([o for o in clean["objects"] if o["role"] == "layout"], INK_MARGIN, size)
            target = next(o for o in clean["objects"] if o["role"] == "target")
            tmask = _stroke_mask(drawn(target), extra, size)
            clean_arr = arrays["clean"]
            ink = _channel_min(clean_arr) < 255
            rep.check(not (ink & ~(tmask | layout)).any(), f"{tag}/clean: ink outside the target line, dots and boxes")
            dark = _channel_min(clean_arr) < 200
            rep.check((dark & tmask & ~layout).sum() >= MIN_VISIBLE_PX, f"{tag}/clean: target line not visible")
            masks = {}
            free = ~layout & ~tmask
            for v in versions[1:]:
                arr = arrays[v]
                others = [o for o in vers[v]["objects"] if o["role"] == "distractor"]
                dmask = np.zeros((size, size), bool)
                dark_v = _channel_min(arr) < 200
                for o in others:
                    key = json.dumps(o["points"])
                    if key not in masks:
                        masks[key] = _stroke_mask(drawn(o), extra, size)
                    m = masks[key]
                    dmask |= m
                    if o["color"] == cfg["line_color"]:
                        vis = int((dark_v & m & free).sum())
                    else:                                # the line shows its own colour
                        ys, xs = np.nonzero(m & free)
                        vis = int((np.abs(arr[ys, xs].astype(int) - palette[o["color"]]).sum(axis=1) < 60).sum())
                    rep.check(vis >= MIN_VISIBLE_PX, f"{tag}/{v}: line from {o['from_dot']} not visible")
                changed = (arr[..., 0] != clean_arr[..., 0]) | (arr[..., 1] != clean_arr[..., 1]) | (arr[..., 2] != clean_arr[..., 2])
                rep.check(not (changed & ~dmask).any(), f"{tag}/{v}: pixels differ from clean image outside the other lines")
            if n % 100 == 0:
                print(f"  pixel-checked {n}/{len(scenes)} scenes")

    # ---------------- summary ----------------
    print(f"\nDataset: {root.resolve()}")
    print(f"  {len(scenes)} scenes, {len(rows)} images")
    split_counts = Counter(split_of_scene.values())
    print("  scenes per split: " + ", ".join(f"{s}={split_counts[s]}" for s in ("train", "validation", "test")))
    print("  images per version: " + ", ".join(f"{v}={sum(r['version'] == v for r in rows)}" for v in versions))
    cleans = [vers["clean"] for vers in scenes.values() if "clean" in vers]
    print("  answers (target box): " + str(dict(sorted(Counter(int(r["target_box"]) for r in cleans).items()))))
    print("  target dot: " + str(dict(sorted(Counter(r["target_dot"] for r in cleans).items()))))
    for v in versions[1:]:
        cr = Counter(int(vers[v]["num_crossings"]) for vers in scenes.values() if v in vers)
        kept = stats[f"rank_kept/{v}"]
        print(f"  {v}: crossings {dict(cr)}, target keeps its rank among line ends in "
              f"{100 * sum(kept) / max(1, len(kept)):.0f}% of images")
    if stats["angle"]:
        a = np.array(stats["angle"])
        print(f"  crossing angles: min {a.min():.1f}, median {np.median(a):.1f} deg")

    if args.preview_scenes > 0:
        out = save_preview(root, scenes, versions, size, args.preview_scenes)
        print(f"  preview saved: {out}")

    if rep.errors:
        print(f"\nFAILED: {len(rep.errors)} problem(s). First ones:")
        for e in rep.errors[:30]:
            print("  - " + e)
        sys.exit(1)
    print("\nALL CHECKS PASSED" + (" (metadata only)" if args.skip_pixels else ""))


def _font(px):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:          # Pillow < 10.1
        return ImageFont.load_default()


def save_preview(root, scenes, versions, size, k):
    """Grid: one row per scene, one column per version, answer under each row."""
    ids = sorted(scenes)
    picks = [ids[round(i * (len(ids) - 1) / max(1, k - 1))] for i in range(min(k, len(ids)))]
    tile, pad, head, foot = 300, 10, 26, 30
    W = pad + len(versions) * (tile + pad)
    H = pad + len(picks) * (head + tile + foot + pad)
    sheet = Image.new("RGB", (W, H), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    f_head, f_foot = _font(18), _font(16)
    for row, sid in enumerate(picks):
        y = pad + row * (head + tile + foot + pad)
        for col, v in enumerate(versions):
            x = pad + col * (tile + pad)
            r = scenes[sid][v]
            with Image.open(root / r["file_name"]) as im:
                sheet.paste(im.convert("RGB").resize((tile, tile), Image.LANCZOS), (x, y + head))
            draw.text((x, y + 4), f"scene {sid}  |  {v}", fill=(0, 0, 0), font=f_head)
        r = scenes[sid]["clean"]
        target = f"count the {r['target_color'].upper()} shapes  ->  " if "target_color" in r else ""
        if "target_dot" in r:
            target = f"follow the line from {r['target_dot']}  ->  "
        draw.text((pad, y + head + tile + 6), f"{target}answer (all versions): {r['answer']}   [{r['split']}]",
                  fill=(0, 0, 0), font=f_foot)
    out = root / "preview.png"
    sheet.save(out)
    return out


if __name__ == "__main__":
    main()
