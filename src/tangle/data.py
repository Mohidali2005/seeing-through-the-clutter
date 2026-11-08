import csv
import json

import numpy as np
from PIL import Image

from tangle import config
from tangle.paths import DATA


def load_scenes(name,limit=None):
    """Read one dataset and group its rows by scene with the clean version first"""
    folder = DATA/config.DATASETS[name]["folder"]
    with open(folder/"metadata.csv",encoding="utf-8",newline="") as f:
        rows = list(csv.DictReader(f))
    scenes = {}
    for row in rows:
        scenes.setdefault(int(row["scene_id"]),[]).append(row)
    out = []
    for scene_id in sorted(scenes):
        versions = sorted(scenes[scene_id],key=lambda r:int(r["difficulty"]))
        assert versions[0]["version"]=="clean",f"scene {scene_id} has no clean version first"
        out.append({"scene_id":scene_id,"split":versions[0]["split"],"rows":versions})
    return out[:limit] if limit else out


def load_image(name,row):
    """Load one picture as an rgb array of bytes"""
    return np.asarray(Image.open(DATA/config.DATASETS[name]["folder"]/row["file_name"]).convert("RGB"))


def patch_cells():
    """Give the row and column on the 16 pixel grid of every patch in the order the processor lists them"""
    # the processor lists patches token by token and each token holds a two by two block of patches
    p = np.arange(config.PATCHES)
    token = p//4
    sub = p%4
    rows = 2*(token//config.GRID)+sub//2
    cols = 2*(token%config.GRID)+sub%2
    return np.stack([rows,cols],1)


def changed_cells(clean,image,size):
    """Mark every cell of the given size where at least one pixel differs between two pictures"""
    n = config.IMAGE_SIZE//size
    return (clean!=image).any(2).reshape(n,size,n,size).any((1,3))


def bbox_cells(bbox,size):
    """Mark every cell of the given size that overlaps a bounding box"""
    n = config.IMAGE_SIZE//size
    x0,y0,x1,y1 = bbox
    start = np.arange(n)*size
    rows = (start<=y1)&(start+size-1>=y0)
    cols = (start<=x1)&(start+size-1>=x0)
    return rows[:,None]&cols[None,:]


def masks(clean,image,objects):
    """Split the squares and patches of a picture into object background and clutter parts"""
    cells = patch_cells()
    targets = [o for o in objects if o["role"]=="target"]
    changed_sq = changed_cells(clean,image,config.SQUARE).reshape(-1)
    changed_patch = changed_cells(clean,image,config.PATCH)[cells[:,0],cells[:,1]]
    target_sq = np.zeros(config.TOKENS,bool)
    target_patch = np.zeros(config.PATCHES,bool)
    per_target = []
    for o in targets:
        sq = bbox_cells(o["bbox"],config.SQUARE).reshape(-1)
        patch = bbox_cells(o["bbox"],config.PATCH)[cells[:,0],cells[:,1]]
        target_sq |= sq
        target_patch |= patch
        per_target.append({"id":o["id"],"shape":o["shape"],"squares":np.flatnonzero(sq&~changed_sq),"patches":np.flatnonzero(patch&~changed_patch)})
    return {
        "untouched":~changed_sq,
        "object":target_sq&~changed_sq,
        "background":~target_sq&~changed_sq,
        "clutter":changed_sq,
        "untouched_patch":~changed_patch,
        "object_patch":target_patch&~changed_patch,
        "background_patch":~target_patch&~changed_patch,
        "targets":per_target,
    }


def objects_of(row):
    """Parse the objects column of a metadata row"""
    return json.loads(row["objects"])
