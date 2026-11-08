import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tangle import config
from tangle.analysis.common import boot_ratio
from tangle.analysis.common import fmt
from tangle.analysis.common import load_notes
from tangle.analysis.common import scene_resamples
from tangle.paths import ANALYSIS

# clutter versions drawn at the same spots with the same sizes and colours so only their look differs
SAME_SPOTS = {
    "shapes":["squiggles","blobs","lookalikes"],
    "hard":["squiggles_2","squiggles_4","squiggles_8","squiggles_16","other_shapes"],
}
COLORS = ["steelblue","darkorange","seagreen","firebrick","mediumpurple","goldenrod","teal","sienna","slategray","crimson"]

parser = argparse.ArgumentParser(description="leak of clutter into the notes of untouched squares")
parser.add_argument("results",type=Path,help="the results folder of a downloaded run")
parser.add_argument("--model",default="qwen4b")
parser.add_argument("--dataset",default="objects")
args = parser.parse_args()

notes = load_notes(args.results/"notes"/args.model/args.dataset)
scenes = np.sort(np.unique(notes["scene_id"]))
draws = scene_resamples(scenes)
index = np.searchsorted(scenes,notes["scene_id"])
versions = list(dict.fromkeys(notes["version"]))
n_vision = notes["leak_vision"].shape[1]
n_llm = notes["leak_llm"].shape[1]


def masked_mean(values,mask):
    """Average each image over the masked squares at every layer"""
    return (values*mask[:,None,:]).sum(-1)/mask.sum(-1).clip(1)[:,None]


# one curve point per layer running through the vision part then the language part where llm point 0 is the hand over
obj_curve = np.concatenate([masked_mean(notes["leak_vision"],notes["object_patch"]),masked_mean(notes["leak_llm"],notes["object"])],1)
bg_curve = np.concatenate([masked_mean(notes["leak_vision"],notes["background_patch"]),masked_mean(notes["leak_llm"],notes["background"])],1)
names = ["patch embedding"]+[f"vision {k}" for k in range(1,n_vision)]+["hand over"]+[f"llm {k}" for k in range(1,n_llm)]
handover = n_vision

lines = [f"# Leak test: {args.model} on {args.dataset}","",f"{len(notes['image_id'])} cluttered images from {len(scenes)} scenes. Leak = 1 - cosine between clean and cluttered notes of untouched squares.","","## Checks",""]
checks = {
    "untouched patches whose patch embedding changed (must be 0)":int(notes["embed_diff"].sum()),
    "patches where pixel and processor untouched disagree (must be 0)":int(notes["pixel_mismatch"].sum()),
    "largest difference between the two clean copies (0 means the gpu is exact)":float(notes["dup_diff"].max()),
    "non finite values in the notes (must be 0)":int(notes["nonfinite"].sum()),
    "largest leak at the patch embedding on untouched patches (must be 0)":float((notes["leak_vision"][:,0]*notes["untouched_patch"]).max()),
}
for name,value in checks.items():
    lines.append(f"- {name}: {value}")

# the middle of the vision part then the hand over then llm 8 where the sparse autoencoder may go then the middle and the end
shown = [f"vision {(n_vision-1)//2}","hand over",f"llm {min(8,n_llm-1)}",f"llm {(n_llm-1)//2}",f"llm {n_llm-1}"]
lines += ["","## Leak on object squares by version (mean x 1000)","","| version | "+" | ".join(shown)+" |","|---|---|---|---|---|---|"]
picks = [names.index(n) for n in shown]
rows = []
for v in versions:
    sel = notes["version"]==v
    n = np.bincount(index[sel],minlength=len(scenes)).astype(float)
    cells = []
    for p in picks:
        estimate = boot_ratio(np.bincount(index[sel],weights=obj_curve[sel,p],minlength=len(scenes)),n,draws)
        cells.append(fmt(*estimate,scale=1000,digits=2))
    lines.append(f"| {v} | "+" | ".join(cells)+" |")
    for p,name in enumerate(names):
        rows.append({"model":args.model,"dataset":args.dataset,"version":v,"layer":name,"object_leak":obj_curve[sel,p].mean(),"background_leak":bg_curve[sel,p].mean()})

if args.dataset in SAME_SPOTS:
    same = SAME_SPOTS[args.dataset]
    lines += ["","## Same spots, different look (object leak at the hand over, paired by scene, x 1000)","","| comparison | difference |","|---|---|"]
    by_version = {v:pd.Series(obj_curve[notes["version"]==v,handover],index=notes["scene_id"][notes["version"]==v]) for v in same}
    for first,second in zip(same[:-1],same[1:]):
        diff = (by_version[second]-by_version[first]).reindex(scenes).fillna(0).to_numpy()
        lines.append(f"| {second} minus {first} | {fmt(*boot_ratio(diff,np.ones(len(scenes)),draws),scale=1000,digits=2)} |")

# distance in squares from each untouched square to the nearest clutter square
grid = np.stack(np.meshgrid(np.arange(config.GRID),np.arange(config.GRID),indexing="ij"),-1).reshape(-1,2)
far = np.abs(grid[:,None,:]-grid[None,:,:]).max(-1)
distance = np.stack([far[:,c].min(1) if c.any() else np.full(config.TOKENS,99) for c in notes["clutter"]])
lines += ["","## Leak at the hand over against distance to the nearest clutter square (mean x 1000)","","| distance in squares | untouched squares | leak |","|---|---|---|"]
leak_hand = notes["leak_llm"][:,0]
for low,high in [(1,1),(2,2),(3,3),(4,5),(6,99)]:
    mask = notes["untouched"]&(distance>=low)&(distance<=high)
    lines.append(f"| {low} to {high} | {int(mask.sum())} | {1000*leak_hand[mask].mean():.2f} |")

out = ANALYSIS/"stats"
out.mkdir(parents=True,exist_ok=True)
(out/f"leak_{args.model}_{args.dataset}.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
pd.DataFrame(rows).to_csv(out/f"leak_{args.model}_{args.dataset}.csv",index=False,lineterminator="\n")

figures = ANALYSIS/"figures"
figures.mkdir(parents=True,exist_ok=True)
fig,ax = plt.subplots(figsize=(10,4))
for v,color in zip(versions,COLORS):
    sel = notes["version"]==v
    ax.plot(range(len(names)),obj_curve[sel].mean(0),color=color,label=v)
ax.axvline(handover,color="gray",linestyle="--",linewidth=1)
ax.set_xticks([0,handover//2,handover,handover+8,len(names)-1])
ax.set_xticklabels(["patch embedding",f"vision {handover//2}","hand over","llm 8",f"llm {n_llm-1}"])
ax.set_ylabel("leak on object squares (1 - cosine)")
ax.set_title(f"How much clutter leaks into untouched object notes, {args.model}, {args.dataset}")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(figures/f"leak_{args.model}_{args.dataset}.png",dpi=150)
print(f"leak {args.model} {args.dataset}: "+"; ".join(f"{k} {v}" for k,v in checks.items()))
