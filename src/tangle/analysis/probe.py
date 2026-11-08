import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tangle.analysis.common import boot_ratio
from tangle.analysis.common import fmt
from tangle.analysis.common import load_notes
from tangle.analysis.common import scene_resamples
from tangle.paths import ANALYSIS

parser = argparse.ArgumentParser(description="read back probes on the notes of untouched objects")
parser.add_argument("results",type=Path,help="the results folder of a downloaded run")
parser.add_argument("--model",default="qwen4b")
parser.add_argument("--dataset",default="objects")
args = parser.parse_args()

notes = load_notes(args.results/"notes"/args.model/args.dataset)
image_scene = notes["scene_id"]
image_version = notes["version"]
scenes = np.sort(np.unique(image_scene))
draws = scene_resamples(scenes)
versions = list(dict.fromkeys(image_version))
layers = [("vision",k,i) for i,k in enumerate(notes["vision_layers"])]+[("llm",k,i) for i,k in enumerate(notes["llm_layers"])]


def label(kind,k):
    """Name a layer for tables and plots"""
    if kind=="vision":
        return "patch embedding" if k==0 else f"vision {k}"
    return "hand over" if k==0 else f"llm {k}"


def cross_validated(x,y,groups):
    """Predict every sample with a probe trained on the other scenes and return which predictions are right"""
    right = np.zeros(len(y),bool)
    for train,test in GroupKFold(n_splits=5).split(x,y,groups):
        probe = make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=3000))
        probe.fit(x[train],y[train])
        right[test] = probe.predict(x[test])==y[test]
    return right


def presence(clean,cluttered,scene,version):
    """Probe clean against cluttered notes of the same things one version at a time and pool the right answers"""
    right_all = []
    scene_all = []
    per_version = {}
    for v in versions:
        pick = version==v
        x = np.concatenate([clean[pick],cluttered[pick]]).astype(np.float32)
        y = np.r_[np.zeros(pick.sum()),np.ones(pick.sum())]
        groups = np.r_[scene[pick],scene[pick]]
        right = cross_validated(x,y,groups)
        per_version[v] = right.mean()
        right_all.append(right)
        scene_all.append(groups)
    right = np.concatenate(right_all)
    index = np.searchsorted(scenes,np.concatenate(scene_all))
    counts = np.bincount(index,minlength=len(scenes)).astype(float)
    hits = np.bincount(index,weights=right.astype(float),minlength=len(scenes))
    return boot_ratio(hits,counts,draws),per_version


obj_scene = image_scene[notes["obj_image"]]
obj_version = image_version[notes["obj_image"]]
rows = []
for kind,k,i in layers:
    for part in ["object","background"]:
        if part=="object":
            clean = notes[f"obj_{kind}_clean"][:,i]
            cluttered = notes[f"obj_{kind}_clut"][:,i]
            scene,version = obj_scene,obj_version
        else:
            clean = notes[f"bg_{kind}_clean"][:,i]
            cluttered = notes[f"bg_{kind}_clut"][:,i]
            scene,version = image_scene,image_version
        (estimate,low,high),per_version = presence(clean,cluttered,scene,version)
        rows.append({"model":args.model,"dataset":args.dataset,"layer":label(kind,k),"part":part,"accuracy":estimate,"low":low,"high":high,**{f"acc_{v}":a for v,a in per_version.items()}})
    print(f"probe {args.model} {args.dataset} {label(kind,k)}: object {rows[-2]['accuracy']:.3f} background {rows[-1]['accuracy']:.3f}")

# the type probe reads the change itself so it cannot use which squares were averaged
type_rows = []
for kind,k,i in layers:
    change = (notes[f"obj_{kind}_clut"][:,i].astype(np.float32)-notes[f"obj_{kind}_clean"][:,i].astype(np.float32))
    y = np.array([versions.index(v) for v in obj_version])
    right = cross_validated(change,y,obj_scene)
    index = np.searchsorted(scenes,obj_scene)
    estimate,low,high = boot_ratio(np.bincount(index,weights=right.astype(float),minlength=len(scenes)),np.bincount(index,minlength=len(scenes)).astype(float),draws)
    type_rows.append({"model":args.model,"dataset":args.dataset,"layer":label(kind,k),"accuracy":estimate,"low":low,"high":high,"chance":1/len(versions)})

table = pd.DataFrame(rows)
types = pd.DataFrame(type_rows)
out = ANALYSIS/"stats"
out.mkdir(parents=True,exist_ok=True)
table.to_csv(out/f"probe_{args.model}_{args.dataset}.csv",index=False,lineterminator="\n")
types.to_csv(out/f"probe_type_{args.model}_{args.dataset}.csv",index=False,lineterminator="\n")

lines = [f"# Read back probes: {args.model} on {args.dataset}","","Probe 1 tells clean from cluttered notes of the same thing (50% is guessing). Five folds grouped by scene, ranges from scene bootstrap.","",
         "| layer | object notes % | background notes % |","|---|---|---|"]
for name in table["layer"].unique():
    o = table[(table["layer"]==name)&(table["part"]=="object")].iloc[0]
    b = table[(table["layer"]==name)&(table["part"]=="background")].iloc[0]
    lines.append(f"| {name} | {fmt(o['accuracy'],o['low'],o['high'])} | {fmt(b['accuracy'],b['low'],b['high'])} |")
lines += ["",f"Probe 2 names the clutter version from the change in object notes (chance {100/len(versions):.1f}%).","","| layer | accuracy % |","|---|---|"]
for _,t in types.iterrows():
    lines.append(f"| {t['layer']} | {fmt(t['accuracy'],t['low'],t['high'])} |")
first = table[(table["layer"]=="patch embedding")&(table["part"]=="object")].iloc[0]["accuracy"]
lines += ["",f"Check: probe 1 at the patch embedding must be exactly 50%: {first:.4%}",""]
if args.model=="qwen4b" and args.dataset=="objects":
    h = table[(table["layer"]=="hand over")&(table["part"]=="object")].iloc[0]
    verdict = "yes" if h["low"]>0.5 else "no"
    lines += ["## Primary test (fixed before any data)","",f"Is it there? Probe 1 on object notes at the hand over, pooled: {fmt(h['accuracy'],h['low'],h['high'])} % -> **{verdict}**",""]
(out/f"probe_{args.model}_{args.dataset}.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")

figures = ANALYSIS/"figures"
figures.mkdir(parents=True,exist_ok=True)
fig,ax = plt.subplots(figsize=(9,4))
names = list(table["layer"].unique())
for part,color in [("object","steelblue"),("background","darkorange")]:
    sub = table[table["part"]==part].set_index("layer").loc[names]
    ax.plot(range(len(names)),sub["accuracy"]*100,marker="o",color=color,label=f"{part} notes")
    ax.fill_between(range(len(names)),sub["low"]*100,sub["high"]*100,color=color,alpha=0.2)
ax.axhline(50,color="gray",linestyle="--",linewidth=1)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names,rotation=45,ha="right")
ax.set_ylabel("probe accuracy (%)")
ax.set_title(f"Can a probe tell clean from cluttered notes? {args.model}, {args.dataset}")
ax.legend()
fig.tight_layout()
fig.savefig(figures/f"probe_{args.model}_{args.dataset}.png",dpi=150)
