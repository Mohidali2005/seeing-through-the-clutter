import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tangle import config
from tangle.analysis.common import boot_ratio
from tangle.analysis.common import fmt
from tangle.analysis.common import load_swap
from tangle.analysis.common import mcnemar
from tangle.analysis.common import scene_resamples
from tangle.paths import ANALYSIS

parser = argparse.ArgumentParser(description="swap test effects with scene bootstrap ranges")
parser.add_argument("results",type=Path,help="the results folder of a downloaded run")
parser.add_argument("--model",default="qwen4b")
parser.add_argument("--dataset",default="objects")
args = parser.parse_args()

table = load_swap(args.results/"swap"/args.model/args.dataset)
key = ["scene_id","image_id","version"]
correct = table.pivot_table(index=key,columns="condition",values="exact_match",aggfunc="first").reset_index()
logprob = table.pivot_table(index=key,columns="condition",values="logprob",aggfunc="first").reset_index()
answers = table.pivot_table(index=key,columns="condition",values="answer",aggfunc="first").reset_index()
scenes = np.sort(correct["scene_id"].unique())
draws = scene_resamples(scenes)
answer_rows = config.ANSWER_ROWS[config.DATASETS[args.dataset]["swap_rows"]]
score_rows = config.SCORE_ROWS[config.DATASETS[args.dataset]["swap_rows"]]
versions = [v for v in correct["version"].unique()]


def per_scene(frame,values):
    """Sum a per image quantity within each scene"""
    index = np.searchsorted(scenes,frame["scene_id"].to_numpy())
    return np.bincount(index,weights=np.asarray(values,dtype=float),minlength=len(scenes))


lines = [f"# Swap test: {args.model} on {args.dataset}","",f"{len(correct)} cluttered images from {len(scenes)} scenes. Ranges: 95%, {draws.shape[0]:,} bootstrap resamples over scenes.",""]
rows = []
for version in ["pooled"]+versions:
    c = correct if version=="pooled" else correct[correct["version"]==version]
    lp = logprob if version=="pooled" else logprob[logprob["version"]==version]
    n = per_scene(c,np.ones(len(c)))
    lines += [f"## {version}","","| run | accuracy % | minus A (points) | McNemar p | log p minus A |","|---|---|---|---|---|"]
    a = c["A"].astype(float).to_numpy()
    for name in ["clean","cluttered"]+score_rows:
        lp_diff = lp[name].to_numpy()-lp["A"].to_numpy()
        lp_text = fmt(*boot_ratio(per_scene(lp,lp_diff),n,draws),scale=1,digits=2)
        row = {"model":args.model,"dataset":args.dataset,"version":version,"run":name,"images":len(c),"logp_minus_A":lp_diff.mean()}
        if name in answer_rows or name in ["clean","cluttered"]:
            x = c[name].astype(float).to_numpy()
            acc = boot_ratio(per_scene(c,x),n,draws)
            diff = boot_ratio(per_scene(c,x-a),n,draws)
            p = mcnemar(int(((x==1)&(a==0)).sum()),int(((x==0)&(a==1)).sum()))
            lines.append(f"| {name} | {fmt(*acc)} | {fmt(*diff)} | {p:.2g} | {lp_text} |")
            row.update({"accuracy":acc[0],"minus_A":diff[0],"minus_A_low":diff[1],"minus_A_high":diff[2],"mcnemar_p":p})
        else:
            lines.append(f"| {name} | | | | {lp_text} |")
        rows.append(row)
    b = c["B"].astype(float).to_numpy()
    d = c["D"].astype(float).to_numpy()
    cc = c["C"].astype(float).to_numpy()
    share = boot_ratio(per_scene(c,b-a),per_scene(c,d-a),draws)
    leak_effect = boot_ratio(per_scene(c,((b-a)+(d-cc))/2),n,draws)
    clutter_effect = boot_ratio(per_scene(c,((cc-a)+(d-b))/2),n,draws)
    lines += ["",f"- share of the clean gap won back by removing the leak, (B - A) / (D - A): {fmt(*share)} %",
              f"- leak effect, average of B - A and D - C: {fmt(*leak_effect)} points",
              f"- clutter effect, average of C - A and D - B: {fmt(*clutter_effect)} points",""]
    if version=="pooled":
        primary = boot_ratio(per_scene(c,b-a),n,draws)

lines += ["## Checks",""]
same_a = (answers["A"]==answers["cluttered"]).mean()
same_d = (answers["D"]==answers["clean"]).mean()
a_diff = table.loc[table["condition"]=="A","check_diff"].max()
d_diff = table.loc[table["condition"]=="D","check_diff"].max()
plain = table[table["condition"]=="cluttered"]
agree = (plain["greedy_gold"]==plain["exact_match"]).mean()
lines += [f"- A answers equal the plain cluttered answers: {same_a:.2%} (largest logit difference {a_diff})",
          f"- D answers equal the plain clean answers: {same_d:.2%} (largest logit difference {d_diff})",
          f"- parsed answers: {table.loc[table['answer']!='','parsed'].mean():.2%}",
          f"- answers that hit the {config.MAX_NEW_TOKENS} token cap: {(table['new_tokens']>=config.MAX_NEW_TOKENS).sum()}",
          f"- greedy path equals the correct answer agrees with the scorer on the plain cluttered rows: {agree:.2%}",""]

if args.model=="qwen4b" and args.dataset=="objects":
    verdict = "yes" if primary[1]>0 else "no"
    lines += ["## Primary test (fixed before any data)","",f"Does the leak matter? B - A pooled over the cluttered versions: {fmt(*primary)} points -> **{verdict}**",""]

out = ANALYSIS/"stats"
out.mkdir(parents=True,exist_ok=True)
(out/f"swap_{args.model}_{args.dataset}.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
pd.DataFrame(rows).to_csv(out/f"swap_{args.model}_{args.dataset}.csv",index=False,lineterminator="\n")
print(f"swap {args.model} {args.dataset}: {len(correct)} images with B - A pooled {fmt(*primary)} points")
