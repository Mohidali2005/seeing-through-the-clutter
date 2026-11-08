import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from tangle import data
from tangle.analysis.common import load_notes
from tangle.analysis.common import load_swap
from tangle.paths import ANALYSIS

parser = argparse.ArgumentParser(description="do pictures with more leak get more wrong answers")
parser.add_argument("results",type=Path,help="the results folder of a downloaded run")
parser.add_argument("--model",default="qwen4b")
parser.add_argument("--dataset",default="objects")
args = parser.parse_args()

notes = load_notes(args.results/"notes"/args.model/args.dataset)
swap = load_swap(args.results/"swap"/args.model/args.dataset)
plain = swap[swap["condition"]=="cluttered"][["image_id","exact_match","n_clutter"]]
counts = {row["image_id"]:int(row["count_total"]) for scene in data.load_scenes(args.dataset) for row in scene["rows"]}

# the leak of each picture is its mean over object squares at the hand over
leak = (notes["leak_llm"][:,0]*notes["object"]).sum(1)/notes["object"].sum(1).clip(1)
frame = pd.DataFrame({"image_id":notes["image_id"],"scene_id":notes["scene_id"],"version":notes["version"],"leak":leak})
frame = frame.merge(plain,on="image_id")
frame["right"] = frame["exact_match"].astype(int)
frame["targets"] = frame["image_id"].map(counts)
frame["leak_z"] = (frame["leak"]-frame["leak"].mean())/frame["leak"].std()
frame["clutter_z"] = (frame["n_clutter"]-frame["n_clutter"].mean())/frame["n_clutter"].std()

fit = smf.logit("right ~ leak_z + C(version) + clutter_z + targets",data=frame).fit(disp=0,cov_type="cluster",cov_kwds={"groups":frame["scene_id"]})
coef = fit.params["leak_z"]
low,high = fit.conf_int().loc["leak_z"]
lines = [f"# Link test: {args.model} on {args.dataset}","",
         f"Logistic regression of a right answer on the leak at the hand over (per standard deviation), with clutter version, clutter squares and number of targets held fixed; {len(frame)} pictures, ranges clustered by scene.","",
         f"- leak coefficient (log odds per SD): {coef:.3f} [{low:.3f}, {high:.3f}], odds ratio {np.exp(coef):.2f}",
         f"- negative means more leak goes with more wrong answers",""]
lines += ["| version | mean leak x 1000 when right | when wrong |","|---|---|---|"]
for v,group in frame.groupby("version",sort=False):
    lines.append(f"| {v} | {1000*group.loc[group['right']==1,'leak'].mean():.2f} | {1000*group.loc[group['right']==0,'leak'].mean():.2f} |")
out = ANALYSIS/"stats"
out.mkdir(parents=True,exist_ok=True)
(out/f"link_{args.model}_{args.dataset}.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
print(f"link {args.model} {args.dataset}: leak coefficient {coef:.3f} [{low:.3f}, {high:.3f}]")
