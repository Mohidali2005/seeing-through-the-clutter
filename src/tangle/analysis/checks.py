import argparse
import json
from pathlib import Path

from tangle.analysis.common import load_notes
from tangle.analysis.common import load_swap
from tangle.paths import ANALYSIS

# exact match accuracy of study 3 with qwen3 vl 4b in percent so our plain rows can be compared with it
STUDY3 = {
    "objects":{"clean":97.2,"random_objects":90.0,"text_traps":57.0,"lookalikes":31.2,"everything":12.6},
    "shapes":{"clean":45.0,"squiggles":28.8,"lookalikes":3.4,"blobs":2.8},
    "hard":{"clean":42.2,"squiggles_2":36.4,"squiggles_4":31.6,"squiggles_16":30.4,"text_traps":28.0,"squiggles_8":26.0,"other_shapes":17.4,"wrong_color":8.4,"lookalikes":3.2,"everything":0.4},
}

parser = argparse.ArgumentParser(description="health report of one downloaded kaggle run")
parser.add_argument("results",type=Path,help="the results folder of a downloaded run")
parser.add_argument("--name",default=None,help="name of the report file")
args = parser.parse_args()

lines = [f"# Run checks: {args.results}",""]
environment = json.loads((args.results/"environment.json").read_text(encoding="utf-8"))
manifest = json.loads((args.results/"manifest.json").read_text(encoding="utf-8"))
lines += [f"- mode {environment['mode']}, commit {environment['commit'][:10]}, gpus {environment['gpus']}",f"- packages: {environment['packages']}",""]
lines += ["| step | exit codes | minutes | chunks |","|---|---|---|---|"]
for step in manifest["steps"]:
    lines.append(f"| {step['step']} | {step.get('exit_codes',step.get('skipped'))} | {step.get('minutes','')} | {step.get('chunks','')} |")
lines.append("")

for folder in sorted((args.results/"notes").glob("*/*")):
    notes = load_notes(folder)
    lines += [f"## Notes {folder.parent.name} {folder.name}: {len(notes['image_id'])} cluttered images",
              f"- untouched patches whose patch embedding changed (must be 0): {int(notes['embed_diff'].sum())}",
              f"- patches where png and processor disagree (must be 0): {int(notes['pixel_mismatch'].sum())}",
              f"- largest difference between two runs of the same clean picture (0 means exact): {float(notes['dup_diff'].max())}",
              f"- non finite values (must be 0): {int(notes['nonfinite'].sum())}",
              f"- mean leak on object squares at the hand over x 1000: {1000*float((notes['leak_llm'][:,0]*notes['object']).sum()/notes['object'].sum()):.3f}",""]

for folder in sorted((args.results/"swap").glob("*/*")):
    table = load_swap(folder)
    answers = table.pivot_table(index="image_id",columns="condition",values="answer",aggfunc="first")
    plain = table[table["condition"].isin(["clean","cluttered"])]
    lines += [f"## Swap {folder.parent.name} {folder.name}: {table['image_id'].nunique()} cluttered images",
              f"- largest logit gap A vs plain cluttered: {table.loc[table['condition']=='A','check_diff'].max()}",
              f"- largest logit gap D vs plain clean: {table.loc[table['condition']=='D','check_diff'].max()}",
              f"- A answers equal plain cluttered answers: {(answers['A']==answers['cluttered']).mean():.2%}",
              f"- D answers equal plain clean answers: {(answers['D']==answers['clean']).mean():.2%}",
              f"- parsed answers: {table.loc[table['answer']!='','parsed'].mean():.2%}",
              f"- greedy path equals gold agrees with the scorer (plain rows): {(plain['greedy_gold']==plain['exact_match']).mean():.2%}",
              "","| version | images | our accuracy % | study 3 % |","|---|---|---|---|"]
    clean = table[table["condition"]=="clean"].drop_duplicates("scene_id")
    study = STUDY3[folder.name]
    lines.append(f"| clean | {len(clean)} | {100*clean['exact_match'].mean():.1f} | {study['clean']} |")
    cluttered = table[table["condition"]=="cluttered"]
    for version,group in cluttered.groupby("version",sort=False):
        lines.append(f"| {version} | {len(group)} | {100*group['exact_match'].mean():.1f} | {study.get(version,'')} |")
    lines.append("")

out = ANALYSIS/"checks"
out.mkdir(parents=True,exist_ok=True)
name = args.name or args.results.parent.name
(out/f"{name}.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
print("\n".join(lines))
