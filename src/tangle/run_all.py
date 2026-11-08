import argparse
import math
import os
import shutil
import subprocess
import sys
import time

from tangle import config
from tangle import data
from tangle import records
from tangle.paths import MODELS
from tangle.paths import NOTES
from tangle.paths import RESULTS
from tangle.paths import SWAPS

parser = argparse.ArgumentParser(description="run every gpu step on kaggle in priority order")
parser.add_argument("--models",nargs="+",default=["qwen4b"],choices=list(config.MODELS))
parser.add_argument("--steps",nargs="*",default=None,help="only these steps written as stage:dataset")
args = parser.parse_args()

start = time.time()
env = dict(os.environ,TANGLE_START=str(start))
manifest = {"started":time.strftime("%Y-%m-%d %H:%M:%S"),"models":args.models,"steps":[]}


def run(name,commands):
    """Run commands side by side and record their exit codes time and free disk in the manifest"""
    began = time.time()
    processes = [subprocess.Popen(command,env=command_env) for command,command_env in commands]
    codes = [p.wait() for p in processes]
    step = {"step":name,"exit_codes":codes,"minutes":round((time.time()-began)/60,2),"free_disk_gb":round(shutil.disk_usage("/tmp").free/1e9)}
    manifest["steps"].append(step)
    records.write_json(RESULTS/"manifest.json",manifest)
    print(f"step {step}")
    return step


run("fetch data",[([sys.executable,"-m","tangle.fetch","data"],env)])
for model in args.models:
    run(f"fetch {model}",[([sys.executable,"-m","tangle.fetch",model],env)])
    copies = config.MODELS[model]["copies"]
    for stage,dataset in config.RUNS[model]:
        name = f"{stage}:{dataset}"
        if args.steps is not None and name not in args.steps:
            continue
        if (time.time()-start)/3600>config.GEN_DEADLINE_H:
            manifest["steps"].append({"step":f"{model} {name}","skipped":"deadline"})
            records.write_json(RESULTS/"manifest.json",manifest)
            continue
        commands = []
        for w in range(copies):
            # each copy of a small model gets its own gpu
            worker_env = dict(env,CUDA_VISIBLE_DEVICES=str(w)) if copies>1 else env
            commands.append(([sys.executable,"-m",f"tangle.{stage}",dataset,"--model",model,"--worker",str(w),"--workers",str(copies)],worker_env))
        step = run(f"{model} {name}",commands)
        # judging success by the chunk files because a process can exit badly after saving its work
        limit = int(os.environ.get("TANGLE_SCENES","0")) or None
        size = int(os.environ.get("TANGLE_CHUNK",str(config.CHUNK)))
        expected = math.ceil(len(data.load_scenes(dataset,limit))/size)
        folder = (NOTES if stage=="notes" else SWAPS)/model/dataset
        have = len(list(folder.glob("chunk_*.npz" if stage=="notes" else "chunk_*.csv")))
        step["chunks"] = f"{have}/{expected}"
        if have==expected:
            records.write_json(folder/"done.json",{"model":model,"revision":config.MODELS[model]["revision"],"dataset":dataset,"chunks":expected,"minutes":step["minutes"]})
        records.write_json(RESULTS/"manifest.json",manifest)
    shutil.rmtree(MODELS/model,ignore_errors=True)

manifest["hours"] = round((time.time()-start)/3600,2)
manifest["complete"] = [s["step"] for s in manifest["steps"] if "chunks" in s and s["chunks"].split("/")[0]==s["chunks"].split("/")[1]]
records.write_json(RESULTS/"manifest.json",manifest)
print(f"run finished in {manifest['hours']} h with {len(manifest['complete'])} complete steps")
