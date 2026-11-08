import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from kaggle_secrets import UserSecretsClient

# probe tries two scenes and smoke runs every step on ten scenes while full4b and full8b run the whole study for one model
MODE = "probe"
COMMIT = "0000000000000000000000000000000000000000"
REPO = "https://github.com/Mohidali2005/seeing-through-the-clutter.git"
INSTALL = ["transformers==5.22.1"]
CODE = Path("/tmp/code")
RESULTS = Path("/kaggle/working/results")
MODELS = Path("/tmp/tangle_models")
DATA = Path("/tmp/tangle_data")
# a results folder of an earlier run attached as a dataset makes this run continue from it
RESUME = Path("/kaggle/input/tangle-partial/results")
MODES = {
    "probe":{"scenes":"2","chunk":"1","args":["--models","qwen4b","--steps","notes:objects","swap:objects"]},
    "smoke":{"scenes":"10","chunk":"5","args":["--models","qwen4b","qwen8b"]},
    "full4b":{"scenes":"0","chunk":"20","args":["--models","qwen4b"]},
    "full8b":{"scenes":"0","chunk":"20","args":["--models","qwen8b"]},
}


def shell(command):
    """Run a shell command and return its output as text"""
    return subprocess.run(command,shell=True,capture_output=True,text=True,executable="/bin/bash").stdout.strip()


secrets = UserSecretsClient()
hf_token = secrets.get_secret("HF_TOKEN")
github_token = secrets.get_secret("GITHUB_TOKEN")

# passing the github token through git config variables so it never shows up in a command line or a file
auth = base64.b64encode(f"x-access-token:{github_token}".encode()).decode()
git_env = dict(os.environ,GIT_CONFIG_COUNT="1",GIT_CONFIG_KEY_0="http.extraHeader",GIT_CONFIG_VALUE_0=f"Authorization: Basic {auth}")
subprocess.run(["git","clone","--quiet",REPO,str(CODE)],env=git_env,check=True)
subprocess.run(["git","-C",str(CODE),"checkout","--quiet","--detach",COMMIT],check=True)
head = subprocess.run(["git","-C",str(CODE),"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
assert head==COMMIT,f"checked out {head} instead of {COMMIT}"
print(f"code at commit {head}")

start = time.time()
subprocess.run([sys.executable,"-m","pip","install","--quiet",*INSTALL],check=True)
print(f"installed packages in {(time.time()-start)/60:.1f} min")

RESULTS.mkdir(parents=True,exist_ok=True)
if RESUME.exists():
    shutil.copytree(RESUME,RESULTS,dirs_exist_ok=True)
    print(f"continuing from {RESUME}")
(RESULTS/"environment.json").write_text(json.dumps({
    "mode":MODE,
    "commit":COMMIT,
    "started":time.strftime("%Y-%m-%d %H:%M:%S"),
    "gpus":shell("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
    "packages":shell("pip list 2>/dev/null | grep -i -E '^(torch|torchvision|transformers|tokenizers|huggingface.hub|accelerate|safetensors|numpy|pillow) '"),
    "python":sys.version,
},indent=1),encoding="utf-8")

mode = MODES[MODE]
env = dict(os.environ,
    PYTHONPATH=f"{CODE/'src'}:{CODE/'scripts'}",
    TANGLE_RESULTS=str(RESULTS),TANGLE_MODELS=str(MODELS),TANGLE_DATA=str(DATA),
    TANGLE_SCENES=mode["scenes"],TANGLE_CHUNK=mode["chunk"],HF_TOKEN=hf_token)
subprocess.run([sys.executable,"-m","tangle.main","--kaggle",*mode["args"]],env=env,check=True)
