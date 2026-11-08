import argparse
import os
import time

from tangle import config
from tangle import data


def worker_args(description):
    """Read the arguments every gpu worker takes"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("dataset",choices=list(config.DATASETS))
    parser.add_argument("--model",default="qwen4b",choices=list(config.MODELS))
    parser.add_argument("--worker",type=int,default=0)
    parser.add_argument("--workers",type=int,default=1)
    return parser.parse_args()


def my_chunks(dataset,worker,workers):
    """Split the scenes into chunks and keep the chunks this worker handles"""
    # smoke runs use fewer scenes and smaller chunks so every worker still gets work
    limit = int(os.environ.get("TANGLE_SCENES","0")) or None
    size = int(os.environ.get("TANGLE_CHUNK",str(config.CHUNK)))
    scenes = data.load_scenes(dataset,limit)
    chunks = [scenes[i:i+size] for i in range(0,len(scenes),size)]
    return [(k,chunk) for k,chunk in enumerate(chunks) if k%workers==worker],len(chunks)


def past_deadline():
    """Tell whether the run has used up its time so no new chunk should start"""
    start = float(os.environ.get("TANGLE_START",str(time.time())))
    return (time.time()-start)/3600>config.HARD_DEADLINE_H
