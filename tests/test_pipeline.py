import os
import subprocess
import sys

import numpy as np
import pandas as pd

from tangle.analysis.common import load_notes
from tangle.analysis.common import load_swap
from tangle.paths import ROOT


def run(module,env,*args):
    """Run one module of the package as its own process and return what it printed"""
    done = subprocess.run([sys.executable,"-m",module,*args],env=env,capture_output=True,text=True)
    assert done.returncode==0,done.stderr[-3000:]
    return done.stdout


def test_workers_resume_and_analysis(tiny,tmp_path):
    """Two workers fill every chunk of both stages and a second run adds nothing and the analysis reads the result"""
    _,models = tiny
    results = tmp_path/"results"
    env = dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT/"src"),str(ROOT/"scripts")]),
               TANGLE_MODELS=str(models),TANGLE_RESULTS=str(results),TANGLE_ANALYSIS=str(tmp_path/"analysis"),
               TANGLE_DTYPE="float32",TANGLE_DEVICE_MAP="cpu",TANGLE_SCENES="6",TANGLE_CHUNK="3")
    for stage in ["notes","swap"]:
        for worker in ["0","1"]:
            run(f"tangle.{stage}",env,"objects","--worker",worker,"--workers","2")
    assert len(list((results/"notes"/"qwen4b"/"objects").glob("chunk_*.npz")))==2
    assert len(list((results/"swap"/"qwen4b"/"objects").glob("chunk_*.csv")))==2
    again = run("tangle.notes",env,"objects","--worker","0","--workers","2")
    assert "0 new chunks" in again

    notes = load_notes(results/"notes"/"qwen4b"/"objects")
    assert len(notes["image_id"])==24
    assert notes["embed_diff"].sum()==0 and notes["pixel_mismatch"].sum()==0
    assert notes["dup_diff"].max()==0 and notes["nonfinite"].sum()==0
    assert (notes["leak_vision"][:,0]*notes["untouched_patch"]).max()==0
    assert notes["obj_image"].max()==23

    swap = load_swap(results/"swap"/"qwen4b"/"objects")
    assert len(swap)==24*(2+19)
    # cpu matrix products round a row a little differently depending on its place in the batch
    assert (swap.loc[swap["condition"].isin(["A","D"]),"check_diff"]<1e-5).all()
    wide = swap.pivot_table(index="image_id",columns="condition",values="answer",aggfunc="first")
    assert (wide["A"]==wide["cluttered"]).all() and (wide["D"]==wide["clean"]).all()
    lp = swap.pivot_table(index="image_id",columns="condition",values="logprob",aggfunc="first")
    assert np.allclose(lp["A"],lp["cluttered"]) and np.allclose(lp["D"],lp["clean"])

    for module in ["tangle.analysis.effects","tangle.analysis.leak","tangle.analysis.probe"]:
        run(module,env,str(results))
    probe = pd.read_csv(tmp_path/"analysis"/"stats"/"probe_qwen4b_objects.csv")
    first = probe[(probe["layer"]=="patch embedding")&(probe["part"]=="object")].iloc[0]
    assert first["accuracy"]==0.5
    assert (tmp_path/"analysis"/"stats"/"swap_qwen4b_objects.md").exists()
