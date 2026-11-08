import numpy as np
import pandas as pd
from scipy.stats import binomtest

from tangle import records

BOOT = 10000


def load_notes(folder):
    """Join the chunk files of one notes step into one dict of arrays"""
    files = sorted(folder.glob("chunk_*.npz"))
    assert files,f"no notes in {folder}"
    parts = [dict(np.load(f)) for f in files]
    out = {}
    offset = 0
    for part in parts:
        # object rows point at images inside their own chunk so they are shifted to the joined numbering
        part["obj_image"] = part["obj_image"]+offset
        offset += len(part["image_id"])
        for name,value in part.items():
            out.setdefault(name,[]).append(value)
    joined = {}
    for name,values in out.items():
        joined[name] = values[0] if name in ["vision_layers","llm_layers"] else np.concatenate(values)
    return joined


def load_swap(folder):
    """Join the chunk files of one swap step into one table"""
    files = sorted(folder.glob("chunk_*.csv"))
    assert files,f"no swap results in {folder}"
    table = pd.concat([pd.DataFrame(records.read_csv(f)) for f in files],ignore_index=True)
    for name in ["exact_match","parsed","greedy_gold"]:
        table[name] = table[name].map({"True":True,"False":False,"":None})
    for name in ["logprob","check_diff","abs_error","total_bias","new_tokens","n_object","n_background","n_clutter"]:
        table[name] = pd.to_numeric(table[name],errors="coerce")
    table["scene_id"] = table["scene_id"].astype(int)
    return table


def scene_resamples(scenes,seed=0):
    """Draw the same bootstrap resamples of scenes for every quantity"""
    rng = np.random.default_rng(seed)
    return rng.integers(0,len(scenes),size=(BOOT,len(scenes)))


def boot_ratio(num,den,draws):
    """Point estimate and 95 percent range of a ratio of per scene sums over bootstrap resamples of scenes"""
    estimate = num.sum()/den.sum()
    values = num[draws].sum(1)/den[draws].sum(1)
    low,high = np.percentile(values,[2.5,97.5])
    return estimate,low,high


def mcnemar(gain,loss):
    """Exact two sided mcnemar test on the discordant pairs"""
    if gain+loss==0:
        return 1.0
    return binomtest(gain,gain+loss,0.5).pvalue


def fmt(estimate,low,high,scale=100,digits=1):
    """Write an estimate with its range"""
    return f"{estimate*scale:.{digits}f} [{low*scale:.{digits}f}, {high*scale:.{digits}f}]"
