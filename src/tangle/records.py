import csv
import json
import os

import numpy as np


def write_json(path,obj):
    """Write json through a temporary file so a crash never leaves half a file"""
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=1),encoding="utf-8",newline="\n")
    os.replace(tmp,path)


def read_json(path):
    """Read a json file"""
    return json.loads(path.read_text(encoding="utf-8"))


def write_npz(path,arrays):
    """Save numpy arrays through a temporary file"""
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+".tmp.npz")
    np.savez(tmp,**arrays)
    os.replace(tmp,path)


def write_csv(path,rows):
    """Write a list of dicts as csv through a temporary file"""
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    with open(tmp,"w",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp,path)


def read_csv(path):
    """Read a csv file into a list of dicts"""
    with open(path,encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))
