import argparse
import hashlib
import zipfile

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

from tangle import config
from tangle import records
from tangle.paths import DATA
from tangle.paths import MODELS
from tangle.paths import RESULTS

parser = argparse.ArgumentParser(description="download the datasets or one model and check them")
parser.add_argument("what",help="data or a model name")
parser.add_argument("--check",action="store_true",help="only check the local datasets against the pinned hashes")
args = parser.parse_args()

if args.what=="data":
    found = {}
    for name,spec in config.DATASETS.items():
        folder = DATA/spec["folder"]
        sha = None
        if not args.check:
            sha = HfApi().dataset_info(spec["repo"]).sha
            archive = hf_hub_download(spec["repo"],f"{spec['folder']}.zip",repo_type="dataset",revision=sha,cache_dir=str(MODELS/"hf_cache"))
            zipfile.ZipFile(archive).extractall(DATA)
        meta = hashlib.md5((folder/"metadata.csv").read_bytes()).hexdigest()
        images = hashlib.sha256()
        rows = records.read_csv(folder/"metadata.csv")
        for row in rows:
            images.update((folder/row["file_name"]).read_bytes())
        assert meta==spec["metadata_md5"],f"{name} metadata differs from the local build"
        assert images.hexdigest()==spec["images_sha256"],f"{name} images differ from the local build"
        found[name] = {"repo":spec["repo"],"commit":sha,"rows":len(rows),"metadata_md5":meta}
        print(f"{name}: {len(rows)} rows match the local build")
    if not args.check:
        records.write_json(RESULTS/"data.json",found)
else:
    spec = config.MODELS[args.what]
    folder = snapshot_download(spec["repo"],revision=spec["revision"],local_dir=str(MODELS/args.what),allow_patterns=["*.json","*.safetensors","*.txt","*.jinja","tokenizer*"])
    print(f"{args.what}: downloaded {spec['repo']} at {spec['revision'][:10]} into {folder}")
