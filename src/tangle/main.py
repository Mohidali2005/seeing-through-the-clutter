import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description="run every stage of the entanglement study in order")
parser.add_argument("--kaggle",action="store_true",help="run the gpu stages instead of the local ones")
args,rest = parser.parse_known_args()

if args.kaggle:
    subprocess.run([sys.executable,"-m","tangle.run_all",*rest],check=True)
else:
    subprocess.run([sys.executable,"-m","tangle.fetch","data","--check"],check=True)
