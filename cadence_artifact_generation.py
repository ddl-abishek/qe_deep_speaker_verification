from datetime import datetime
import time
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("num_artifacts",help="number of artifacts/textfiles to generate",type=int)
args = parser.parse_args()

for _ in range(args.num_artifacts):
    os.system(f"sudo touch /mnt/artifacts/results/{str(datetime.now()).replace(' ','').replace('.','')}.txt")
    time.sleep(0.5)