import argparse
import glob
import os

import numpy as np

parser = argparse.ArgumentParser(description='Showing Results')
parser.add_argument('--dir', required=True)
args = parser.parse_args()

length = max(map(lambda x: len(x), os.listdir(args.dir)))
for example in sorted(os.listdir(args.dir)):
    logs = glob.glob(os.path.join(args.dir, example, "*.log"))
    accs = []
    for log in logs:
        with open(log, 'r') as f:
            results = f.readlines()[-1].strip()
        if "Epoch: 119" in results:
            acc = results.split()[-1][:-1]
            accs.append(eval(acc))
    if len(accs) > 0:
        print(f"{example:<{length}}, {len(accs)}, {np.mean(accs):<.2f}, {np.std(accs):.2f}")
