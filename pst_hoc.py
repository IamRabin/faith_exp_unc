
import json
import pandas as pd
import glob
import numpy as np

vals = []

for path in glob.glob("runs/cifar10/ours/seed2_20260129_070932/lightning_logs/version_0/metrics.csv"):
    df = pd.read_csv(path)
    print(df["val/acc1"].max())
    vals.append(df["val/acc1"].max())

vals = np.array(vals)
print("val acc1: mean =", vals.mean(), "std =", vals.std())

##############################################

import json
import pandas as pd
import glob
import numpy as np

vals = []

for path in glob.glob("/home/rabink1/D1/projects/rrr/lightning_logs/version_0/metrics.csv"):
    df = pd.read_csv(path)
    print(df["val/acc1"].max())
    vals.append(df["val/acc1"].max())

vals = np.array(vals)
print("val acc1: mean =", vals.mean(), "std =", vals.std())


#standard_ce-cifar10-0.953000009059906
# ours-cifar10-0.9472000002861024
#ours_without faith=0.95599
