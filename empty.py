import pandas as pd
import numpy as np
import config

np.printoptions(threshold=np.inf)
pd.set_option("display.max_rows", None)
datalist = np.load(config.DATA_DIRECTORY / "v4D0_m1.npy", allow_pickle=True)
datalist2 = np.load(config.DATA_DIRECTORY / "v4D1_m1.npy", allow_pickle=True)
index = np.where(np.isin(datalist2, datalist, invert=True))[0]
new_ones = datalist2[index]
dataset = pd.DataFrame(new_ones)
print(dataset)
