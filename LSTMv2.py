from OPT import *
from LSTM_comb import one_hot_encoding, gandetoken
import config
import numpy as np

datalist = np.load(config.DATA_DIRECTORY / "D0.npy", allow_pickle=True)
datalist = gandetoken(datalist)
one_hot_tensors = one_hot_encoding(datalist)

for layout in one_hot_tensors:
    breakpoint()
    equipment, bounds, x, splitter = bound_creation(layout)
    swarmsize_factor = 7
    particle_size = swarmsize_factor * len(bounds)
    if 5 in equipment:
        particle_size += -1 * swarmsize_factor
    if 9 in equipment:
        particle_size += -2 * swarmsize_factor
    iterations = 30
    nv = len(bounds)
    PSO(objective_function, bounds, particle_size, iterations)
    print("done")
