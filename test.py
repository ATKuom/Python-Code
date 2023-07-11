import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.functional as F
import config

# datalist = np.load(config.DATA_DIRECTORY / "D0test.npy", allow_pickle=True)
# example = max(datalist, key=len)
# print(example, len(example))

# ex2 = max(datalist, key=len)
# print(ex2, len(ex2))
# np.save(config.DATA_DIRECTORY / "D0test.npy", allow_pickle=True)

# classes = ["G", "T", "A", "C", "H", "a", "b", "1", "2", "-1", "-2", "E"]
# # datalist = np.array(
# #     [

# #             "GTaACaHE",
# #             "GTaAC-1H1a1HE",
# #             "GTaACH-1H1a1HE",
# #             "GTa1bAC-2H2b2-1aT1HE",

# #     ]
# # )


# def one_hot_encoding(datalist):
#     one_hot_tensors = []
#     for sequence in datalist:
#         # Perform one-hot encoding for the sequence
#         one_hot_encoded = []
#         i = 0
#         while i < len(sequence):
#             char = sequence[i]
#             vector = [0] * len(classes)  # Initialize with zeros

#             if char == "-":
#                 next_char = sequence[i + 1]
#                 unit = char + next_char
#                 if unit in classes:
#                     vector[classes.index(unit)] = 1
#                     i += 1  # Skip the next character since it forms a unit
#             elif char in classes:
#                 vector[classes.index(char)] = 1

#             one_hot_encoded.append(vector)
#             i += 1

#         # Convert the list to a PyTorch tensor
#         one_hot_tensor = torch.tensor(one_hot_encoded)
#         one_hot_tensors.append(one_hot_tensor)

#     return one_hot_tensors


# def padding(one_hot_tensors):
#     # Pad the one-hot tensors to have the same length
#     padded_tensors = pad_sequence(
#         one_hot_tensors, batch_first=True, padding_value=0
#     ).float()

#     return padded_tensors


# one_hot_tensors = one_hot_encoding(datalist)
# padded_tensors = padding(one_hot_tensors)

# print(np.load("D0test.npy", allow_pickle=True))

# comp = np.linspace(5, 25)
# tur = np.linspace(1.01, 25)
# p6 = (-1.5e5 * comp - 2e5) / (1 + comp / tur)
# print(p6)
# p1 = (+1.5e5 * comp + 2e5) / (comp - tur)
# print(p1)

# w = (0.4 / iterations**2) * (i - iterations) ** 2 + 0.4
#             c1 = -3 * (i / iterations) + 3.5
#             c2 = 3 * (i / iterations) + 0.5

# # [p1,p2,p3,p4,p5,p6]
# tur_pratio = 238.9 / 78.5
# comp_pratio = 241.3 / 77
# pres = np.array(
#     [
#         [1, 0, 0, 0, 0, -1 / tur_pratio],
#         [1, -1, 0, 0, 0, 0],
#         [0, 1, -1, 0, 0, 0],
#         [0, 0, comp_pratio, -1, 0, 0],
#         [0, 0, 0, 1, -1, 0],
#         [0, 0, 0, 0, 1, -1],
#         [0, 0, 0, 0, 0, 1],
#     ]
# )
# dp = np.array([0, 1e5, 0.5e5, 0, 1e5, 1e5, np.random.randint(74e5, 300e5)]).reshape(
#     -1, 1
# )
# pressures = np.linalg.lstsq(pres, dp)
# print(pressures)
from functions import gammacalc, enthalpy_entropy, specificheat

tur_pratio = 238.9 / 78.5
ntur = 0.85
m = 93.18
t6 = 411.4
p6 = 238.9e5
t3 = 32.3
p3 = 77e5
p4 = 241.3e5
comp_pratio = p4 / p3
ncomp = 0.82
K = 273.15
p1 = 78.5
# (h6, s6) = enthalpy_entropy(t6, p6)  # J/kg, J/kgK = °C,Pa
# cp6 = specificheat(t6, p6)
# gamma = gammacalc(t6, p6)
# print(gamma)
# t1 = (t6 + K) - ntur * ((t6 + K) * (1 - 1 / (tur_pratio ** (1 - 1 / gamma)))) - K  # °C
# (h1, s1) = enthalpy_entropy(t1, p1)
# cp1 = specificheat(t1, p1)
# dh = m * (cp6 * (t6 - t1))
# w_tur = m * (h6 - h1)  # W = kg/s*J/kg
# (h3, s3) = enthalpy_entropy(t3, p3)
# gamma = gammacalc(t3, p3)
# cp3 = specificheat(t3, p3)
# print(gamma)
# t4 = (t3 + K) + ((t3 + K) * (comp_pratio ** (1 - 1 / gamma) - 1)) / ncomp - K  # °C
# (h4, s4) = enthalpy_entropy(t4, p4)
# cp4 = specificheat(t4, p4)
# dc = m * (cp3 * (t4 - t3))
# w_comp = m * (h4 - h3)  # W = kg/s*J/kg

from pyfluids import Fluid, FluidsList, Input

heater = (
    Fluid(FluidsList.CarbonDioxide)
    .with_state(Input.temperature(206.1), Input.pressure(238.9e5))
    .heating_to_temperature(411.4, 0)
)
h6, s6, t6, p6 = heater.enthalpy, heater.entropy, heater.temperature, heater.pressure
turb = (
    Fluid(FluidsList.CarbonDioxide)
    .with_state(Input.temperature(t6), Input.pressure(p6))
    .expansion_to_pressure(78.5e5, 85)
)
h1, s1, t1, p1 = turb.enthalpy, turb.entropy, turb.temperature, turb.pressure
print(t1, p1)
hxer_hotside = (
    Fluid(FluidsList.CarbonDioxide)
    .with_state(Input.temperature(t1), Input.pressure(p1))
    .cooling_to_temperature(81.4, 0.7e5)
)
h2, s2, t2, p2 = (
    hxer_hotside.enthalpy,
    hxer_hotside.entropy,
    hxer_hotside.temperature,
    hxer_hotside.pressure,
)
cooler = (
    Fluid(FluidsList.CarbonDioxide)
    .with_state(Input.temperature(t2), Input.pressure(p2))
    .cooling_to_temperature(32.3, 0.8e5)
)
h3, s3, t3, p3 = cooler.enthalpy, cooler.entropy, cooler.temperature, cooler.pressure
comp = (
    Fluid(FluidsList.CarbonDioxide)
    .with_state(Input.temperature(t3), Input.pressure(p3))
    .compression_to_pressure(241.3e5, 82)
)
h4, s4, t4 = comp.enthalpy, comp.entropy, comp.temperature
hxer_coldside = (
    Fluid(FluidsList.CarbonDioxide)
    .with_state(Input.temperature(t4), Input.pressure(p4))
    .heating_to_temperature(206.1, 2.4e5)
)
h5, s5, t5, p5 = (
    hxer_coldside.enthalpy,
    hxer_coldside.entropy,
    hxer_coldside.temperature,
    hxer_coldside.pressure,
)
w_tur = m * (h6 - h1)  # W = kg/s*J/kg
w_comp = m * (h4 - h3)  # W = kg/s*J/kg
print(w_tur / 1e6, w_comp / 1e6)
print(h1, h2, h4, h5, (h1 - h2) / 1e6, (h5 - h4) / 1e6)
