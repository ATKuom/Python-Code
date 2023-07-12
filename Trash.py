# def h_s_fg_old(T, P):
#     nitrogen = Fluid(FluidsList.Nitrogen).with_state(
#         Input.pressure(P), Input.temperature(T)
#     )
#     oxygen = Fluid(FluidsList.Oxygen).with_state(
#         Input.pressure(P), Input.temperature(T)
#     )
#     water = Fluid(FluidsList.Water).with_state(Input.pressure(P), Input.temperature(T))
#     carbon_dioxide = Fluid(FluidsList.CarbonDioxide).with_state(
#         Input.pressure(P), Input.temperature(T)
#     )
#     h = (
#         nitrogen.enthalpy * 0.753
#         + oxygen.enthalpy * 0.1553
#         + carbon_dioxide.enthalpy * 0.0505
#         + water.enthalpy * 0.0412
#     )
#     s = (
#         nitrogen.entropy * 0.753
#         + oxygen.entropy * 0.1553
#         + carbon_dioxide.entropy * 0.0505
#         + water.entropy * 0.0412
#     )
#     cp = (
#         nitrogen.specific_heat * 0.753
#         + oxygen.specific_heat * 0.1553
#         + carbon_dioxide.specific_heat * 0.0505
#         + water.specific_heat * 0.0412
#     )
#     return (h, s, cp)

# def exhaust():
#     exhaust_mass_flow = 68.75
#     exhaust_inlet_T = 539.8
#     exhaust_inlet_P = 10e5
#     flue_gas = Mixture(
#         [
#             FluidsList.Nitrogen,
#             FluidsList.Oxygen,
#             FluidsList.CarbonDioxide,
#             FluidsList.Water,
#         ],
#         [75.3, 15.53, 05.05, 04.12],
#     )
#     exhaust_inlet = flue_gas.with_state(
#         Input.temperature(exhaust_inlet_T), Input.pressure(exhaust_inlet_P)
#     )
#     exhaust_inlet_h = exhaust_inlet.enthalpy
#     print(exhaust_inlet_h)

#     def temperature(h, P):
#     """
#     Takes the the temperature and pressure of a CO2 stream and gives enthalpy, entropy and specific heat values at that temperature
#     Temperature input is C, Pressure input is pa
#     Return: Enthalpy (J/kg), Entropy (J/kgK), Specific Heat (J/kgK)

#     """
#     substance = Fluid(FluidsList.CarbonDioxide).with_state(
#         Input.enthalpy(h), Input.pressure(P)
#     )
#     return substance.temperature

# def gammacalc(T, P):
#     substance = Fluid(FluidsList.CarbonDioxide).with_state(
#         Input.pressure(P), Input.temperature(T)
#     )
#     R = 189
#     gamma = substance.specific_heat / (substance.specific_heat - R)
#     return gamma

# def specificheat(T, P):
#     substance = Fluid(FluidsList.CarbonDioxide).with_state(
#         Input.pressure(P), Input.temperature(T)
#     )
#     return substance.specific_heat

# from pyfluids import Fluid, FluidsList, Input, InputHumidAir, HumidAir


# def enthalpy_entropy(T, P):
#     """
#     Takes the the temperature and pressure of a CO2 stream and gives enthalpy, entropy and specific heat values at that temperature
#     Temperature input is C, Pressure input is pa
#     Return: Enthalpy (J/kg), Entropy (J/kgK), Specific Heat (J/kgK)

#     """
#     substance = Fluid(FluidsList.CarbonDioxide).with_state(
#         Input.pressure(P), Input.temperature(T)
#     )
#     return (substance.enthalpy, substance.entropy, substance.specific_heat)

#     # def h_s_fg(T, P):
#     nitrogen = Fluid(FluidsList.Nitrogen).with_state(
#         Input.pressure(P), Input.temperature(T)
#     )
#     oxygen = Fluid(FluidsList.Oxygen).with_state(
#         Input.pressure(P), Input.temperature(T)
#     )
#     h = nitrogen.enthalpy * 0.77 + oxygen.enthalpy * 0.23
#     s = nitrogen.entropy * 0.77 + oxygen.entropy * 0.23
#     return (h, s)


# def h_s_fg(T, P):
#     nitrogen = Fluid(FluidsList.Nitrogen).with_state(
#         Input.pressure(P), Input.temperature(T)
#     )
#     oxygen = Fluid(FluidsList.Oxygen).with_state(
#         Input.pressure(P), Input.temperature(T)
#     )
#     water = Fluid(FluidsList.Water).with_state(Input.pressure(P), Input.temperature(T))
#     carbon_dioxide = Fluid(FluidsList.CarbonDioxide).with_state(
#         Input.pressure(P), Input.temperature(T)
#     )
#     h = (
#         nitrogen.enthalpy * 0.753
#         + oxygen.enthalpy * 0.1553
#         + carbon_dioxide.enthalpy * 0.0505
#         + water.enthalpy * 0.0412
#     )
#     s = (
#         nitrogen.entropy * 0.753
#         + oxygen.entropy * 0.1553
#         + carbon_dioxide.entropy * 0.0505
#         + water.entropy * 0.0412
#     )
#     cp = (
#         nitrogen.specific_heat * 0.753
#         + oxygen.specific_heat * 0.1553
#         + carbon_dioxide.specific_heat * 0.0505
#         + water.specific_heat * 0.0412
#     )
#     return (h, s, cp)


# def air(T, P):
#     air = Fluid(FluidsList.Air).with_state(Input.temperature(T), Input.pressure(P))
#     return (air.enthalpy, air.entropy)


# T0 = 15
# P0 = 101325
# K = 273.15
# (h0, s0, c0) = enthalpy_entropy(T0, P0)

# if __name__ == "__main__":
#     t = 539.76  # 217.99
#     p = 101325
#     # Exergy_stream = h1 - h0 - (T0 + K) * (s1 - s0)
#     hfg0, sfg0, cp0 = h_s_fg(T0, P0)
#     hfg1, sfg1, cp1 = h_s_fg(t, p)
#     hfg2, sfg2, cp2 = h_s_fg(218, 1.01e5)
#     Exergy_stream = hfg1 - hfg0 - (T0 + K) * (sfg1 - sfg0)
#     hex0, sex0 = air(T0, P0)
#     hexin, sexin = air(t, p)
#     hexout, sexout = air(218, 1.01e5)
#     exergy_exhaust1 = hexin - hex0 - (T0 + K) * (sexin - sex0)
#     # h1, s1, c1 = enthalpy_entropy(70.6, 241.3e5)
#     # h2, s2, c2 = enthalpy_entropy(32.3, 77e5)
#     # h3, s3, c3 = enthalpy_entropy(411.4, 238.9e5)
#     # h4, s4, c4 = enthalpy_entropy(297.6, 78.5e5)
#     print(
#         # h1 / 1e3,
#         # s1 / 1e3,
#         # # 1.26 * Exergy_stream / 1e6,
#         hfg1,
#         sfg1 / 1e3,
#         # sfg2 / 1e3,
#         68.75 * Exergy_stream / 1e6,
#         hexin / 1e3,
#         hexout / 1e3,
#         (hexin - hexout) / 1e3,
#         68.75 * exergy_exhaust1 / 1e6,
#         # cp1,
#         # c0,
#     )


# import scipy.optimize as opt
# import CoolProp.CoolProp as CP

# q_heater = 359989.4810921619
# tout = 217.99
# K = 273.15

# fg_0_h = CP.PropsSI(
#     "H",
#     "P|gas",
#     101325,
#     "T",
#     15 + K,
#     "Nitrogen[0.7643]&Oxygen[0.1382]&Water[0.0650]&CarbonDioxide[0.0325]",
# )
# fg_0_s = CP.PropsSI(
#     "S",
#     "P|gas",
#     101325,
#     "T",
#     15 + K,
#     "Nitrogen[0.7643]&Oxygen[0.1382]&Water[0.0650]&CarbonDioxide[0.0325]",
# )
# # fg_out_T = CP.PropsSI(
# #     "T",
# #     "P|gas",
# #     101325,
# #     "H",
# #     fg_in_h - delta_h,
# #     "Nitrogen[0.7643]&Oxygen[0.1382]&Water[0.0650]&CarbonDioxide[0.0325]",
# # )
# # fg_out_s = CP.PropsSI(
# #     "S",
# #     "P|gas",
# #     101325,
# #     "T",
# #     fg_out_T,
# #     "Nitrogen[0.7643]&Oxygen[0.1382]&Water[0.0650]&CarbonDioxide[0.0325]",
# # )
# # fg_in_exergy = fg_in_h - fg_0_h - 298.15 * (fg_in_s - fg_0_s)
# # print(fg_in_h)

# # fg_out_exergy = fg_out_h - fg_0_h - 298.15 * (fg_out_s - fg_0_s)
# # print(68.75 * fg_in_exergy / 1e6, 68.75 * fg_out_exergy / 1e6)
# # print(fg_in_s, fg_out_s)
# # AS = CP.AbstractState("HEOS", "Nitrogen&Oxygen&CO2&Water")
# # AS.specify_phase(CP.iphase_gas)
# # AS.set_mole_fractions([0.7643, 0.1382, 0.065, 0.0325])
# # AS.update(CP.PT_INPUTS, 101325, 217.99 + K)
# # print(AS.hmass())

# # AS = CP.AbstractState("HEOS", "Nitrogen&Oxygen&Water&CO2")
# # AS.set_mole_fractions([0.7643, 0.1382, 0.065, 0.0325])
# # AS.specify_phase(CP.iphase_gas)
# # AS.update(CP.PT_INPUTS, 101325, 15 + K)


# # def objective(T):
# #     AS.update(CP.PT_INPUTS, 101325, T)
# #     return 969266.9985077961 - AS.hmass()


# # def objective2(T):
# #     fg_out_h = CP.PropsSI(
# #         "H",
# #         "P|gas",
# #         101325,
# #         "T",
# #         T,
# #         "Nitrogen[0.7643]&Oxygen[0.1382]&Water[0.0650]&CarbonDioxide[0.0325]",
# #     )
# #     return fg_in_h - fg_out_h


# # Solve for isentropic temperature
# # T2s = opt.newton(objective2, 15 + K)
# # print(T2s - K, AS.hmass(), AS.smass())
# # print(539.76, fg_in_h, fg_in_s)
# # print()
# # Use isentropic temp to get h2s
# # AS.update(CP.PT_INPUTS, 1.6e5, T2s)
# # s2 = AS.smass()
# # h2s = AS.hmass()
# # print((620e3 - 24.7e6 / 68.75), h2s, s2)

# h, s = CP.PropsSI(
#     ("H", "S"),
#     "P|gas",
#     101325,
#     "T",
#     15 + K,
#     "Nitrogen[0.7643]&Oxygen[0.1382]&Water[0.0650]&CarbonDioxide[0.0325]",
# )
# print(h, s)

# import CoolProp.CoolProp as CP
# import matplotlib.pyplot as plt

# HEOS = CP.AbstractState('HEOS', 'Methane&Ethane')

# for x0 in [0.02, 0.2, 0.4, 0.6, 0.8, 0.98]:
#     HEOS.set_mole_fractions([x0, 1 - x0])
#     try:
#         HEOS.build_phase_envelope("dummy")
#         PE = HEOS.get_phase_envelope_data()
#         PELabel = 'Methane, x = ' + str(x0)
#         plt.plot(PE.T, PE.p, '-', label=PELabel)
#     except ValueError as VE:
#         print(VE)

# plt.xlabel('Temperature [K]')
# plt.ylabel('Pressure [Pa]')
# plt.yscale('log')
# plt.title('Phase Envelope for Methane/Ethane Mixtures')
# plt.legend(loc='lower right', shadow=True)
# plt.savefig('methane-ethane.pdf')
# plt.savefig('methane-ethane.png')
# plt.close()

# import torch
# from torch.nn.utils.rnn import pad_sequence
# import numpy as np
# import torch.nn.functional as F
# import config

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
# from functions import gammacalc, enthalpy_entropy, specificheat

# tur_pratio = 238.9 / 78.5
# ntur = 0.85
# m = 93.18
# t6 = 411.4
# p6 = 238.9e5
# t3 = 32.3
# p3 = 77e5
# p4 = 241.3e5
# comp_pratio = p4 / p3
# ncomp = 0.82
# K = 273.15
# p1 = 78.5
# # (h6, s6) = enthalpy_entropy(t6, p6)  # J/kg, J/kgK = °C,Pa
# # cp6 = specificheat(t6, p6)
# # gamma = gammacalc(t6, p6)
# # print(gamma)
# # t1 = (t6 + K) - ntur * ((t6 + K) * (1 - 1 / (tur_pratio ** (1 - 1 / gamma)))) - K  # °C
# # (h1, s1) = enthalpy_entropy(t1, p1)
# # cp1 = specificheat(t1, p1)
# # dh = m * (cp6 * (t6 - t1))
# # w_tur = m * (h6 - h1)  # W = kg/s*J/kg
# # (h3, s3) = enthalpy_entropy(t3, p3)
# # gamma = gammacalc(t3, p3)
# # cp3 = specificheat(t3, p3)
# # print(gamma)
# # t4 = (t3 + K) + ((t3 + K) * (comp_pratio ** (1 - 1 / gamma) - 1)) / ncomp - K  # °C
# # (h4, s4) = enthalpy_entropy(t4, p4)
# # cp4 = specificheat(t4, p4)
# # dc = m * (cp3 * (t4 - t3))
# # w_comp = m * (h4 - h3)  # W = kg/s*J/kg
