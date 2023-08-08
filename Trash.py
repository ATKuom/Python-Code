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

#     h5 = h4 + q_hx
#     t5 = temperature(h5, p5)

#     t2 = [t2 for t2 in range(int(t4) + 5, int(t1))]
#     if len(t2) == 0:
#         return float("inf")
#     h2 = list()
#     for temp in t2:
#         a, _ = enthalpy_entropy(temp, p2)
#         h2.append(a)
#     h2 = np.asarray(h2)
#     q_hx1 = m * h1 - m * h2
#     t5 = [t2 for t2 in range(int(t4), int(t1) - 5)]
#     if len(t5) == 0:
#         return float("inf")
#     h5 = list()
#     for temp in t5:
#         a, _ = enthalpy_entropy(temp, p5)
#         h5.append(a)
#     h5 = np.asarray(h5)
#     q_hx2 = m * h5 - m * h4
#     q_hx = q_hx1 - q_hx2
#     idx1 = np.where(q_hx[:-1] * q_hx[1:] < 0)[0]

# fuel_tur = e6 - e1
#     prod_tur = w_tur

# cost_prod_execo_tur = (fuel_tur + cost_tur) / w_tur

#  fuel_HX = e1 - e2

# fuel_cooler = q_c
#     prod_cooler = e2 - e3

# fuel_comp = w_comp
#     prod_comp = e4 - e3

# ##Cooler
#     t3 = t2 - q_cool / (m * cp)
#     t2 = t1 - q_hx / (m * cp)
#     t5 = t4 + q_hx / (m * cp)
#     t6 = t5 + q_heater / (m * cp)

# ##Economic Analysis

#     ##Exergoeconomic Analysis


#     breakpoint()
#     z = cost_comp
# m1 = np.array(
#         [
#             [e1 - e6, 0, 0, 0, 0, 0, w_tur, 0, 0],
#             [-e1 + e2, 0, 0, -e4, e5, 0, 0, 0, 0],
#             [0, e2, -e3, 0, 0, 0, 0, 0, 0],
#             [0, 0, -e3, e4, 0, 0, 0, w_comp, 0],
#             [0, 0, 0, 0, -e5, e6, 0, 0, q_heater],
#             [1,0,0,0,0,-1,0,0,0],
#             [1,-1,0,0,0,0,0,0,0],
#             [0,0,0,0,0,0,0,0,0],
#             [0,0,0,0,0,0,0,0,1]

# ])

# W
# [c1,c2,c3,c4,c5,c6,cw,cfg]

# m1 = np.array(
#     [
#         [e1, 0, 0, 0, 0, -e6, w_tur, 0],
#         [e1, e2, 0, -e4, e5, 0, 0, 0],
#         [0, -e2, e3, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, -e5, e6, 0, -(e_fgin - e_fgout)],
#         [0, 0, -e3, e4, 0, 0, -w_comp, 0],
#         [1, 0, 0, 0, 0, -1, 0, 0],
#         [1, -1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1],
#     ]
# )
# m1 = np.array(
#     [
#         [e1, 0, 0, 0, 0, -e6, w_tur, 0, 0, 0],
#         [-e1, e2, 0, -e4, e5, 0, 0, 0, 0, 0],
#         [0, -e2, e3, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, -e5, e6, 0, -e_fgin, e_fgout, 0],
#         [0, 0, -e3, e4, 0, 0, -w_comp, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, w_gt, e_fgin, 0, -e_fuel],
#         [1, 0, 0, 0, 0, -1, 0, 0, 0, 0],
#         [1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
#         # [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     ]
# )  # W
# m2 = np.asarray(
#     zk
#     + [
#         0,
#         0,
#         0,
#         # 8.7e-9 * 3600,
#         cfuel * 3600,
#     ]
# ).reshape(-1, 1)

"""
    fuel_chem_ex = 1.26/16.043*824.348  # MW = kg/s /kg/kmol *MJ/kmol
    fuel_phys_ex = 1.26*(0.39758) #MW = kg/s * MJ/kg
    Efuel = fuel_chem_ex + fuel_phys_ex  # MW
    Cp=cfuel*Efuel  + Ztot # $/h
    Ep = 22.4 + w_tur/1e6 - w_comp/1e6 # MW
    Cdiss = c2*e2 - c3*e3 + zk[2] # $/h = $/Wh * W - $/Wh * W + $/h
    Cp = 8700 * (q_heater / 1e6) * 3600 + Ztot  # $/h = $/MJ * MJ/s * s/h + $/h
    Ep = (w_tur - w_comp) / 1e6  # MW
    """

# Economic analysis with np.where
#   for work in w_tur:
#         if work > 0:
#             index = np.where(w_tur == work)[0][0]
#             if index == 0:
#                 index = -1
#             else:
#                 index = index - 1
#             if Temperatures[index] > 550:
#                 ft_tur = 1 + 1.137e-5 * (Temperatures[index] - 550) ** 2
#             else:
#                 ft_tur = 1
#             cost_tur[index + 1] = 406200 * ((work / 1e6) ** 0.8) * ft_tur

#     for work in w_comp:
#         if work > 0:
#             cost_comp[np.where(w_comp == work)[0][0]] = 1230000 * (work / 1e6) ** 0.3992

#     for work in q_cooler:
#         if work > 0:
#             index = np.where(q_cooler == work)[0][0]
#             if index == 0:
#                 index = -1
#             else:
#                 index = index - 1
#             dt1_cooler = Temperatures[index + 1] - cw_temp
#             dt2_cooler = Temperatures[index] - cw_Tout(work)
#             if dt2_cooler < 0 or dt1_cooler < 0:
#                 return PENALTY_VALUE
#             UA_cooler = (work / 1) / lmtd(dt1_cooler, dt2_cooler)  # W / °C
#             if Temperatures[index - 1] > 550:
#                 ft_cooler = 1 + 0.02141 * (Temperatures[index - 1] - 550)
#             else:
#                 ft_cooler = 1
#             cost_cooler[index + 1] = 49.45 * UA_cooler**0.7544 * ft_cooler  # $
#     for work in q_heater:
#         if work > 0:
#             index = np.where(q_heater == work)[0][0]
#             if index == 0:
#                 index = -1
#             else:
#                 index = index - 1
#             fg_tout_i = fg_calculation(fg_m * work / total_heat, work)
#             dt1_heater = fg_tin - Temperatures[index + 1]
#             dt2_heater = fg_tout_i - Temperatures[index]
#             if dt2_heater < 0 or dt1_heater < 0:
#                 return PENALTY_VALUE
#             UA_heater = (work / 1e3) / lmtd(dt1_heater, dt2_heater)  # W / °C
#             cost_heater[index + 1] = 5000 * UA_heater  # Thesis 97/pdf116
#     for work in q_hx:
#         if work > 0:
#             index = np.where(q_hx == work)[0][0]
#             if index == 0:
#                 index = -1
#             else:
#                 index = index - 1
#             dt1_hx = Temperatures[hotside_index - 1] - Temperatures[coldside_index]
#             dt2_hx = Temperatures[hotside_index] - Temperatures[coldside_index - 1]
#             if dt2_hx < 0 or dt1_hx < 0:
#                 return PENALTY_VALUE
#             UA_hx = (work / 1) / lmtd(dt1_hx, dt2_hx)  # W / °C
#             if Temperatures[hotside_index - 1] > 550:
#                 ft_hx = 1 + 0.02141 * (Temperatures[hotside_index - 1] - 550)
#             else:
#                 ft_hx = 1
#             cost_hx[index + 1] = 49.45 * UA_hx**0.7544 * ft_hx  # $

# m1 list method
# m1 = [
#     i[:]
#     for i in [[0] * (len(equipment) + equipment.count(1) + equipment.count(3) + 3)]
#     * len(equipment)
# ]

# a = Fluid(FluidsList.CarbonDioxide).with_state(
#     Input.temperature(100), Input.pressure(101.3e5)
# )
# b = Fluid(FluidsList.CarbonDioxide).with_state(
#     Input.temperature(200), Input.pressure(101.3e5)
# )
# c = b.isenthalpic_expansion_to_pressure(90e5)
# print(c.temperature, b.enthalpy, c.enthalpy)
# d = Fluid(FluidsList.CarbonDioxide).mixing(4, a, 0.4, b)
# print(d.temperature)
# import numpy as np

# ee = [(0, 1), (1, 5), (2, 2), (3, 3), (4, 9), (5, 4), (6, 7), (7, 5), (8, 7), (9, 4)]
# equipment = [1, 5, 2, 3, 9, 4, 7, 5, 7, 4]
# ee2 = np.asarray(ee)
# start = np.where(9 == ee2[:, 1])[0][0]
# end1, end2 = np.where(7 == ee2[:, 1])[0]
# print(start, end1, end2)
# print(equipment)
# equipment = np.roll(equipment, -start)
# ee2 = np.roll(ee2, -start, axis=0)
# start = np.where(9 == ee2[:, 1])[0][0]
# end1, end2 = np.where(7 == ee2[:, 1])[0]
# print(start, end1, end2)
# print(equipment)
# mass = np.ones(len(equipment))
# split_ratio = 0.4
# for i in range(start + 1, end1 + 1):
#     mass[i] = mass[i] * split_ratio
# for i in range(end1 + 1, end2 + 1):
#     mass[i] = mass[i] * (1 - split_ratio)
# print(mass)
# print(np.where(equipment == 7)[0])
