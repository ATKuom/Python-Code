from pyfluids import Fluid, FluidsList, Input
import numpy as np
import scipy.optimize as optimize

# heater = (
#     Fluid(FluidsList.CarbonDioxide)
#     .with_state(Input.temperature(206.1), Input.pressure(238.9e5))
#     .heating_to_temperature(411.4, 0)
# )
# h6, s6, t6, p6 = heater.enthalpy, heater.entropy, heater.temperature, heater.pressure
# turb = (
#     Fluid(FluidsList.CarbonDioxide)
#     .with_state(Input.temperature(t6), Input.pressure(p6))
#     .expansion_to_pressure(78.5e5, 85)
# )
# h1, s1, t1, p1 = turb.enthalpy, turb.entropy, turb.temperature, turb.pressure

# hxer_hotside = (
#     Fluid(FluidsList.CarbonDioxide)
#     .with_state(Input.temperature(t1), Input.pressure(p1))
#     .cooling_to_temperature(81.4, 0.7e5)
# )
# h2, s2, t2, p2 = (
#     hxer_hotside.enthalpy,
#     hxer_hotside.entropy,
#     hxer_hotside.temperature,
#     hxer_hotside.pressure,
# )
# cooler = (
#     Fluid(FluidsList.CarbonDioxide)
#     .with_state(Input.temperature(t2), Input.pressure(p2))
#     .cooling_to_temperature(32.3, 0.8e5)
# )
# h3, s3, t3, p3 = cooler.enthalpy, cooler.entropy, cooler.temperature, cooler.pressure
# comp = (
#     Fluid(FluidsList.CarbonDioxide)
#     .with_state(Input.temperature(t3), Input.pressure(p3))
#     .compression_to_pressure(241.3e5, 82)
# )
# h4, s4, t4, p4 = comp.enthalpy, comp.entropy, comp.temperature, comp.pressure
# hxer_coldside = (
#     Fluid(FluidsList.CarbonDioxide)
#     .with_state(Input.temperature(t4), Input.pressure(p4))
#     .heating_to_temperature(206.1, 2.4e5)
# )
# h5, s5, t5, p5 = (
#     hxer_coldside.enthalpy,
#     hxer_coldside.entropy,
#     hxer_coldside.temperature,
#     hxer_coldside.pressure,
# )
# delta_t = 10
# t2 = t4 + delta_t
# import scipy.optimize as opt


# def objective(t):
#     print(t)
#     hotside = (
#         Fluid(FluidsList.CarbonDioxide)
#         .with_state(Input.temperature(t1), Input.pressure(p1))
#         .cooling_to_temperature(t[0], 1e5)
#     )
#     coldside = (
#         Fluid(FluidsList.CarbonDioxide)
#         .with_state(Input.temperature(t4), Input.pressure(p5))
#         .heating_to_temperature(t[1], 1e5)
#     )
#     return hotside.enthalpy - coldside.enthalpy


# x0 = [t1 - delta_t, t4 + delta_t]
# t2, t5 = opt.brent(objective, x0)
# printt2)
# e1 = 0.08e6
# e2 = 27.62e6
# e3 = 65.4e6
# e4 = 73.97e6
# e5 = 18.02e6
# e6 = 4.43e6
# e7 = 40.08e6  # e6
# e8 = 28.98e6  # e1
# e9 = 20.39e6  # e2
# e10 = 18.77e6  # e3
# e11 = 21.12e6  # e4
# e12 = 27.79e6  # e5
# e13 = 0.52e6
# e14 = 1.33e6
# e101 = 29.72e6
# e102 = 22.85e6
# e103 = 2.77e6
# e104 = 10.20e6
# e105 = 22.4e6
# e106 = 2.77e6
# e107 = 9.69e6
# e108 = 29.3126e6
# m1 = np.array(
#     [
#         [e1, -e2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e101, 0, 0, 0, 0, 0, 0],
#         [0, e2, e3, -e4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, e4, -e5, 0, 0, 0, 0, 0, 0, 0, 0, 0, -e101, -e102, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e102, 0, 0, -e105, 0, 0],
#         [0, 0, 0, 0, e5, -e6, -e7, 0, 0, 0, 0, e12, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, e7, -e8, 0, 0, 0, 0, 0, 0, 0, 0, 0, -e104, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, e8, -e9, 0, e11, -e12, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, e9, -e10, 0, 0, e13, -e14, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, e10, -e11, 0, 0, 0, 0, 0, e103, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e104, 0, 0, -e107],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -e103, 0, 0, e106, 0],
#         [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     ]
# )  # W
# # m1 = m1 * 3600 / 1e9  # GJ/h
# x = np.array(
#     [
#         0,
#         13.6,
#         3.8,
#         8.9,
#         8.9,
#         8.9,
#         21.9,
#         21.9,
#         21.9,
#         21.9,
#         23.7,
#         25.6,
#         0,
#         0,
#         10.8,
#         10.8,
#         20.3,
#         28.4,
#         11.8,
#         17.7,
#         31.2,
#     ]
# )
# x = x * 3600 / 1e9
# m2 = np.array(
#     [
#         -191.31,
#         -127.54,
#         -255.08,
#         -63.77,
#         -165.54,
#         -170.85,
#         -76.94,
#         -82.92,
#         -121.17,
#         -44.91,
#         -26.15,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         3.8 * 3600 / 1e9,
#         0,
#         0,
#     ]
# )

# c1 = m1 @ x
# print(c1)
# c, _, _, _ = np.linalg.lstsq(m1, m2, rcond=None)
# costs = np.linalg.solve(m1, m2)
# costs2, _ = optimize.nnls(m1, m2)
# # print(c)
# # print(c[1])
# # print(c[2])
# print(costs2 * 1e9 / 3600)
# # print(costs2[1])
# # print(costs2[2])
# print(costs * 1e9 / 3600)
# cdiss = (
#     e9 * costs[8] - e10 * costs[9] + e13 * costs[12] - e14 * costs[13]
# ) * 1e9 / 3600 + 82.92
# print(cdiss)
