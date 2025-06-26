# Design is TaACaH no splitter, no mixer, 1 mass flow of CO2
from pyfluids import Fluid, FluidsList, Input
import numpy as np

# Parameters
ntur = 0.93  # 2019 Nabil
ncomp = 0.89  # 2019 Nabil
gamma = 1.28  # 1.28 or 1.33 can be used based on the assumption
t0 = 15  # °C
p0 = 101325  # pa
K = 273.15
# Variables
m = 1
t = [t1, t2, t3, t4, t5, t6]
p = [p1, p2, p3, p4, p5, p6]
tur_pratio = 1
comp_pratio = 1

# Inequality Constraints
t1, t2, t3, t4, t5, t6 >= 35
t1, t2, t3, t4, t5, t6 <= 560
p1, p2, p3, p4, p5, p6 <= 250e6


def enthalpy_entropy(T, P):
    """
    Takes the the temperature and pressure of a CO2 stream and gives enthalpy, entropy and specific heat values at that temperature
    Temperature input is C, Pressure input is pa
    Return: Enthalpy (J/kg), Entropy (J/kgK), Specific Heat (J/kgK)

    """
    substance = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.pressure(P), Input.temperature(T)
    )
    return (substance.enthalpy, substance.entropy)


h0, s0 = enthalpy_entropy(t0, p0)

h, s = enthalpy_entropy(t, p)

e1 = m * (h[0] - h0 - t0 * (s[0] - s0))
e2 = m * (h[1] - h0 - t0 * (s[1] - s0))
e3 = m * (h[2] - h0 - t0 * (s[2] - s0))
e4 = m * (h[3] - h0 - t0 * (s[3] - s0))
e5 = m * (h[4] - h0 - t0 * (s[4] - s0))
e6 = m * (h[5] - h0 - t0 * (s[5] - s0))


def lmtd(dt1, dt2):
    return (dt1 - dt2) / np.log(dt1 - dt2)


def turbine():
    t[0] = (
        (t[5] + K)
        - ntur * ((t[5] + K) - (t[5] + K) / (tur_pratio ** (1 - 1 / gamma)))
        - K
    )
    p[5] = tur_pratio * p[0]
    h[0], _ = enthalpy_entropy(t[0], p[0])
    h[5], _ = enthalpy_entropy(t[5], p[5])
    w_tur = m * (h[5] - h[0])
    efuel = e6 - e1
    eprod = w_tur

    ##cost function

    return efuel, eprod


def HX():
    q_hx = h[0] - h[1]
    q_hx = h[4] - h[3]
    dt1 = t[0] - t[4]
    dt2 = t[1] - t[3]
    q_hx = U_hx * A_hx * lmtd(dt1, dt2)
    efuel = e1 - e2
    eprod = e5 - e4
    ##cost function

    return


def cooler():
    q_c = h[1] - h[2]
    efuel = q_c
    eprod = h[1] - h[2]
    ##cost function

    return efuel, eprod


def compressor():
    t[3] = (
        (t[2] + K)
        + ((t[2] + K) * comp_pratio ** (1 - 1 / gamma) - (t[2] + K)) / ncomp
        - K
    )
    w_comp = m * (h[3] - h[2])
    efuel = w_comp
    eprod = e4 - e3
    ##cost function

    return efuel, eprod


def heater():
    q_h = h[5] - h[4]
    efuel = q_h
    eprod = h[1] - h[2]
    ##cost function

    return efuel, eprod
