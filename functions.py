from pyfluids import Fluid, FluidsList, Input
import numpy as np


##Specific heat calculation works fine with DT similar to estimate h2-h1
##However, h2 =/= cp*t2
def lmtd(dthin, dt2):
    return (dthin - dt2) / np.log(dthin / dt2)


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


def specific_heat(T, P):
    substance = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.pressure(P), Input.temperature(T)
    )
    return substance.specific_heat


T0 = 15
P0 = 101325
K = 273.15
(h0, s0) = enthalpy_entropy(T0, P0)


def temperature(h, P):
    """
    Takes the the temperature and pressure of a CO2 stream and gives enthalpy, entropy and specific heat values at that temperature
    Temperature input is C, Pressure input is pa
    Return: Enthalpy (J/kg), Entropy (J/kgK), Specific Heat (J/kgK)

    """
    substance = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.enthalpy(h), Input.pressure(P)
    )
    return substance.temperature


def pinch_calculation(thin, tcin, phout, pcout, m, hhin, hcin):
    t2 = [t2 for t2 in range(int(tcin) + 5, int(thin))]
    if len(t2) == 0:
        return float(1e6)
    h2 = list()
    for temp in t2:
        a, _ = enthalpy_entropy(temp, phout)
        h2.append(a)
    h2 = np.asarray(h2)
    q_hx1 = m * hhin - m * h2
    t5 = [t2 for t2 in range(int(tcin), int(thin) - 5)]
    if len(t5) == 0:
        return float(1e6)
    h5 = list()
    for temp in t5:
        a, _ = enthalpy_entropy(temp, pcout)
        h5.append(a)
    h5 = np.asarray(h5)
    q_hx2 = m * h5 - m * hcin
    q_hx = q_hx1 - q_hx2
    index = np.where(q_hx[:-1] * q_hx[1:] < 0)[0]
    t2 = t2[index[0]]
    t5 = t5[index[0]]
    return (t2, t5)
