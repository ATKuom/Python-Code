from pyfluids import Fluid, FluidsList, Input
import numpy as np


def lmtd(dt1, dt2):
    return (dt1 - dt2) / np.log(dt1 / dt2)


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


print(enthalpy_entropy(560, 250e6))
print(temperature(1091129.2381446492, 250e6))
