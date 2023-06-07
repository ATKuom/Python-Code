from pyfluids import Fluid, FluidsList, Input


def enthalpy_entropy(T, P):
    """
    Takes the the temperature and pressure of a CO2 stream and gives enthalpy, entropy and specific heat values at that temperature
    Temperature input is C, Pressure input is pa
    Return: Enthalpy (J/kg), Entropy (J/kgK), Specific Heat (J/kgK)

    """
    substance = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.pressure(P), Input.temperature(T)
    )
    return (substance.enthalpy, substance.entropy, substance.specific_heat)


T0 = 15
P0 = 101325
t = 350
p = 2e6
(h0, s0, c0) = enthalpy_entropy(T0, P0)
(hl, sl, c1) = enthalpy_entropy(t, p)
Exergy_stream = hl - h0 - T0 * (sl - s0)
# print(c1, c0)
print(hl, c1 * (t + 273.15))
# Exergy_with_cp = c1 * (T + 273.15) - c0 * (T0 + 273.15) - (T0 + 273.15) * (sl - s0)
# print(Exergy_stream, Exergy_with_cp)
