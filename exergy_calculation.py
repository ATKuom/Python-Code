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
K = 273.15
(h0, s0, c0) = enthalpy_entropy(T0, P0)

if __name__ == "__main__":
    t = 15
    p = 18.20e5
    (hl, sl, c1) = enthalpy_entropy(t, p)
    Exergy_stream = hl - h0 - (T0 + K) * (sl - s0)
    print(
        hl / 1e6,
        sl / 1e6,
        1.26 * Exergy_stream / 1e6,
    )
