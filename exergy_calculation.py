from pyfluids import Fluid, FluidsList, Input, InputHumidAir, HumidAir


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

    # def h_s_fg(T, P):
    nitrogen = Fluid(FluidsList.Nitrogen).with_state(
        Input.pressure(P), Input.temperature(T)
    )
    oxygen = Fluid(FluidsList.Oxygen).with_state(
        Input.pressure(P), Input.temperature(T)
    )
    h = nitrogen.enthalpy * 0.77 + oxygen.enthalpy * 0.23
    s = nitrogen.entropy * 0.77 + oxygen.entropy * 0.23
    return (h, s)


def h_s_fg(T, P):
    nitrogen = Fluid(FluidsList.Nitrogen).with_state(
        Input.pressure(P), Input.temperature(T)
    )
    oxygen = Fluid(FluidsList.Oxygen).with_state(
        Input.pressure(P), Input.temperature(T)
    )
    water = Fluid(FluidsList.Water).with_state(Input.pressure(P), Input.temperature(T))
    carbon_dioxide = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.pressure(P), Input.temperature(T)
    )
    h = (
        nitrogen.enthalpy * 0.753
        + oxygen.enthalpy * 0.1553
        + carbon_dioxide.enthalpy * 0.0505
        + water.enthalpy * 0.0412
    )
    s = (
        nitrogen.entropy * 0.753
        + oxygen.entropy * 0.1553
        + carbon_dioxide.entropy * 0.0505
        + water.entropy * 0.0412
    )
    cp = (
        nitrogen.specific_heat * 0.753
        + oxygen.specific_heat * 0.1553
        + carbon_dioxide.specific_heat * 0.0505
        + water.specific_heat * 0.0412
    )
    return (h, s, cp)


T0 = 15
P0 = 101325
K = 273.15
(h0, s0) = enthalpy_entropy(T0, P0)

if __name__ == "__main__":
    t = 539.76  # 217.99
    p = 1.01e5
    (h1, s1) = enthalpy_entropy(t, p)
    # Exergy_stream = h1 - h0 - (T0 + K) * (s1 - s0)
    hfg0, sfg0, cp0 = h_s_fg(T0, P0)
    hfg1, sfg1, cp1 = h_s_fg(t, p)
    Exergy_stream = hfg1 - hfg0 - (T0 + K) * (sfg1 - sfg0)
    print(
        h1 / 1e3,
        s1 / 1e3,
        # 1.26 * Exergy_stream / 1e6,
        hfg1 / 1e3,
        sfg1 / 1e3,
        68.75 * Exergy_stream / 1e6,
        cp1,
    )
