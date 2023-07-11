from pyfluids import Fluid, FluidsList, Input, Mixture
import numpy as np


##Specific heat calculation works fine with DT similar to estimate h2-h1
##However, h2 =/= cp*T_hotout
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


def specificheat(T, P):
    substance = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.pressure(P), Input.temperature(T)
    )
    return substance.specific_heat


def gammacalc(T, P):
    substance = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.pressure(P), Input.temperature(T)
    )
    R = 189
    gamma = substance.specific_heat / (substance.specific_heat - R)
    return gamma


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


def pinch_calculation(
    T_hin, H_hotin, T_coldin, H_coldin, P_hotout, P_coldout, m, pinch_temp
):
    list_T_hotout = [
        T_hotout for T_hotout in range(int(T_coldin) + int(pinch_temp), int(T_hin))
    ]
    if len(list_T_hotout) == 0:
        return (0, 0)
    h2 = list()
    for temp in list_T_hotout:
        a, _ = enthalpy_entropy(temp, P_hotout)
        h2.append(a)
    h2 = np.asarray(h2)
    q_hx1 = m * H_hotin - m * h2
    list_T_coldout = [
        T_coldout for T_coldout in range(int(T_coldin), int(T_hin) - int(pinch_temp))
    ]
    if len(list_T_coldout) == 0:
        return (0, 0)
    h5 = list()
    for temp in list_T_coldout:
        a, _ = enthalpy_entropy(temp, P_coldout)
        h5.append(a)
    h5 = np.asarray(h5)
    q_hx2 = m * h5 - m * H_coldin
    q_hx = q_hx1 - q_hx2
    index = np.where(q_hx[:-1] * q_hx[1:] < 0)[0]
    if len(index) == 0:
        return (0, 0)
    T_hotout = list_T_hotout[index[0]]
    T_coldout = list_T_coldout[index[0]]
    return (T_hotout, T_coldout)


def Pressure_calculation(tur_pratio, comp_pratio):
    # [p1,p2,p3,p4,p5,p6]
    pres = np.array(
        [
            [1, 0, 0, 0, 0, -1 / tur_pratio],
            [1, -1, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0],
            [0, 0, comp_pratio, -1, 0, 0],
            [0, 0, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, -1],
        ]
    )
    dp = np.array([0, 1e5, 0.5e5, 0, 1e5, 1e5]).reshape(-1, 1)
    try:
        pressures = np.linalg.solve(pres, dp)
    except:
        # print("singular matrix", tur_pratio, comp_pratio)
        return [0, 0, 0, 0, 0, 0]
    p1 = pressures.item(0)
    if p1 < 0:
        # print("negative Pressure")
        return [0, 0, 0, 0, 0, 0]
    # ub = 300e5 / max(pressures)
    # lb = 74e5 / max(pressures)
    # pres_coeff = np.random.uniform(lb, ub)
    # pressures = pres_coeff * pressures
    p1 = pressures.item(0)
    p2 = pressures.item(1)
    p3 = pressures.item(2)
    p4 = pressures.item(3)
    p5 = pressures.item(4)
    p6 = pressures.item(5)
    return (p1, p2, p3, p4, p5, p6)


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


def exhaust():
    exhaust_mass_flow = 68.75
    exhaust_inlet_T = 539.8
    exhaust_inlet_P = 10e5
    flue_gas = Mixture(
        [
            FluidsList.Nitrogen,
            FluidsList.Oxygen,
            FluidsList.CarbonDioxide,
            FluidsList.Water,
        ],
        [75.3, 15.53, 05.05, 04.12],
    )
    exhaust_inlet = flue_gas.with_state(
        Input.temperature(exhaust_inlet_T), Input.pressure(exhaust_inlet_P)
    )
    exhaust_inlet_h = exhaust_inlet.enthalpy
    print(exhaust_inlet_h)


def turbine(tin, pin, pout, ntur):
    turb_out = (
        Fluid(FluidsList.CarbonDioxide)
        .with_state(Input.temperature(tin), Input.pressure(pin))
        .expansion_to_pressure(pout, ntur)
    )
    turb_inlet = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.temperature(tin), Input.pressure(pin)
    )
    delta_h = turb_inlet.enthalpy - turb_out.enthalpy
    return (
        turb_out.enthalpy,
        turb_out.entropy,
        turb_out.temperature,
        turb_out.pressure,
        delta_h,
    )


def compressor(tin, pin, pout, ncomp):
    comp_out = (
        Fluid(FluidsList.CarbonDioxide)
        .with_state(Input.temperature(tin), Input.pressure(pin))
        .compression_to_pressure(pout, ncomp)
    )
    comp_inlet = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.temperature(tin), Input.pressure(pin)
    )
    delta_h = comp_out.enthalpy - comp_inlet.enthalpy
    return (
        comp_out.enthalpy,
        comp_out.entropy,
        comp_out.temperature,
        comp_out.pressure,
        delta_h,
    )


def cooler(tin, pin, tout, pdrop):
    cooler_out = (
        Fluid(FluidsList.CarbonDioxide)
        .with_state(Input.temperature(tin), Input.pressure(pin))
        .cooling_to_temperature(tout, pdrop)
    )
    cooler_inlet = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.temperature(tin), Input.pressure(pin)
    )
    delta_h = cooler_inlet.enthalpy - cooler_out.enthalpy
    return (
        cooler_out.enthalpy,
        cooler_out.entropy,
        cooler_out.temperature,
        cooler_out.pressure,
        delta_h,
    )


def heater(tin, pin, tout, pdrop):
    heater_out = (
        Fluid(FluidsList.CarbonDioxide)
        .with_state(Input.temperature(tin), Input.pressure(pin))
        .heating_to_temperature(tout, pdrop)
    )
    heater_inlet = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.temperature(tin), Input.pressure(pin)
    )
    delta_h = heater_out.enthalpy - heater_inlet.enthalpy
    return (
        heater_out.enthalpy,
        heater_out.entropy,
        heater_out.temperature,
        heater_out.pressure,
        delta_h,
    )


# T0 = 15
exhaust()
h0_fg, s0_fg, cp0_fg = h_s_fg(T0, P0)
h_fg, s_fg, cp_fg = h_s_fg(539, 1.01e5)
print(h_s_fg(539, 1.01e5))
