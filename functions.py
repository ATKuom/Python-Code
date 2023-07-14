from pyfluids import Fluid, FluidsList, Input, Mixture
import numpy as np
import scipy.optimize as opt
import CoolProp.CoolProp as CP


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


def h_s_fg(t, p):
    h, s = CP.PropsSI(
        ["H", "S"],
        "P|gas",
        p,
        "T",
        t + K,
        "Nitrogen[0.7643]&Oxygen[0.1382]&Water[0.0650]&CarbonDioxide[0.0325]",
    )
    return (h, s)


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


def fg_calculation(fg_m, q_heater):
    fg_in_h = CP.PropsSI(
        "H",
        "P|gas",
        101325,
        "T",
        539.76 + K,
        "Nitrogen[0.7643]&Oxygen[0.1382]&Water[0.0650]&CarbonDioxide[0.0325]",
    )

    def objective(T):
        fg_out_h = CP.PropsSI(
            "H",
            "P|gas",
            101325,
            "T",
            T + K,
            "Nitrogen[0.7643]&Oxygen[0.1382]&Water[0.0650]&CarbonDioxide[0.0325]",
        )
        return fg_m * (fg_in_h - fg_out_h) - q_heater

    fg_tout = opt.newton(objective, T0 + K)
    return fg_tout


def HX_calculation(t1, p1, h1, t4, p4, h4, dt, hx_pdrop):
    try:
        hotside_outlet = (
            Fluid(FluidsList.CarbonDioxide)
            .with_state(Input.temperature(t1), Input.pressure(p1))
            .cooling_to_temperature(t4 + dt, hx_pdrop)
        )

        dh = h1 - hotside_outlet.enthalpy

        coldside_outlet = (
            Fluid(FluidsList.CarbonDioxide)
            .with_state(Input.temperature(t4), Input.pressure(p4))
            .heating_to_enthalpy(h4 + dh, hx_pdrop)
        )
        if t1 - coldside_outlet.temperature < dt:
            raise Exception
    except:
        coldside_outlet = (
            Fluid(FluidsList.CarbonDioxide)
            .with_state(Input.temperature(t4), Input.pressure(p4))
            .heating_to_temperature(t1 - dt, hx_pdrop)
        )

        dh = coldside_outlet.enthalpy - h4

        hotside_outlet = (
            Fluid(FluidsList.CarbonDioxide)
            .with_state(Input.temperature(t1), Input.pressure(p1))
            .cooling_to_enthalpy(h1 - dh, hx_pdrop)
        )
    return (
        hotside_outlet.temperature,
        hotside_outlet.pressure,
        hotside_outlet.enthalpy,
        hotside_outlet.entropy,
        coldside_outlet.temperature,
        coldside_outlet.pressure,
        coldside_outlet.enthalpy,
        coldside_outlet.entropy,
        dh,
    )


def cw_Tout(q_cooler):
    m_cw = 200  # kg/s
    cw = Fluid(FluidsList.Water).with_state(
        Input.temperature(19), Input.pressure(101325)
    )
    cw_outlet = cw.heating_to_enthalpy(cw.enthalpy + q_cooler / m_cw, 0)
    return cw_outlet.temperature


print(cw_Tout(17.3e6))
T0 = 15
P0 = 101325
K = 273.15
h0, s0 = enthalpy_entropy(T0, P0)
h0_fg, s0_fg = h_s_fg(T0, P0)
hin_fg, sin_fg = h_s_fg(539.76, 101325)
exergy_new = hin_fg - h0_fg - (T0 + K) * (sin_fg - s0_fg)
