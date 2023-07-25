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


def old_Pressure_calculation(tur_pratio, comp_pratio):
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


def turbine(tin, pin, pout, ntur, m):
    turb_out = (
        Fluid(FluidsList.CarbonDioxide)
        .with_state(Input.temperature(tin), Input.pressure(pin))
        .expansion_to_pressure(pout, ntur)
    )
    turb_inlet = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.temperature(tin), Input.pressure(pin)
    )
    delta_h = turb_inlet.enthalpy - turb_out.enthalpy
    w_tur = delta_h * m
    return (
        turb_out.enthalpy,
        turb_out.entropy,
        turb_out.temperature,
        w_tur,
    )


def compressor(tin, pin, pout, ncomp, m):
    comp_out = (
        Fluid(FluidsList.CarbonDioxide)
        .with_state(Input.temperature(tin), Input.pressure(pin))
        .compression_to_pressure(pout, ncomp)
    )
    comp_inlet = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.temperature(tin), Input.pressure(pin)
    )
    delta_h = comp_out.enthalpy - comp_inlet.enthalpy
    w_comp = delta_h * m
    return (
        comp_out.enthalpy,
        comp_out.entropy,
        comp_out.temperature,
        w_comp,
    )


def cooler(tin, pin, tout, pdrop, m):
    cooler_out = (
        Fluid(FluidsList.CarbonDioxide)
        .with_state(Input.temperature(tin), Input.pressure(pin))
        .cooling_to_temperature(tout, pdrop)
    )
    cooler_inlet = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.temperature(tin), Input.pressure(pin)
    )
    delta_h = cooler_inlet.enthalpy - cooler_out.enthalpy
    q_cooler = delta_h * m
    return (
        cooler_out.enthalpy,
        cooler_out.entropy,
        q_cooler,
    )


def heater(tin, pin, tout, pdrop, m):
    heater_out = (
        Fluid(FluidsList.CarbonDioxide)
        .with_state(Input.temperature(tin), Input.pressure(pin))
        .heating_to_temperature(tout, pdrop)
    )
    heater_inlet = Fluid(FluidsList.CarbonDioxide).with_state(
        Input.temperature(tin), Input.pressure(pin)
    )
    delta_h = heater_out.enthalpy - heater_inlet.enthalpy
    q_heater = delta_h * m
    return (
        heater_out.enthalpy,
        heater_out.entropy,
        q_heater,
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

    try:
        fg_tout = opt.newton(objective, T0 + K)
    except:
        breakpoint()
    return fg_tout


##Heat exchanger hot and cold side determination needs to be implemented
def HX_calculation(Thotin, photin, hhotin, tcoldin, pcoldin, hcoldin, dt, hx_pdrop, m):
    try:
        hotside_outlet = (
            Fluid(FluidsList.CarbonDioxide)
            .with_state(Input.temperature(Thotin), Input.pressure(photin))
            .cooling_to_temperature(tcoldin + dt, hx_pdrop)
        )

        dh = hhotin - hotside_outlet.enthalpy
        q_hx = dh * m
        coldside_outlet = (
            Fluid(FluidsList.CarbonDioxide)
            .with_state(Input.temperature(tcoldin), Input.pressure(pcoldin))
            .heating_to_enthalpy(hcoldin + dh, hx_pdrop)
        )
        if Thotin - coldside_outlet.temperature < dt:
            raise Exception
    except:
        try:
            coldside_outlet = (
                Fluid(FluidsList.CarbonDioxide)
                .with_state(Input.temperature(tcoldin), Input.pressure(pcoldin))
                .heating_to_temperature(Thotin - dt, hx_pdrop)
            )

            dh = coldside_outlet.enthalpy - hcoldin

            hotside_outlet = (
                Fluid(FluidsList.CarbonDioxide)
                .with_state(Input.temperature(Thotin), Input.pressure(photin))
                .cooling_to_enthalpy(hhotin - dh, hx_pdrop)
            )
            q_hx = dh * m
        except:
            return (0, 0, 0, 0, 0, 0, 0)
    return (
        hotside_outlet.temperature,
        hotside_outlet.enthalpy,
        hotside_outlet.entropy,
        coldside_outlet.temperature,
        coldside_outlet.enthalpy,
        coldside_outlet.entropy,
        q_hx,
    )


def cw_Tout(q_cooler):
    m_cw = 200  # kg/s
    cw = Fluid(FluidsList.Water).with_state(
        Input.temperature(19), Input.pressure(101325)
    )
    cw_outlet = cw.heating_to_enthalpy(cw.enthalpy + q_cooler / m_cw, 0)
    return cw_outlet.temperature


T0 = 15
P0 = 101325
K = 273.15
h0, s0 = enthalpy_entropy(T0, P0)
h0_fg, s0_fg = h_s_fg(T0, P0)
hin_fg, sin_fg = h_s_fg(539.76, 101325)
exergy_new = hin_fg - h0_fg - (T0 + K) * (sin_fg - s0_fg)


def NG_exergy():
    """
    Fuel exergy calculation with 100% methane assumption
    """
    methane = Fluid(FluidsList.Methane).with_state(
        Input.temperature(25), Input.pressure(18.2e5)
    )
    m0 = Fluid(FluidsList.Methane).with_state(
        Input.temperature(25), Input.pressure(101325)
    )
    Pexergy = methane.enthalpy - m0.enthalpy - (T0 + K) * (methane.entropy - m0.entropy)
    Cexergy = 824.348 * 1.26 / 16.043 * 1e6
    return Pexergy + Cexergy


def decision_variable_placement(x, enumerated_equipment, pressures, temperatures):
    approach_temp = 1
    split_ratio = 1
    hx_token = 1
    for index, equip in enumerated_equipment:
        if equip == 1:
            pressures[index] = x[index]
        if equip == 2:
            temperatures[index] = x[index]
        if equip == 3:
            pressures[index] = x[index]
        if equip == 4:
            temperatures[index] = x[index]
        if equip == 5:
            if hx_token == 1:
                approach_temp = x[index]
                hx_token += -1
            else:
                pass
        if equip == 6:
            split_ratio = x[index]
    return (pressures, temperatures, approach_temp, split_ratio)


def Pressure_calculation(Pressures, equipment, cooler_pdrop, heater_pdrop, hx_pdrop):
    while Pressures.prod() == 0:
        for i in range(len(Pressures)):
            if Pressures[i] != 0:
                if i == len(Pressures) - 1:
                    if equipment[0] == 2:
                        Pressures[0] = Pressures[i] - cooler_pdrop
                    if equipment[0] == 4:
                        Pressures[0] = Pressures[i] - heater_pdrop
                    if equipment[0] == 5:
                        Pressures[0] = Pressures[i] - hx_pdrop

                else:
                    if equipment[i + 1] == 2:
                        Pressures[i + 1] = Pressures[i] - cooler_pdrop
                    if equipment[i + 1] == 4:
                        Pressures[i + 1] = Pressures[i] - heater_pdrop
                    if equipment[i + 1] == 5:
                        Pressures[i + 1] = Pressures[i] - hx_pdrop
    return Pressures


def tur_comp_pratio(enumerated_equipment, Pressures):
    for index, equip in enumerated_equipment:
        if equip == 1:
            if index != 0:
                tur_pratio = Pressures[index - 1] / Pressures[index]
            else:
                tur_pratio = Pressures[-1] / Pressures[index]
        if equip == 3:
            if index != 0:
                comp_pratio = Pressures[index] / Pressures[index - 1]
            else:
                comp_pratio = Pressures[index] / Pressures[-1]
    return (tur_pratio, comp_pratio)


def turbine_compressor_calculation(
    Temperatures,
    Pressures,
    enthalpies,
    entropies,
    w_tur,
    w_comp,
    equipment,
    ntur,
    ncomp,
    m,
):
    for i in range(len(Temperatures)):
        if Temperatures[i] != 0:
            if i == len(Temperatures) - 1:
                if equipment[0] == 1:
                    (
                        enthalpies[0],
                        entropies[0],
                        Temperatures[0],
                        w_tur[0],
                    ) = turbine(Temperatures[i], Pressures[i], Pressures[0], ntur, m)
                if equipment[0] == 3:
                    (
                        enthalpies[0],
                        entropies[0],
                        Temperatures[0],
                        w_comp[0],
                    ) = compressor(
                        Temperatures[i], Pressures[i], Pressures[0], ncomp, m
                    )

            else:
                if equipment[i + 1] == 1:
                    (
                        enthalpies[i + 1],
                        entropies[i + 1],
                        Temperatures[i + 1],
                        w_tur[i + 1],
                    ) = turbine(
                        Temperatures[i], Pressures[i], Pressures[i + 1], ntur, m
                    )
                if equipment[i + 1] == 3:
                    (
                        enthalpies[i + 1],
                        entropies[i + 1],
                        Temperatures[i + 1],
                        w_comp[i + 1],
                    ) = compressor(
                        Temperatures[i], Pressures[i], Pressures[i + 1], ncomp, m
                    )
    return (Temperatures, enthalpies, entropies, w_tur, w_comp)


def cooler_calculation(
    enumerated_equipment,
    Temperatures,
    Pressures,
    enthalpies,
    entropies,
    q_cooler,
    cooler_pdrop,
    m,
):
    cooler_position = [i for i, j in enumerated_equipment if j == 2]
    for i in cooler_position:
        (
            enthalpies[i],
            entropies[i],
            q_cooler[i],
        ) = cooler(
            Temperatures[i - 1], Pressures[i - 1], Temperatures[i], cooler_pdrop, m
        )
    return (enthalpies, entropies, q_cooler)


def heater_calculation(
    enumerated_equipment,
    Temperatures,
    Pressures,
    enthalpies,
    entropies,
    q_heater,
    heater_pdrop,
    m,
):
    heater_position = [i for i, j in enumerated_equipment if j == 4]
    for i in heater_position:
        (
            enthalpies[i],
            entropies[i],
            q_heater[i],
        ) = heater(
            Temperatures[i - 1], Pressures[i - 1], Temperatures[i], heater_pdrop, m
        )
    return (enthalpies, entropies, q_heater)


def hx_side_selection(hx_position, Temperatures):
    if hx_position[0] != 0 and hx_position[1] != 0:
        if Temperatures[hx_position[0] - 1] > Temperatures[hx_position[1] - 1]:
            hotside_index = hx_position[0]
            coldside_index = hx_position[1]
        else:
            hotside_index = hx_position[1]
            coldside_index = hx_position[0]
    if hx_position[0] == 0:
        if Temperatures[-1] > Temperatures[hx_position[1] - 1]:
            hotside_index = -1
            coldside_index = hx_position[1]
        else:
            hotside_index = hx_position[1]
            coldside_index = -1
    if hx_position[1] == 0:
        if Temperatures[hx_position[0] - 1] > Temperatures[-1]:
            hotside_index = hx_position[0]
            coldside_index = -1
        else:
            hotside_index = -1
            coldside_index = hx_position[0]
    return (hotside_index, coldside_index)
