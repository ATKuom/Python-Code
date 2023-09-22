from pyfluids import Fluid, FluidsList, Input, Mixture
import numpy as np
import scipy.optimize as opt
import CoolProp.CoolProp as CP

T0 = 15
P0 = 101325
K = 273.15


def lmtd(dt1, dt2):
    if dt1 == dt2:
        return dt1
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


def fg_calculation(fg_m, q_heater, fg_tin=539.76):
    fg_in_h = CP.PropsSI(
        "H",
        "P|gas",
        101325,
        "T",
        fg_tin + K,
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
        fg_tout = opt.newton(objective, 100)
    except:
        # print("unfeasible fg_exit temperature")
        fg_tout = 0
    return fg_tout


def HX_calculation(
    Thotin,
    photin,
    hhotin,
    tcoldin,
    pcoldin,
    hcoldin,
    dt,
    hx_pdrop,
    m_hotside,
    m_coldside,
):
    try:
        hotside_outlet = (
            Fluid(FluidsList.CarbonDioxide)
            .with_state(Input.temperature(Thotin), Input.pressure(photin))
            .cooling_to_temperature(tcoldin + dt, hx_pdrop)
        )

        dh_hotside = hhotin - hotside_outlet.enthalpy
        q_hotside = dh_hotside * m_hotside
        dh_coldside = q_hotside / m_coldside

        coldside_outlet = (
            Fluid(FluidsList.CarbonDioxide)
            .with_state(Input.temperature(tcoldin), Input.pressure(pcoldin))
            .heating_to_enthalpy(hcoldin + dh_coldside, hx_pdrop)
        )
        q_hx = q_hotside

        if Thotin - coldside_outlet.temperature < dt:
            raise Exception
    except:
        # try:
        coldside_outlet = (
            Fluid(FluidsList.CarbonDioxide)
            .with_state(Input.temperature(tcoldin), Input.pressure(pcoldin))
            .heating_to_temperature(Thotin - dt, hx_pdrop)
        )
        dh_coldside = coldside_outlet.enthalpy - hcoldin
        q_coldside = dh_coldside * m_coldside
        dh_hotside = q_coldside / m_hotside
        hotside_outlet = (
            Fluid(FluidsList.CarbonDioxide)
            .with_state(Input.temperature(Thotin), Input.pressure(photin))
            .cooling_to_enthalpy(hhotin - dh_hotside, hx_pdrop)
        )
        q_hx = q_coldside
    # except:
    #     return (0, 0, 0, 0, 0, 0, 0)
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


h0, s0 = enthalpy_entropy(T0, P0)
h0_fg, s0_fg = h_s_fg(T0, P0)
hin_fg, sin_fg = h_s_fg(539.76, 101325)
fg_m = 68.75
FGINLETEXERGY = fg_m * (hin_fg - h0_fg - (T0 + K) * (sin_fg - s0_fg)) + 0.5e6


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


def decision_variable_placement(x, enumerated_equipment):
    approach_temp = 1
    split_ratio = 1
    hx_token = 1
    Temperatures = np.zeros(len(enumerated_equipment))
    Pressures = np.zeros(len(enumerated_equipment))
    mass_flow = np.ones(len(enumerated_equipment)) * x[-1]
    for index, equip in enumerated_equipment:
        if equip == 1:
            Pressures[index] = x[index]
        elif equip == 2:
            Temperatures[index] = x[index]
        elif equip == 3:
            Pressures[index] = x[index]
        elif equip == 4:
            Temperatures[index] = x[index]
        elif equip == 5 and hx_token == 1:
            approach_temp = x[index]
            hx_token = 0
        elif equip == 5 and hx_token == 0:
            x[index] = approach_temp
        elif equip == 9:
            split_ratio = x[index]
            branching_start = index
            branching_end1, branching_end2 = np.where(
                7 == np.array(enumerated_equipment)[:, 1]
            )[0]
            for i in range(branching_start, branching_end1):
                mass_flow[i] = mass_flow[i] * split_ratio
            for i in range(branching_end1, branching_end2):
                mass_flow[i] = mass_flow[i] * (1 - split_ratio)

    return (Pressures, Temperatures, approach_temp, split_ratio, mass_flow)


def Pressure_calculation(
    Pressures, equipment, cooler_pdrop, heater_pdrop, hx_pdrop, splitter=False
):
    if splitter == True:
        equipment = np.asarray(equipment)
        mixer1, mixer2 = np.where(7 == equipment)[0]
    while Pressures.prod() == 0:
        if (
            splitter == True
            and Pressures[mixer1 - 1] != 0
            and Pressures[mixer2 - 1] != 0
        ):
            Pressures[mixer2] = min(Pressures[mixer1 - 1], Pressures[mixer2 - 1])

        if Pressures.prod() == 0:
            for i in range(len(Pressures)):
                if Pressures[i] != 0:
                    if i == len(Pressures) - 1:
                        if equipment[0] == 2:
                            Pressures[0] = Pressures[i] - cooler_pdrop
                        if equipment[0] == 4:
                            Pressures[0] = Pressures[i] - heater_pdrop
                        if equipment[0] == 5:
                            Pressures[0] = Pressures[i] - hx_pdrop
                        if equipment[0] == 9:
                            Pressures[0] = Pressures[i]
                            Pressures[mixer1] = Pressures[i]

                    else:
                        if equipment[i + 1] == 2:
                            Pressures[i + 1] = Pressures[i] - cooler_pdrop
                        if equipment[i + 1] == 4:
                            Pressures[i + 1] = Pressures[i] - heater_pdrop
                        if equipment[i + 1] == 5:
                            Pressures[i + 1] = Pressures[i] - hx_pdrop
                        if equipment[i + 1] == 9:
                            Pressures[i + 1] = Pressures[i]
                            Pressures[mixer1] = Pressures[i]
        breakpoint()
    return Pressures


def tur_comp_pratio(enumerated_equipment, Pressures):
    tur_pratio = np.ones(len(enumerated_equipment)) * 1.0001
    comp_pratio = np.ones(len(enumerated_equipment)) * 1.0001
    for index, equip in enumerated_equipment:
        if equip == 1:
            tur_pratio[index] = Pressures[index - 1] / Pressures[index]
        if equip == 3:
            comp_pratio[index] = Pressures[index] / Pressures[index - 1]
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
    mass_flow,
):
    for i in range(len(Temperatures)):
        if Temperatures[i] != 0:
            if i == len(Temperatures) - 1:
                if Temperatures[0] == 0:
                    if equipment[0] == 1:
                        (
                            enthalpies[0],
                            entropies[0],
                            Temperatures[0],
                            w_tur[0],
                        ) = turbine(
                            Temperatures[i],
                            Pressures[i],
                            Pressures[0],
                            ntur,
                            mass_flow[0],
                        )
                    elif equipment[0] == 3:
                        (
                            enthalpies[0],
                            entropies[0],
                            Temperatures[0],
                            w_comp[0],
                        ) = compressor(
                            Temperatures[i],
                            Pressures[i],
                            Pressures[0],
                            ncomp,
                            mass_flow[0],
                        )
                    elif equipment[0] == 9:
                        enthalpies[0] = enthalpies[i]
                        entropies[0] = entropies[i]
                        Temperatures[0] = Temperatures[i]

            else:
                if Temperatures[i + 1] == 0:
                    if equipment[i + 1] == 1:
                        (
                            enthalpies[i + 1],
                            entropies[i + 1],
                            Temperatures[i + 1],
                            w_tur[i + 1],
                        ) = turbine(
                            Temperatures[i],
                            Pressures[i],
                            Pressures[i + 1],
                            ntur,
                            mass_flow[i + 1],
                        )
                    elif equipment[i + 1] == 3:
                        (
                            enthalpies[i + 1],
                            entropies[i + 1],
                            Temperatures[i + 1],
                            w_comp[i + 1],
                        ) = compressor(
                            Temperatures[i],
                            Pressures[i],
                            Pressures[i + 1],
                            ncomp,
                            mass_flow[i + 1],
                        )

    return (Temperatures, enthalpies, entropies, w_tur, w_comp)


def cooler_calculation(
    cooler_position,
    Temperatures,
    Pressures,
    enthalpies,
    entropies,
    q_cooler,
    cooler_pdrop,
    mass_flow,
):
    for i in cooler_position:
        (
            enthalpies[i],
            entropies[i],
            q_cooler[i],
        ) = cooler(
            Temperatures[i - 1],
            Pressures[i - 1],
            Temperatures[i],
            cooler_pdrop,
            mass_flow[i],
        )
    return (enthalpies, entropies, q_cooler)


def heater_calculation(
    heater_position,
    Temperatures,
    Pressures,
    enthalpies,
    entropies,
    q_heater,
    heater_pdrop,
    mass_flow,
):
    for i in heater_position:
        (
            enthalpies[i],
            entropies[i],
            q_heater[i],
        ) = heater(
            Temperatures[i - 1],
            Pressures[i - 1],
            Temperatures[i],
            heater_pdrop,
            mass_flow[i],
        )
    return (enthalpies, entropies, q_heater)


def hx_side_selection(hx_position, Temperatures):
    if Temperatures[hx_position[0] - 1] >= Temperatures[hx_position[1] - 1]:
        hotside_index = hx_position[0]
        coldside_index = hx_position[1]
    else:
        hotside_index = hx_position[1]
        coldside_index = hx_position[0]
    return (hotside_index, coldside_index)


def splitter_mixer_calc(
    Temperatures, Pressures, enthalpies, entropies, mass_flow, equipment
):
    equipment = np.asarray(equipment)
    splitter = np.where(equipment == 9)[0]
    mixer1, mixer2 = np.where(equipment == 7)[0]
    if Temperatures[splitter] != 0 and Temperatures[mixer1] == 0:
        Temperatures[mixer1] = Temperatures[splitter]
        enthalpies[mixer1] = enthalpies[splitter]
        entropies[mixer1] = entropies[splitter]
    if Pressures[mixer1 - 1] == Pressures[mixer2 - 1] and Temperatures[mixer2] == 0:
        inlet1 = Fluid(FluidsList.CarbonDioxide).with_state(
            Input.temperature(Temperatures[mixer1 - 1]),
            Input.pressure(Pressures[mixer1 - 1]),
        )
        inlet2 = Fluid(FluidsList.CarbonDioxide).with_state(
            Input.temperature(Temperatures[mixer2 - 1]),
            Input.pressure(Pressures[mixer2 - 1]),
        )
        outlet = Fluid(FluidsList.CarbonDioxide).mixing(
            mass_flow[mixer1], inlet1, mass_flow[mixer2], inlet2
        )
        Temperatures[mixer2] = outlet.temperature
        enthalpies[mixer2] = outlet.enthalpy
        entropies[mixer2] = outlet.entropy
        return (Temperatures, enthalpies, entropies)
    if (
        Temperatures[mixer1 - 1] != 0
        and Temperatures[mixer2 - 1] != 0
        and Temperatures[mixer2] == 0
    ):
        if Pressures[mixer1 - 1] > Pressures[mixer2 - 1]:
            hp_index = mixer1 - 1
            lp_index = mixer2 - 1
        elif Pressures[mixer1 - 1] < Pressures[mixer2 - 1]:
            hp_index = mixer2 - 1
            lp_index = mixer1 - 1

        hp_inlet = (
            Fluid(FluidsList.CarbonDioxide)
            .with_state(
                Input.temperature(Temperatures[hp_index]),
                Input.pressure(Pressures[hp_index]),
            )
            .isenthalpic_expansion_to_pressure(Pressures[lp_index])
        )
        lp_inlet = Fluid(FluidsList.CarbonDioxide).with_state(
            Input.temperature(Temperatures[lp_index]),
            Input.pressure(Pressures[lp_index]),
        )
        outlet = Fluid(FluidsList.CarbonDioxide).mixing(
            mass_flow[hp_index], hp_inlet, mass_flow[lp_index], lp_inlet
        )
        Temperatures[mixer2] = outlet.temperature
        enthalpies[mixer2] = outlet.enthalpy
        entropies[mixer2] = outlet.entropy
    return (Temperatures, enthalpies, entropies)


def bound_creation(layout):
    units = layout[1:-1]
    # print(units)
    x = []
    splitter = False
    equipment = np.zeros(len(units)).tolist()
    bounds = list(range(len(units)))
    hx_token = 1
    for i in range(len(units)):
        unit_type = np.where(units[i] == 1)[0][0]
        if unit_type == 1:
            equipment[i] = 1
            bounds[i] = (74e5, 300e5)
        elif unit_type == 2:
            equipment[i] = 2
            bounds[i] = (32, 38)
        elif unit_type == 3:
            equipment[i] = 3
            bounds[i] = (74e5, 300e5)
        elif unit_type == 4:
            equipment[i] = 4
            bounds[i] = (180, 530)
        elif unit_type == 5:
            equipment[i] = 5
            if hx_token == 1:
                bounds[i] = (4, 11)
                # hx_token = 0
            else:
                bounds[i] = (0, 0)
        elif unit_type == 7:
            equipment[i] = 7
            bounds[i] = (0, 0)
        elif unit_type == 9:
            equipment[i] = 9
            bounds[i] = (0.01, 0.99)
            splitter = True
            branch_start = i
    if splitter == True:
        equipment = np.roll(equipment, -branch_start, axis=0).tolist()
        bounds = np.roll(bounds, -branch_start, axis=0).tolist()
    bounds.append((50, 160))
    print(equipment)
    # print(bounds)
    return (equipment, bounds, x, splitter)
