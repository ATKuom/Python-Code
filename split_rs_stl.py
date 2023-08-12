import numpy as np
import torch
import matplotlib.pyplot as plt
from econ import economics
from split_functions import (
    lmtd,
    turbine,
    compressor,
    cooler,
    heater,
    h_s_fg,
    fg_calculation,
    HX_calculation,
    cw_Tout,
    decision_variable_placement,
    Pressure_calculation,
    tur_comp_pratio,
    turbine_compressor_calculation,
    cooler_calculation,
    heater_calculation,
    hx_side_selection,
    h0_fg,
    s0_fg,
    hin_fg,
    sin_fg,
    h0,
    s0,
    T0,
    P0,
    K,
)


def results_analysis(x, equipment):
    ntur = 85  # turbine efficiency     2019 Nabil
    ncomp = 82  # compressor efficiency 2019 Nabil
    cw_temp = 19  # °C
    fg_tin = 539  # °C
    fg_m = 68.75  # kg/s
    cooler_pdrop = 1e5
    heater_pdrop = 0
    hx_pdrop = 0.5e5
    PENALTY_VALUE = float(1e6)

    enumerated_equipment = list(enumerate(equipment))
    Temperatures = np.zeros(len(equipment))
    Pressures = np.zeros(len(equipment))
    enthalpies = np.zeros(len(equipment))
    entropies = np.zeros(len(equipment))
    exergies = np.zeros(len(equipment))
    w_comp = np.zeros(len(equipment))
    cost_comp = np.zeros(len(equipment))
    comp_pratio = np.ones(len(equipment))
    w_tur = np.zeros(len(equipment))
    cost_tur = np.zeros(len(equipment))
    tur_pratio = np.ones(len(equipment))
    q_cooler = np.zeros(len(equipment))
    cost_cooler = np.zeros(len(equipment))
    dissipation = np.zeros(len(equipment))
    q_heater = np.zeros(len(equipment))
    cost_heater = np.zeros(len(equipment))
    e_fgin = np.zeros(len(equipment))
    e_fgout = np.zeros(len(equipment))
    q_hx = np.zeros(len(equipment))
    cost_hx = np.zeros(len(equipment))

    (
        Pressures,
        Temperatures,
        approach_temp,
        split_ratio,
        mass_flow,
    ) = decision_variable_placement(x, enumerated_equipment)

    # Pressure calculation splitter part is missing still
    Pressures = Pressure_calculation(
        Pressures, equipment, cooler_pdrop, heater_pdrop, hx_pdrop
    )
    # Turbine and Compressor pressure ratio calculation and checking
    tur_pratio, comp_pratio = tur_comp_pratio(enumerated_equipment, Pressures)

    if np.any(tur_pratio < 1) or np.any(comp_pratio < 1):
        print("Turbine or Compressor pressure ratio is less than 1")
        return PENALTY_VALUE

    while Temperatures.prod() == 0:
        (
            Temperatures,
            enthalpies,
            entropies,
            w_tur,
            w_comp,
        ) = turbine_compressor_calculation(
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
        )
        for work in w_tur:
            if work < 0:
                print("Turbine work is negative")
                return PENALTY_VALUE
        for work in w_comp:
            if work < 0:
                print("Compressor work is negative")
                return PENALTY_VALUE
        hx_position = [i for i, j in enumerated_equipment if j == 5]
        if (
            hx_position != []
            and Temperatures[hx_position[0] - 1] != 0
            and Temperatures[hx_position[1] - 1] != 0
        ):
            hotside_index, coldside_index = hx_side_selection(hx_position, Temperatures)
            if (
                Temperatures[hotside_index - 1]
                < Temperatures[coldside_index - 1] + approach_temp
            ):
                print("Infeasible HX")
                return PENALTY_VALUE
            try:
                (
                    Temperatures[hotside_index],
                    enthalpies[hotside_index],
                    entropies[hotside_index],
                    Temperatures[coldside_index],
                    enthalpies[coldside_index],
                    entropies[coldside_index],
                    q_hx[min(hotside_index, coldside_index)],
                ) = HX_calculation(
                    Temperatures[hotside_index - 1],
                    Pressures[hotside_index - 1],
                    enthalpies[hotside_index - 1],
                    Temperatures[coldside_index - 1],
                    Pressures[coldside_index - 1],
                    enthalpies[coldside_index - 1],
                    approach_temp,
                    hx_pdrop,
                    mass_flow[hotside_index],
                    mass_flow[coldside_index],
                )
            except:
                breakpoint()
    enthalpies, entropies, q_cooler = cooler_calculation(
        enumerated_equipment,
        Temperatures,
        Pressures,
        enthalpies,
        entropies,
        q_cooler,
        cooler_pdrop,
        mass_flow,
    )
    for work in q_cooler:
        if work < 0:
            print("Infeasible Cooler")
            return PENALTY_VALUE

    enthalpies, entropies, q_heater = heater_calculation(
        enumerated_equipment,
        Temperatures,
        Pressures,
        enthalpies,
        entropies,
        q_heater,
        heater_pdrop,
        mass_flow,
    )

    total_heat = sum(q_heater)
    fg_tout = fg_calculation(fg_m, total_heat)
    if fg_tout < 90:
        print("Too low stack temperature")
        return PENALTY_VALUE
    hout_fg, sout_fg = h_s_fg(fg_tout, P0)

    # Exergy Analysis
    for streams in range(len(exergies)):
        exergies[streams] = mass_flow[streams] * (
            enthalpies[streams] - h0 - (T0 + K) * (entropies[streams] - s0)
        )
    for i, work in enumerate(q_heater):
        e_fgin[i] = (work / total_heat) * (
            fg_m * (hin_fg - h0_fg - (T0 + K) * (sin_fg - s0_fg)) + 0.5e6
        )
        e_fgout[i] = (work / total_heat) * (
            fg_m * (hout_fg - h0_fg - (T0 + K) * (sout_fg - s0_fg)) + 0.5e6
        )

    # Economic Analysis
    for index, work in enumerate(w_tur):
        if work > 0:
            index = np.where(w_tur == work)[0][0]
            if Temperatures[index - 1] > 550:
                ft_tur = 1 + 1.137e-5 * (Temperatures[index - 1] - 550) ** 2
            else:
                ft_tur = 1
            cost_tur[index] = 406200 * ((work / 1e6) ** 0.8) * ft_tur
    for index, work in enumerate(w_comp):
        if work > 0:
            cost_comp[index] = 1230000 * (work / 1e6) ** 0.3992

    for index, work in enumerate(q_cooler):
        if work > 0:
            dt1_cooler = Temperatures[index] - cw_temp
            dt2_cooler = Temperatures[index - 1] - cw_Tout(work)
            if dt2_cooler < 0 or dt1_cooler < 0:
                return PENALTY_VALUE
            UA_cooler = (work / 1) / lmtd(dt1_cooler, dt2_cooler)  # W / °C
            if Temperatures[index - 1] > 550:
                ft_cooler = 1 + 0.02141 * (Temperatures[index - 1] - 550)
            else:
                ft_cooler = 1
            cost_cooler[index] = 49.45 * UA_cooler**0.7544 * ft_cooler  # $

    for index, work in enumerate(q_heater):
        if work > 0:
            fg_tout_i = fg_calculation(fg_m * work / total_heat, work)
            dt1_heater = fg_tin - Temperatures[index]
            dt2_heater = fg_tout_i - Temperatures[index - 1]
            if dt2_heater < 0 or dt1_heater < 0:
                return PENALTY_VALUE
            UA_heater = (work / 1e3) / lmtd(dt1_heater, dt2_heater)  # W / °C
            cost_heater[index] = 5000 * UA_heater  # Thesis 97/pdf116
    for index, work in enumerate(q_hx):
        if work > 0:
            dt1_hx = Temperatures[hotside_index - 1] - Temperatures[coldside_index]
            dt2_hx = Temperatures[hotside_index] - Temperatures[coldside_index - 1]
            if dt2_hx < 0 or dt1_hx < 0:
                return PENALTY_VALUE
            UA_hx = (work / 1) / lmtd(dt1_hx, dt2_hx)  # W / °C
            if Temperatures[hotside_index - 1] > 550:
                ft_hx = 1 + 0.02141 * (Temperatures[hotside_index - 1] - 550)
            else:
                ft_hx = 1
            cost_hx[index] = 49.45 * UA_hx**0.7544 * ft_hx  # $
    pec = cost_tur + cost_hx + cost_cooler + cost_comp + cost_heater
    prod_capacity = (sum(w_tur) - sum(w_comp)) / 1e6
    zk, cfueltot, lcoe = economics(pec, prod_capacity)

    # Thermo-economic Analysis
    m1 = np.zeros(
        (
            len(equipment),
            (len(equipment) + equipment.count(1) + equipment.count(3) + 3),
        )
    )
    zero_row = np.zeros(m1.shape[1]).reshape(1, -1)
    total_electricity_production = np.copy(zero_row)
    total_electricity_aux = np.copy(zero_row)
    turbine_token = equipment.count(1)
    comp_token = equipment.count(3)
    hx_token = 1
    for i, j in enumerated_equipment:
        if i == 0:
            inlet = len(Temperatures) - 1
        else:
            inlet = i - 1
        outlet = i

        if j == 1:
            m1[i][inlet] = -1 * exergies[inlet]
            m1[i][outlet] = exergies[outlet]
            m1[i][len(equipment) + 1 + turbine_token] = w_tur[outlet]
            total_electricity_production[0][len(equipment) + 1 + turbine_token] = w_tur[
                outlet
            ]
            turbine_aux = np.copy(zero_row)
            turbine_aux[0][inlet] = 1
            turbine_aux[0][outlet] = -1
            m1 = np.concatenate((m1, turbine_aux), axis=0)
            turbine_token -= 1
        elif j == 2:
            cooler_aux = np.copy(zero_row)
            cooler_aux[0][inlet] = 1
            cooler_aux[0][outlet] = -1
            m1 = np.concatenate((m1, cooler_aux), axis=0)
        elif j == 3:
            m1[i][inlet] = -1 * exergies[inlet]
            m1[i][outlet] = exergies[outlet]
            m1[i][-(1 + comp_token)] = -1 * w_comp[outlet]
            total_electricity_production[0][-(1 + comp_token)] = -1 * w_comp[outlet]
            total_electricity_aux[0][-(1 + comp_token)] = -1
            comp_token -= 1
        elif j == 4:
            m1[i][inlet] = -1 * exergies[inlet]
            m1[i][outlet] = exergies[outlet]
            m1[i][len(equipment)] = -1 * e_fgin[outlet]
            m1[i][len(equipment) + 1] = e_fgout[outlet]
            heater_aux = np.copy(zero_row)
            heater_aux[0][len(equipment)] = 1
            heater_aux[0][len(equipment) + 1] = -1
            m1 = np.concatenate((m1, heater_aux), axis=0)
            ### needs more specification for each possible heater
        elif j == 5 and hx_token == 1:
            m1[i][hotside_index - 1] = -1 * exergies[hotside_index - 1]
            m1[i][hotside_index] = exergies[hotside_index]
            m1[i][coldside_index - 1] = -1 * exergies[coldside_index - 1]
            m1[i][coldside_index] = exergies[coldside_index]
            hxer_aux = np.copy(zero_row)
            hxer_aux[0][hotside_index - 1] = 1
            hxer_aux[0][hotside_index] = -1
            m1 = np.concatenate((m1, hxer_aux), axis=0)
            hx_token = 0
    total_electricity_production[0][-1] = -1 * (sum(w_tur) - sum(w_comp))
    total_electricity_aux[0][-1] = 1
    m1 = np.concatenate((m1, total_electricity_production), axis=0)
    m1 = np.concatenate((m1, total_electricity_aux), axis=0)
    cost_of_fg = np.copy(zero_row)
    cost_of_fg[0][len(equipment)] = 1
    m1 = np.concatenate((m1, cost_of_fg), axis=0)
    m2 = zk + [0] * (len(m1) - len(zk))
    m2[-1] = 8.9e-9 * 3600
    m2 = np.asarray(m2).reshape(-1)
    redundancy = np.where(np.all(m1 == 0, axis=1) == True)[0]
    m1 = np.delete(m1, redundancy, axis=0)
    m2 = np.delete(m2, redundancy, axis=0)
    try:
        costs = np.linalg.solve(m1, m2)
    except:
        print("Matrix solution problem")
        return PENALTY_VALUE
    Closs = costs[len(equipment) + 1] * sum(e_fgout)
    Cfuel = costs[len(equipment)] * sum(e_fgin)
    Ztot = sum(zk)
    Cproduct = Cfuel + Ztot - Closs
    Ep = sum(w_tur) - sum(w_comp)
    for i, j in enumerated_equipment:
        if j == 2:
            dissipation[i] = costs[i] * (exergies[i - 1] - exergies[i]) + zk[i]
    Cdiss = sum(dissipation)
    lcoe_calculated = (costs[-1] * Ep + Cdiss + Closs) / (Ep / 1e6)
    c = lcoe_calculated
    np.set_printoptions(precision=2, suppress=True)
    thermal_efficiency = (Ep) / 40.53e6
    if thermal_efficiency < 0.1575:
        j = 1000 * (0.30 - thermal_efficiency)
    else:
        j = c + max(0, 0.1 - sum(q_hx) / sum(q_heater))
    print(
        f"""
    Turbine Pratio = {tur_pratio[np.where(tur_pratio>1)[0]]}
    Turbine Output = {w_tur[np.where(w_tur>0)[0]]/1e6}MW
    Compressor Pratio = {comp_pratio[np.where(comp_pratio>1)[0]]}
    Compressor Input = {w_comp[np.where(w_comp>0)[0]]/1e6}MW
    Temperatures = {Temperatures}   Tstack = {fg_tout:.1f}    DT = {approach_temp} 
    Pressures =    {Pressures/1e5}bar
    Equipment Cost = {pec/1e3}
    Equipment Duty = Qheater={q_heater[np.where(q_heater>0)[0]]/1e6}MW Qcooler={q_cooler[np.where(q_cooler>0)[0]]/1e6}MW Qhx={q_hx[np.where(q_hx>0)[0]]/1e6}MW
    Objective Function value = {c}
    Exergy of streams = {exergies/1e6}MW
    Exergy costing of streams = {costs/3600*1e9} $/GJ
    Total PEC = {sum(pec):.2f} $
    Total Zk  = {sum(zk):.2f} $/h
    Cdiss = {Cdiss:.2f} Cl = {Closs:.2f} Cp ={costs[-1]*Ep:.2f} LCOE = {lcoe:.2f} LCOEX = {lcoe_calculated:.2f}
    Cp/Ep = {Cproduct/(Ep/1e6):.2f}
    Thermal efficiency = {thermal_efficiency:.2f}%
    j = {j:.2f}
        """
    )
    return c


if __name__ == "__main__":
    layout = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    unitsx = layout[1:-1]
    equipment = np.zeros(len(unitsx)).tolist()
    x = []
    bounds = list(range(len(unitsx)))
    hx_token = 1
    for i in range(len(unitsx)):
        a = np.where(unitsx[i] == 1)[0][0]
        if a == 1:
            equipment[i] = 1
            bounds[i] = (74e5, 300e5)
        elif a == 2:
            equipment[i] = 2
            bounds[i] = (32, 38)
        elif a == 3:
            equipment[i] = 3
            bounds[i] = (74e5, 300e5)
        elif a == 4:
            equipment[i] = 4
            bounds[i] = (180, 530)
        elif a == 5:
            equipment[i] = 5
            if hx_token == 1:
                bounds[i] = (4, 11)
                hx_token = 0
            else:
                bounds[i] = (0, 0)
        elif a == 6:
            equipment[i] = 6
    bounds.append((50, 160))
    nv = len(bounds)
    enumerated_equipment = list(enumerate(equipment))
    # x = [
    #     78.5e5,
    #     10.8,
    #     32.3,
    #     241.3e5,
    #     10.8,
    #     411.4,
    #     93.18,
    # ]
    x = [
        10877767.337246142,
        4.819268742401082,
        35.10533285719053,
        20057147.03993073,
        0.0,
        354.25265693030866,
        153.4111500436299,
    ]
    results_analysis(x, equipment)
