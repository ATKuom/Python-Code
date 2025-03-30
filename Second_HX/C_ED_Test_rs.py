import numpy as np
import torch
import matplotlib.pyplot as plt
from C_econ import economics
from C_split_functions import (
    fg_calculation,
    HX_calculation,
    decision_variable_placement,
    Pressure_calculation,
    tur_comp_pratio,
    turbine_compressor_calculation,
    splitter_mixer_calc,
    cooler_calculation,
    heater_calculation,
    hx_side_selection,
    enthalpy_entropy,
    bound_creation,
    exergoeconomic_calculation,
    turbine_econ,
    cooler_econ,
    hx_econ,
    heater_econ,
    comp_econ,
    exergy_calculation,
    string_to_layout,
    string_to_equipment,
    T0,
    P0,
    K,
    FGINLETEXERGY,
)
from designs import ED1, ED2, ED3, bestfourthrun

np.set_printoptions(precision=2, suppress=True)


def results_analysis(x, equipment):
    ntur = 85  # 2019 Nabil 93
    ncomp = 82  #  89
    fg_tin = 539.76  # °C 630
    fg_m = 68.75  # kg/s 935
    cooler_pdrop = 1e5  # 0.5e5
    heater_pdrop = 0  # 1e5
    hx_pdrop = 0.5e5  # 1e5
    PENALTY_VALUE = float(1e6)
    splitter = False
    if 9 in equipment:
        splitter = True
    enumerated_equipment = list(enumerate(equipment))
    equipment_length = len(equipment)
    enthalpies = np.zeros(equipment_length)
    entropies = np.zeros(equipment_length)
    exergies = np.zeros(equipment_length)
    w_comp = np.zeros(equipment_length)
    cost_comp = np.zeros(equipment_length)
    w_tur = np.zeros(equipment_length)
    cost_tur = np.zeros(equipment_length)
    q_cooler = np.zeros(equipment_length)
    cost_cooler = np.zeros(equipment_length)
    dissipation = np.zeros(equipment_length)
    q_heater = np.zeros(equipment_length)
    cost_heater = np.zeros(equipment_length)
    q_hx = np.zeros(equipment_length)
    cost_hx = np.zeros(equipment_length)

    (
        Pressures,
        Temperatures,
        approach_temp,
        split_ratio,
        mass_flow,
    ) = decision_variable_placement(x, enumerated_equipment, equipment_length)

    Pressures = Pressure_calculation(
        Pressures, equipment, cooler_pdrop, heater_pdrop, hx_pdrop, splitter
    )

    # it can benefit from tur_ppisition and comp_position
    # Turbine and Compressor pressure ratio calculation and checking
    tur_pratio, comp_pratio = tur_comp_pratio(
        enumerated_equipment, Pressures, equipment_length
    )

    cooler_position = [i for i, j in enumerated_equipment if j == 2]
    for index in cooler_position:
        enthalpies[index], entropies[index] = enthalpy_entropy(
            Temperatures[index], Pressures[index]
        )

    heater_position = [i for i, j in enumerated_equipment if j == 4]
    for index in heater_position:
        enthalpies[index], entropies[index] = enthalpy_entropy(
            Temperatures[index], Pressures[index]
        )

    while_counter = 0
    while Temperatures.prod() == 0:
        # restructuring this part can be useful, separating splitter information from tur/comp calculation while adding if checks
        # combinnig two power checks within the if check
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

        if splitter == True:
            (Temperatures, enthalpies, entropies) = splitter_mixer_calc(
                Temperatures, Pressures, enthalpies, entropies, mass_flow, equipment
            )

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
                # print("Infeasible HX")
                return PENALTY_VALUE
            # if (
            #     mass_flow[hotside_index - 1] * enthalpies[hotside_index - 1]
            #     < mass_flow[coldside_index - 1] * enthalpies[coldside_index - 1]
            # ):
            #     # print("Infeasible HX")
            #     return PENALTY_VALUE
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
                # print("HX calculation error")
                return PENALTY_VALUE
        if while_counter == 3:
            # print("Infeasible Temperatures")
            return PENALTY_VALUE
        while_counter += 1

    for index in cooler_position:
        if Temperatures[index] >= Temperatures[index - 1]:
            # print("Infeasible Cooler")
            return PENALTY_VALUE
    enthalpies, entropies, q_cooler = cooler_calculation(
        cooler_position,
        Temperatures,
        Pressures,
        enthalpies,
        entropies,
        q_cooler,
        cooler_pdrop,
        mass_flow,
    )

    for index in heater_position:
        if Temperatures[index] <= Temperatures[index - 1]:
            # print("Infeasible Temperatures for heater")
            return PENALTY_VALUE
    enthalpies, entropies, q_heater = heater_calculation(
        heater_position,
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

    # Economic Analysis
    cost_tur = turbine_econ(w_tur, Temperatures, cost_tur)
    cost_comp = comp_econ(w_comp, cost_comp)
    cost_cooler = cooler_econ(q_cooler, Temperatures, cost_cooler)
    if np.all(cost_cooler == 0):
        return PENALTY_VALUE
    try:
        cost_heater, fg_mlist, fg_tinlist, fg_toutlist, descending_temp = heater_econ(
            equipment,
            q_heater,
            Temperatures,
            cost_heater,
            heater_position,
            total_heat,
            fg_m,
            fg_tin,
        )
    except:
        # print("Heater calculation error")
        return PENALTY_VALUE
    if hx_position != []:
        cost_hx = hx_econ(q_hx, Temperatures, cost_hx, hotside_index, coldside_index)
        if np.all(cost_hx == 0):
            return PENALTY_VALUE
    pec = cost_tur + cost_hx + cost_cooler + cost_comp + cost_heater
    prod_capacity = (sum(w_tur) - sum(w_comp)) / 1e6
    zk, cfueltot, lcoe = economics(pec, prod_capacity)

    # Exergy Analysis
    exergies, e_fgin, e_fgout = exergy_calculation(
        mass_flow,
        enthalpies,
        entropies,
        q_heater,
        fg_mlist,
        fg_tinlist,
        fg_toutlist,
        equipment_length,
    )
    # Thermo-economic Analysis
    if hx_position == []:
        hotside_index = 0
        coldside_index = 0
    m1, m2 = exergoeconomic_calculation(
        equipment,
        Temperatures,
        enumerated_equipment,
        exergies,
        w_tur,
        w_comp,
        descending_temp,
        e_fgin,
        e_fgout,
        zk,
        hotside_index,
        coldside_index,
    )
    try:
        costs = np.linalg.solve(m1, m2)
    except:
        print("Matrix solution problem")
        return PENALTY_VALUE
    Closs = costs[equipment_length + 1] * min(x for x in e_fgout if x != 0)
    Cfuel = costs[equipment_length] * FGINLETEXERGY
    Ztot = sum(zk)
    Cproduct = Cfuel + Ztot - Closs
    Ep = sum(w_tur) - sum(w_comp)
    for i, j in enumerated_equipment:
        if j == 2:
            dissipation[i] = costs[i] * (exergies[i - 1] - exergies[i]) + zk[i]
    Cdiss = sum(dissipation)
    lcoe_calculated = (costs[-1] * Ep + Cdiss + Closs) / (Ep / 1e6)
    c = lcoe_calculated
    thermal_efficiency = (Ep) / 40.53e6
    if thermal_efficiency < 0.1575:
        j = 100 * (0.30 - thermal_efficiency)
    else:
        j = c + 1 * max(0, 0.1 - sum(q_hx) / sum(q_heater))
    print(
        f"""
    Equipment = {equipment}
    Turbine Pratio = {tur_pratio[np.where(tur_pratio>1.0001)[0]]}
    Turbine Output = {w_tur[np.where(w_tur>0)[0]]/1e6}MW
    Compressor Pratio = {comp_pratio[np.where(comp_pratio>1.0001)[0]]}
    Compressor Input = {w_comp[np.where(w_comp>0)[0]]/1e6}MW
    Split Ratio = {split_ratio:.2f} mass_flow = {mass_flow[:5]}
    Temperatures = {Temperatures}   Tstack = {min(fg_toutlist[np.where(fg_toutlist>0)[0]]):.2f}    DT = {approach_temp}
    Pressures =    {Pressures/1e5} bar
    Equipment Cost = {pec/1e3}
    Equipment Duty = Qheater={q_heater[np.where(q_heater>0)[0]]/1e6}MW   Qcooler={q_cooler[np.where(q_cooler>0)[0]]/1e6}MW   Qhx={q_hx[np.where(q_hx>0)[0]]/1e6}MW
    Objective Function value = {c:.2f} LCOE = {lcoe:.2f} LCOEX = {lcoe_calculated:.2f}
    Exergy of streams = {exergies/1e6}MW
    Exergy of FG streams = {e_fgin[np.where(e_fgin>0)]/1e6}
                           {e_fgout[np.where(e_fgout>0)]/1e6}
                           {fg_mlist[np.where(e_fgout>0)]}
    Exergy costing of streams = {costs/3600*1e9} $/GJ
    Total PEC = {sum(pec):.2f} $
    Total Zk  = {sum(zk):.2f} $/h
    Cdiss = {Cdiss:.2f} Cl = {Closs:.2f} Cp ={costs[-1]*Ep:.2f} LCOE = {lcoe:.2f} LCOEX = {lcoe_calculated:.2f}
    Cp/Ep = {Cproduct/(Ep/1e6):.2f}
    Thermal efficiency = {thermal_efficiency*100:.2f}%
    Heat recuperation ratio = {sum(q_hx)/(sum(q_heater))*100:.2f}
    j = {j:.2f}
        """
    )
    return [
        sum(w_tur) / 1e6,
        sum(w_comp) / 1e6,
        (sum(w_tur) / 1e6 - sum(w_comp) / 1e6),
        sum(q_heater) / 1e6,
        sum(q_cooler) / 1e6,
        sum(q_hx) / 1e6,
        thermal_efficiency,
        sum(zk),
        lcoe,
        lcoe_calculated,
        j,
    ]


if __name__ == "__main__":
    layout = string_to_layout("GT1AT1CaHTaA-1E")
    # layout = ED1
    equipment, bounds, x, splitter = bound_creation(layout)

    x = [
        0.8331321476642084,
        7400000.0,
        0.0,
        32.0,
        7400000.0,
        0.0,
        24870364.745278634,
        11.0,
        381.54252009222853,
        7973375.533590103,
        11.0,
        33.108183738287146,
        119.94762691370104,
    ]
    if torch.equal(layout, ED1):
        x = [
            78.5e5,
            10.8,
            32.3,
            241.3e5,
            10.8,
            411.4,
            93.18,
        ]
    if torch.equal(layout, bestfourthrun):
        x = [
            0.38689869436572294,
            258.2316574319786,
            0.0,
            11.0,
            274.1450967589571,
            0.0,
            420.6860563418102,
            7896946.537168221,
            11.0,
            32.0,
            30000000.0,
            29998432.0682453,
            95.39715459612655,
        ]
    # x = [
    #     0.01,
    #     530.0,
    #     0.0,
    #     11.0,
    #     0.0,
    #     473.7059427076416,
    #     8506178.458468681,
    #     11.0,
    #     32.0,
    #     17011390.549356423,
    #     416.88559492368364,
    #     50,
    # ]
    results_analysis(x, equipment)
