import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from econ import economics
from functions import (
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


def results_analysis(x, unitsx):
    ntur = 85  # turbine efficiency     2019 Nabil
    ncomp = 82  # compressor efficiency 2019 Nabil
    cw_temp = 19  # °C
    fg_tin = 539  # °C
    fg_m = 68.75  # kg/s
    cooler_pdrop = 1e5
    heater_pdrop = 0
    hx_pdrop = 0.5e5
    PENALTY_VALUE = float(1e6)
    pec = list()
    hx_position = list()
    m = x[-1]

    Temperatures = np.zeros(len(unitsx))
    Pressures = np.zeros(len(unitsx))
    enthalpies = np.zeros(len(unitsx))
    entropies = np.zeros(len(unitsx))
    exergies = np.zeros(len(unitsx))
    w_comp = np.zeros(len(unitsx))
    cost_comp = np.zeros(len(unitsx))
    w_tur = np.zeros(len(unitsx))
    cost_tur = np.zeros(len(unitsx))
    q_cooler = np.zeros(len(unitsx))
    cost_cooler = np.zeros(len(unitsx))
    q_heater = np.zeros(len(unitsx))
    cost_heater = np.zeros(len(unitsx))
    q_hx = np.zeros(len(unitsx))
    cost_hx = np.zeros(len(unitsx))

    Pressures, Temperatures, approach_temp, split_ratio = decision_variable_placement(
        x, enumerated_equipment, Pressures, Temperatures
    )

    # Pressure calculation splitter part is missing still
    Pressures = Pressure_calculation(
        Pressures, equipment, cooler_pdrop, heater_pdrop, hx_pdrop
    )
    # Turbine and Compressor pressure ratio calculation and checking
    tur_pratio, comp_pratio = tur_comp_pratio(enumerated_equipment, Pressures)
    if tur_pratio < 1 or comp_pratio < 1:
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
            m,
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
                    m,
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
        m,
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
        m,
    )

    total_heat = sum(q_heater)
    fg_tout = fg_calculation(fg_m, total_heat)
    if fg_tout < 90:
        print("Too low stack temperature")
        return PENALTY_VALUE
    hout_fg, sout_fg = h_s_fg(fg_tout, P0)

    # Exergy Analysis
    for streams in range(len(exergies)):
        exergies[streams] = m * (
            enthalpies[streams] - h0 - (T0 + K) * (entropies[streams] - s0)
        )
    e_fgin = fg_m * (hin_fg - h0_fg - (T0 + K) * (sin_fg - s0_fg)) + 0.5e6
    e_fgout = fg_m * (hout_fg - h0_fg - (T0 + K) * (sout_fg - s0_fg)) + 0.5e6

    # Economic Analysis

    for work in w_tur:
        if work > 0:
            index = np.where(w_tur == work)[0][0]
            if index == 0:
                index = -1
            else:
                index = index - 1
            if Temperatures[index] > 550:
                ft_tur = 1 + 1.137e-5 * (Temperatures[index] - 550) ** 2
            else:
                ft_tur = 1
            cost_tur[index + 1] = 406200 * ((work / 1e6) ** 0.8) * ft_tur

    for work in w_comp:
        if work > 0:
            cost_comp[np.where(w_comp == work)[0][0]] = 1230000 * (work / 1e6) ** 0.3992

    for work in q_cooler:
        if work > 0:
            index = np.where(q_cooler == work)[0][0]
            if index == 0:
                index = -1
            else:
                index = index - 1
            dt1_cooler = Temperatures[index + 1] - cw_temp
            dt2_cooler = Temperatures[index] - cw_Tout(work)
            if dt2_cooler < 0 or dt1_cooler < 0:
                return PENALTY_VALUE
            UA_cooler = (work / 1) / lmtd(dt1_cooler, dt2_cooler)  # W / °C
            if Temperatures[index - 1] > 550:
                ft_cooler = 1 + 0.02141 * (Temperatures[index - 1] - 550)
            else:
                ft_cooler = 1
            cost_cooler[index + 1] = 49.45 * UA_cooler**0.7544 * ft_cooler  # $
    for work in q_heater:
        if work > 0:
            index = np.where(q_heater == work)[0][0]
            if index == 0:
                index = -1
            else:
                index = index - 1
            fg_tout_i = fg_calculation(fg_m * work / total_heat, work)
            dt1_heater = fg_tin - Temperatures[index + 1]
            dt2_heater = fg_tout_i - Temperatures[index]
            if dt2_heater < 0 or dt1_heater < 0:
                return PENALTY_VALUE
            UA_heater = (work / 1e3) / lmtd(dt1_heater, dt2_heater)  # W / °C
            cost_heater[index + 1] = 5000 * UA_heater  # Thesis 97/pdf116
    for work in q_hx:
        if work > 0:
            index = np.where(q_hx == work)[0][0]
            if index == 0:
                index = -1
            else:
                index = index - 1
            dt1_hx = Temperatures[hotside_index - 1] - Temperatures[coldside_index]
            dt2_hx = Temperatures[hotside_index] - Temperatures[coldside_index - 1]
            if dt2_hx < 0 or dt1_hx < 0:
                return PENALTY_VALUE
            UA_hx = (work / 1) / lmtd(dt1_hx, dt2_hx)  # W / °C
            if Temperatures[hotside_index - 1] > 550:
                ft_hx = 1 + 0.02141 * (Temperatures[hotside_index - 1] - 550)
            else:
                ft_hx = 1
            cost_hx[index + 1] = 49.45 * UA_hx**0.7544 * ft_hx  # $
    pec = cost_tur + cost_hx + cost_cooler + cost_comp + cost_heater
    prod_capacity = (sum(w_tur) - sum(w_comp)) / 1e6
    zk, cfueltot, lcoe = economics(pec, prod_capacity)

    # Exergy Analysis

    ecosts = np.array(
        np.zeros(len(unitsx) + equipment.count(1) + equipment.count(3) + 3)
    )

    breakpoint()
    np.set_printoptions(precision=2, suppress=True)
    print(
        f"""
    Turbine Pratio = {tur_pratio:.2f}
    Compressor Pratio = {comp_pratio:.2f} 
    Temperatures = {Temperatures}
    Pressures =    {Pressures}

        """
    )
    return lcoe


# Turbine Pratio = {tur_pratio:.2f}   p6/p1={Pressure[5]:.2f}bar/{Pressure[0]:.2f}bar
#     Turbine output = {unit_energy[0]:.2f}MW
#     Compressor Pratio = {comp_pratio:.2f}   p3/p4={Pressure[3]:.2f}bar/{Pressure[2]:.2f}bar
#     Compressor Input = {unit_energy[1]:.2f}MW
#     Temperatures = t1={t1:.1f}   t2={t2:.1f}    t3={t3:.1f}    t4={t4:.1f}     t5={t5:.1f}    t6={t6:.1f}   Tstack={fg_tout:.1f}    DT ={approach_temp:.1f}
#     Pressures =    p1={Pressure[0]:.1f}bar p2={Pressure[1]:.1f}bar p3={Pressure[2]:.1f}bar p4={Pressure[3]:.1f}bar p5={Pressure[4]:.1f}bar p6={Pressure[5]:.1f}bar
#     Equipment Cost = Tur={cost_tur/1e3:.0f}    HX={cost_hx/1e3:.0f}    Cooler={cost_cooler/1e3:.0f}    Compr={cost_comp/1e3:.0f}   Heater={cost_heater/1e3:.0f}
#     Equipment Energy = Qheater={unit_energy[2]:.2f}MW  Qcooler={unit_energy[3]:.2f}MW  Qhx={unit_energy[4]:.2f}MW
#     Objective Function value = {c}
#     Exergy of streams = {e1/1e6:.2f}MW {e2/1e6:.2f}MW {e3/1e6:.2f}MW {e4/1e6:.2f}MW {e5/1e6:.2f}MW {e6/1e6:.2f}MW {e_fgin/1e6:.2f}MW {e_fgout/1e6:.2f}MW
#     {costs/3600*1e9}
#     {sum(pec)}
#     {sum(zk)}
#     Cdiss = {cdiss:.2f} Cl = {Cl:.2f} Cp ={costs[-3]*Ep:.2f} LCOE = {lcoe:.2f} LCOEX = {lcoex:.2f}
#     Cp/Ep = {Cp/(Ep/1e6)}
#     Thermal efficiency = {Ep/40.53e6}


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
    particle_size = 7 * len(bounds)
    iterations = 30
    nv = len(bounds)
    enumerated_equipment = list(enumerate(equipment))
    x = [
        78.5e5,
        10.8,
        32.3,
        241.3e5,
        10.8,
        411.4,
        93.18,
    ]
    results_analysis(x, unitsx)
