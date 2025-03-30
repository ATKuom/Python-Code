# Splitter/mixer sitatuion will create different m values which necessitates unit_type more complex approach to the hx.
# ------------Completed Tasks------------
# More than one heater fg_out and exergy analysis maybe necessary?
# At least unit_type partioning between the heaters based on their share on the total heat duty is unit_type reasonable appraoch?
# The partioning is done but the extra e_fgin and e_fgout may be necessary to include per each heater to satisfy the square matrix requirement of the exergy analysis
# Mixing with different pressures is unit_type problem.
# Assumption of flashing the higher pressure stream to the lower pressure stream can be made to mix them.
# After determining the pressures of the system without the mixer, then the mixer must adjust the pressure of the output using the lowest pressure input
# All the m inputs in the functions must be changed accordingly after the implementation of splitter/mixer
# 2 bounds coming from hxer is not affecting anything, so I left it alone. The latter one in the sequence is the one that is used due to decision variable placement. It can be changed or enforced to be the same. The first one goes to lower bound right now without any affect.
# Similarly after determining the temperatures of the system without the mixer, then the mixer must adjust the temperature of the output using mixing method from pyfluids
# Splitter/mixer effects on exergy and overall structure must be analysed
import numpy as np

# import config
# import torch
import random
import matplotlib.pyplot as plt
import time

# from designs import ED1, ED2, ED3, bestfourthrun
# from C_ED_Test_rs import results_analysis
from C_econ import economics
from C_split_functions import (
    string_to_layout,
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
    splitter_mixer_calc_2,
    turbine,
    compressor,
    T0,
    P0,
    K,
    FGINLETEXERGY,
)
from pyfluids import Fluid, FluidsList, Input, Mixture

s = time.time()


# layout = ED1

# layouts = np.load(
#     config.DATA_DIRECTORY / "len20m2_d0.npy",
#     allow_pickle=True,
# )
# layout = layouts[9550]
# layout = "Ta1HTA1ACaH-1H"
# ED1 LCOE = 190.33 / 1316 /16.29
layout = "GTaACaHE"
# ED2 LCOE = 176.98 / 1947 / 31.88
layout = "GTaAC-1H1a1HE"
# ED3 LCOE = 189.40 / 5571 / 49.79
layout = "GTaACH-1H1a1HE"
# ED4 LCOE = 190.18 / 3637 / 1761.70
# [0.5623198665724883, 23.0, 7400000.0, 0.0, 480.9650055930451, 7400000.0, 23.0, 0.0, 22.651074514178212, 32.0, 22674030.21408332, 0.2182940438037221, 293.88010517172813, 0.0, 22.651074514178212, 0.0, 158.8823066511589]
layout = "GTa1bAC-2H2b2-1aT1HE"
# # Turchi 1 LCOE =  189.63 / 3983 / 24.01
# layout = "GTHTaACaHE"
# # Turchi 2 LCOE = 225.8 / 7505 / 641
# # [0.99, 32.0, 24812204.14514859, 22.028828749498928, 0.0, 24789337.922581658, 0.0, 23.0, 316.507960350003, 17986221.211238813, 318.6297118733159, 7847191.108999339, 23.0, 22.028828749498928, 106.7979806920952]
# layout = "GTHTab-1ACb1C1aHE"
# # Turchi 3 LCOE = 192.1 / 7705 / 212
# # [0.01, 32.0, 30000000.0, 23.0, 0.0, 29940954.122531477, 0.0, 10.742237954447821, 241.6636160464584, 29814958.26629695, 407.277458658822, 8052056.127876646, 10.742237954447821, 23.0, 32.35520316129503, 90.36073763794849]
# layout = "GTHTabA-1ACb1C1aHE"
# # Turchi 4 LCOE = 223 / 7924 / 927
# # [0.99, 34.856590651499694, 30000000.0, 33.926360618631506, 30000000.0, 23.0, 0.0, 30000000.0, 0.0, 4.0, 354.89226703384287, 22058224.86662427, 359.0528202433441, 9575081.681978729, 4.0, 23.0, 127.33388639232369]
# layout = "GTHTab-1ACACb1C1aHE"
# # Wright 2 (Cascaded Cycle) LCOE = 188.98 / 2405 / 770
# layout = "GTa1bAC-1baT1HE"
# # Wright 3 (Dual Recuperated) LCOE = 186 / 2475 / 38
# # [0.34619998582590017, 260.2127842133923, 0.0, 18.498551828551193, 0.0, 0.99, 414.48666756624897, 9235181.618685713, 11.347182765230693, 0.0, 11.347182765230693, 9314920.34074137, 0.0, 18.498551828551193, 38.0, 30000000.0, 105.27508757031745]
# layout = "GTa1AC-1aTb1bHE"
# # Noaman 1 (Cascade 3) LCOE = 201 / 3874 / 284
# # [0.99, 180.0, 0.0, 6.112737148253885, 0.0, 0.01, 530.0, 21584882.559266858, 7.3846417476705035, 0.0, 7.3846417476705035, 7749744.42060959, 0.0, 6.112737148253885, 32.0, 30000000.0, 160]
# layout = "GTa2aT2bAC-1H1b1-2HE"
# # Noaman 2 (Example string p125) LCOE = 203 / 2798 / 600
# # [0.01, 30000000.0, 0.0, 33.1461927896632, 30000000.0, 4.263272843768154, 0.0, 13.239562341731204, 437.1213069999182, 8093204.227056596, 13.239562341731204, 4.263272843768154, 80.77610786161615]
# layout = "GTab-1C1ACb1aHE"
# # from 12.Meeting I do not know where LCOE = 186.7 / 2362 / 38.7
# # [0.4211380917601318, 18.274349225423787, 7857359.337067506, 17.479206773468913, 0.0, 17.479206773468913, 497.05536218687956, 7882504.00049771, 18.274349225423787, 0.0, 32.0, 30000000.0, 96.72416689302628]
# layout = "GTa1AC-1aTb1bHE"

layout = string_to_layout(layout)
print(layout)

equipment, bounds, x, splitter = bound_creation(layout)
print(equipment)

# PSO Parameters
swarmsize_factor = 7
particle_size = swarmsize_factor * len(bounds)
if 5 in equipment:
    particle_size += -1 * swarmsize_factor
if 9 in equipment:
    particle_size += -2 * swarmsize_factor
iterations = 30
nv = len(bounds)


def objective_function(x, equipment):
    ntur = 85  # 2019 Nabil 93
    ncomp = 82  #  89
    fg_tin = 539.76  # °C 630
    fg_m = 68.75  # kg/s 935
    cooler_pdrop = 1e5  # 0.5e5
    heater_pdrop = 0  # 1e5
    hx_pdrop = 0.5e5  # 1e5
    PENALTY_VALUE = float(1e6)
    splitter = False
    splitter2 = False
    looping = False
    enumerated_equipment = list(enumerate(equipment))
    equipment_length = len(equipment)
    equipment_array = np.array(equipment)
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
    cost_hx2 = np.zeros(equipment_length)
    if 9 in equipment:
        splitter = True
        if 6 in equipment:
            looping = True
    if 10 in equipment:
        splitter2 = True
        looping = True

    def inlet_equipment_check(
        equipment_array, assumed_inlet, Temperatures, enthalpies, entropies
    ):
        if equipment[assumed_inlet] == 7:
            splitter_position = np.where(equipment_array == 9)[0]
            mixer1, mixer2 = np.where(equipment_array == 7)[0]
            if mixer1 == assumed_inlet:
                Temperatures[splitter_position] = Temperatures[mixer1]
                enthalpies[splitter_position] = enthalpies[mixer1]
                entropies[splitter_position] = entropies[mixer1]

        if equipment[assumed_inlet] == 8:
            splitter_position = np.where(equipment_array == 10)[0]
            mixer3, mixer4 = np.where(equipment_array == 8)[0]
            if mixer3 == assumed_inlet:
                Temperatures[splitter_position] = Temperatures[mixer3]
                enthalpies[splitter_position] = enthalpies[mixer3]
                entropies[splitter_position] = entropies[mixer3]

        if equipment[assumed_inlet] == 9:
            splitter_position = np.where(equipment_array == 9)[0]
            mixer1, mixer2 = np.where(equipment_array == 7)[0]
            Temperatures[mixer1] = Temperatures[splitter_position]
            enthalpies[mixer1] = enthalpies[splitter_position]
            entropies[mixer1] = entropies[splitter_position]
            Temperatures[splitter_position - 1] = Temperatures[assumed_inlet]
            enthalpies[splitter_position - 1] = enthalpies[assumed_inlet]
            entropies[splitter_position - 1] = entropies[assumed_inlet]

        if equipment[assumed_inlet] == 10:
            splitter_position = np.where(equipment_array == 9)[0]
            mixer3, mixer4 = np.where(equipment_array == 8)[0]
            Temperatures[mixer3] = Temperatures[splitter_position]
            enthalpies[mixer3] = enthalpies[splitter_position]
            entropies[mixer3] = entropies[splitter_position]
            Temperatures[splitter_position - 1] = Temperatures[assumed_inlet]
            enthalpies[splitter_position - 1] = enthalpies[assumed_inlet]
            entropies[splitter_position - 1] = entropies[assumed_inlet]
        return Temperatures, enthalpies, entropies

    def c_solver_splitter_mixer_calc(
        splitter_position,
        mixer1,
        mixer2,
        Temperatures,
        Pressures,
        enthalpies,
        entropies,
        mass_flow,
    ):
        if Temperatures[splitter_position - 1] != 0:
            Temperatures[splitter_position] = Temperatures[splitter_position - 1]
            enthalpies[splitter_position] = enthalpies[splitter_position - 1]
            entropies[splitter_position] = entropies[splitter_position - 1]
        # if Temperatures[mixer1] != 0:
        #     Temperatures[splitter_position] = Temperatures[mixer1]
        #     enthalpies[splitter_position] = enthalpies[mixer1]
        #     entropies[splitter_position] = entropies[mixer1]
        Temperatures[mixer1] = Temperatures[splitter_position]
        enthalpies[mixer1] = enthalpies[splitter_position]
        entropies[mixer1] = entropies[splitter_position]
        if (
            Pressures[mixer1 - 1] == Pressures[mixer2 - 1]
            and Temperatures[mixer1 - 1] != 0
            and Temperatures[mixer2 - 1] != 0
        ):
            inlet1 = Fluid(FluidsList.CarbonDioxide).with_state(
                Input.temperature(Temperatures[mixer1 - 1]),
                Input.pressure(Pressures[mixer1 - 1]),
            )
            inlet2 = Fluid(FluidsList.CarbonDioxide).with_state(
                Input.temperature(Temperatures[mixer2 - 1]),
                Input.pressure(Pressures[mixer2 - 1]),
            )
            outlet = Fluid(FluidsList.CarbonDioxide).mixing(
                mass_flow[mixer1 - 1], inlet1, mass_flow[mixer2 - 1], inlet2
            )
            Temperatures[mixer2] = outlet.temperature
            enthalpies[mixer2] = outlet.enthalpy
            entropies[mixer2] = outlet.entropy
        if Temperatures[mixer1 - 1] != 0 and Temperatures[mixer2 - 1] != 0:
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
        return Temperatures, Pressures, enthalpies, entropies

    (
        Pressures,
        Temperatures,
        approach_temp,
        approach_temp_2,
        split_ratio,
        split_ratio_2,
        mass_flow,
        assumed_temperature,
    ) = decision_variable_placement(x, enumerated_equipment, equipment_length, looping)
    Pressures = Pressure_calculation(
        Pressures, equipment, cooler_pdrop, heater_pdrop, hx_pdrop, splitter, splitter2
    )
    if Pressures.prod() == 0:
        # print("Infeasible Pressure")
        return PENALTY_VALUE
    tur_pratio, comp_pratio = tur_comp_pratio(
        enumerated_equipment, Pressures, equipment_length
    )
    if np.any(tur_pratio <= 1) or np.any(comp_pratio <= 1):
        # print("Turbine or Compressor pressure ratio is less than 1")
        return PENALTY_VALUE
    turbine_position = np.where(equipment_array == 1)[0]
    cooler_position = np.where(equipment_array == 2)[0]
    for index in cooler_position:
        enthalpies[index], entropies[index] = enthalpy_entropy(
            Temperatures[index], Pressures[index]
        )
    compressor_position = np.where(equipment_array == 3)[0]
    heater_position = np.where(equipment_array == 4)[0]
    for index in heater_position:
        enthalpies[index], entropies[index] = enthalpy_entropy(
            Temperatures[index], Pressures[index]
        )
    hx_position = np.where(equipment_array == 5)[0]
    hx2_position = np.where(equipment_array == 6)[0]
    # Normal Solver
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

        if np.any(w_tur < 0) or np.any(w_comp < 0):
            # print("Turbine or Compressor output is less than 0")
            return PENALTY_VALUE

        if splitter == True:
            (Temperatures, enthalpies, entropies) = splitter_mixer_calc(
                Temperatures, Pressures, enthalpies, entropies, mass_flow, equipment
            )
        if splitter2 == True:
            (Temperatures, enthalpies, entropies) = splitter_mixer_calc_2(
                Temperatures, Pressures, enthalpies, entropies, mass_flow, equipment
            )

        if (
            hx_position.size != 0
            and Temperatures[hx_position[0] - 1] != 0
            and Temperatures[hx_position[1] - 1] != 0
        ):
            hotside_index, coldside_index = hx_side_selection(hx_position, Temperatures)
            if (
                Temperatures[hotside_index - 1]
                < Temperatures[coldside_index - 1] + approach_temp
            ):
                # print("Infeasible HX1.1")
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
                # print("HX1 calculation error")
                return PENALTY_VALUE

        if (
            hx2_position.size != 0
            and Temperatures[hx2_position[0] - 1] != 0
            and Temperatures[hx2_position[1] - 1] != 0
        ):
            hotside_index2, coldside_index2 = hx_side_selection(
                hx2_position, Temperatures
            )
            if (
                Temperatures[hotside_index2 - 1]
                < Temperatures[coldside_index2 - 1] + approach_temp_2
            ):
                # print("Infeasible HX2.1")
                return PENALTY_VALUE
            try:
                (
                    Temperatures[hotside_index2],
                    enthalpies[hotside_index2],
                    entropies[hotside_index2],
                    Temperatures[coldside_index2],
                    enthalpies[coldside_index2],
                    entropies[coldside_index2],
                    q_hx[min(hotside_index2, coldside_index2)],
                ) = HX_calculation(
                    Temperatures[hotside_index2 - 1],
                    Pressures[hotside_index2 - 1],
                    enthalpies[hotside_index2 - 1],
                    Temperatures[coldside_index2 - 1],
                    Pressures[coldside_index2 - 1],
                    enthalpies[coldside_index2 - 1],
                    approach_temp_2,
                    hx_pdrop,
                    mass_flow[hotside_index2],
                    mass_flow[coldside_index2],
                )
            except:
                # print("HX2 calculation error")
                return PENALTY_VALUE
        if while_counter == 3:
            # print("Infeasible Temperatures")
            if looping:
                break
            else:
                return PENALTY_VALUE
        while_counter += 1
    if splitter == True:
        splitter1_position = np.where(equipment_array == 9)[0]
        mixer1, mixer2 = np.where(equipment_array == 7)[0]
    if splitter2 == True:
        splitter2_position = np.where(equipment_array == 10)[0]
        mixer3, mixer4 = np.where(equipment_array == 8)[0]
    # Initial Guess solver
    if Temperatures.prod() == 0 and looping == True:
        # print(Temperatures, "Initial Temperatures")
        # print("Entering Looping")
        Temperatures_old = Temperatures.copy()
        Temperatures[hx_position - 1]
        assumed_inlet = (
            hx_position[np.where(Temperatures[hx_position - 1] == 0)[0].item()] - 1
        )
        converged = False
        for while_counter in range(10):
            if converged == True:
                break
            hx_converged = False
            Temperatures = Temperatures_old.copy()
            Temperatures[assumed_inlet] = np.random.uniform(50, 500)
            enthalpies[assumed_inlet], entropies[assumed_inlet] = enthalpy_entropy(
                Temperatures[assumed_inlet], Pressures[assumed_inlet]
            )

            # Branch check
            Temperatures, enthalpies, entropies = inlet_equipment_check(
                equipment_array, assumed_inlet, Temperatures, enthalpies, entropies
            )
            assumed_temperature = Temperatures[assumed_inlet].copy()
            hotside_index, coldside_index = hx_side_selection(hx_position, Temperatures)
            if (
                Temperatures[hotside_index - 1]
                > Temperatures[coldside_index - 1] + approach_temp
            ):
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
                    hx_converged = True
                    # print(assumed_temperature, "HX1 Converged")
                except:
                    hx_converged = False
                if hx_converged == True:
                    for while_counter2nd in range(10):
                        # print(Temperatures, while_counter2nd)
                        for index in turbine_position:
                            if Temperatures[index - 1] != 0:
                                (
                                    enthalpies[index],
                                    entropies[index],
                                    Temperatures[index],
                                    w_tur[index],
                                ) = turbine(
                                    Temperatures[index - 1],
                                    Pressures[index - 1],
                                    Pressures[index],
                                    ntur,
                                    mass_flow[index],
                                )
                        for index in compressor_position:
                            if Temperatures[index - 1] != 0:
                                (
                                    enthalpies[index],
                                    entropies[index],
                                    Temperatures[index],
                                    w_comp[index],
                                ) = compressor(
                                    Temperatures[index - 1],
                                    Pressures[index - 1],
                                    Pressures[index],
                                    ncomp,
                                    mass_flow[index],
                                )
                        if splitter == True:
                            Temperatures, Pressures, enthalpies, entropies = (
                                c_solver_splitter_mixer_calc(
                                    splitter1_position,
                                    mixer1,
                                    mixer2,
                                    Temperatures,
                                    Pressures,
                                    enthalpies,
                                    entropies,
                                    mass_flow,
                                )
                            )
                        if splitter2 == True:
                            Temperatures, Pressures, enthalpies, entropies = (
                                c_solver_splitter_mixer_calc(
                                    splitter2_position,
                                    mixer3,
                                    mixer4,
                                    Temperatures,
                                    Pressures,
                                    enthalpies,
                                    entropies,
                                    mass_flow,
                                )
                            )
                        if (
                            hx_position.size != 0
                            and Temperatures[hx_position[0] - 1] != 0
                            and Temperatures[hx_position[1] - 1] != 0
                        ):
                            hotside_index, coldside_index = hx_side_selection(
                                hx_position, Temperatures
                            )
                            if (
                                Temperatures[hotside_index - 1]
                                < Temperatures[coldside_index - 1] + approach_temp
                            ):
                                c = PENALTY_VALUE
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
                                # print("Succesful HX1")
                            except:
                                # print("HX1 calculation error")
                                c = PENALTY_VALUE
                        if (
                            hx2_position.size != 0
                            and Temperatures[hx2_position[0] - 1] != 0
                            and Temperatures[hx2_position[1] - 1] != 0
                        ):
                            hotside_index2, coldside_index2 = hx_side_selection(
                                hx2_position, Temperatures
                            )
                            if (
                                Temperatures[hotside_index2 - 1]
                                < Temperatures[coldside_index2 - 1] + approach_temp_2
                            ):
                                c = PENALTY_VALUE
                            try:
                                (
                                    Temperatures[hotside_index2],
                                    enthalpies[hotside_index2],
                                    entropies[hotside_index2],
                                    Temperatures[coldside_index2],
                                    enthalpies[coldside_index2],
                                    entropies[coldside_index2],
                                    q_hx[min(hotside_index2, coldside_index2)],
                                ) = HX_calculation(
                                    Temperatures[hotside_index2 - 1],
                                    Pressures[hotside_index2 - 1],
                                    enthalpies[hotside_index2 - 1],
                                    Temperatures[coldside_index2 - 1],
                                    Pressures[coldside_index2 - 1],
                                    enthalpies[coldside_index2 - 1],
                                    approach_temp_2,
                                    hx_pdrop,
                                    mass_flow[hotside_index2],
                                    mass_flow[coldside_index2],
                                )
                                # print("Succesful HX2")
                            except:
                                # print("HX2 calculation error")
                                c = PENALTY_VALUE

                        if (
                            np.round(Temperatures[assumed_inlet]) < 32
                            or np.round(Temperatures[assumed_inlet]) > 530
                        ):
                            converged = False
                            break
                        if np.round(Temperatures[assumed_inlet]) != np.round(
                            assumed_temperature
                        ):
                            # print("Old assumption", assumed_temperature)
                            # print(Temperatures[assumed_inlet], "Old Temperature")
                            # print(
                            #     Temperatures[assumed_inlet - 1],
                            #     "Previous equipment Temperature",
                            # )
                            Temperatures[assumed_inlet] = (
                                Temperatures[assumed_inlet] * 2 + assumed_temperature
                            ) / 3
                            enthalpies[assumed_inlet], entropies[assumed_inlet] = (
                                enthalpy_entropy(
                                    Temperatures[assumed_inlet],
                                    Pressures[assumed_inlet],
                                )
                            )
                            # Branch check
                            Temperatures, enthalpies, entropies = inlet_equipment_check(
                                equipment_array,
                                assumed_inlet,
                                Temperatures,
                                enthalpies,
                                entropies,
                            )
                            assumed_temperature = Temperatures[assumed_inlet].copy()
                            # print("New assumption", assumed_temperature)
                        else:
                            if Temperatures.prod() != 0 and while_counter2nd > 1:
                                if splitter == True:
                                    splitter_position = np.where(equipment_array == 9)[
                                        0
                                    ]
                                    mixer1, mixer2 = np.where(equipment_array == 7)[0]
                                    if (
                                        Temperatures[mixer1]
                                        == Temperatures[splitter_position]
                                        and Temperatures[splitter_position]
                                        == Temperatures[splitter_position - 1]
                                    ):
                                        if splitter2 == True:
                                            splitter_position = np.where(
                                                equipment_array == 10
                                            )[0]
                                            mixer1, mixer2 = np.where(
                                                equipment_array == 8
                                            )[0]
                                            if (
                                                Temperatures[mixer1]
                                                == Temperatures[splitter_position]
                                                and Temperatures[splitter_position]
                                                == Temperatures[splitter_position - 1]
                                            ):
                                                converged = True
                                                break
                                        else:
                                            converged = True
                                            break
        if converged == False:
            # print("Infeasible Temperatures")
            return PENALTY_VALUE
        # print(Temperatures, "Final Temperatures")
        # print(assumed_temperature, "Final Assumed Temperature")
        # print(converged, Temperatures[assumed_inlet])

    if sum(w_tur) < sum(w_comp):
        # print("Negative Net Power Production")
        return PENALTY_VALUE

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
    if np.any(q_cooler < 0):
        # print("Negative Cooler Work")
        return PENALTY_VALUE

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

    if np.unique(Temperatures[heater_position]).size != len(
        Temperatures[heater_position]
    ):
        # print("Same Temperature for heater")
        return PENALTY_VALUE

    total_heat = sum(q_heater)
    fg_tout = fg_calculation(fg_m, total_heat)
    if fg_tout < 90:
        # print("Too low stack temperature")
        return PENALTY_VALUE
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
    if hx_position.size != 0:
        cost_hx = hx_econ(q_hx, Temperatures, cost_hx, hotside_index, coldside_index)
        if np.all(cost_hx == 0):
            return PENALTY_VALUE
    if hx2_position.size != 0:
        cost_hx2 = hx_econ(q_hx, Temperatures, cost_hx, hotside_index2, coldside_index2)
        if np.all(cost_hx == 0):
            return PENALTY_VALUE
    pec = cost_tur + cost_hx + cost_cooler + cost_comp + cost_heater + cost_hx2
    prod_capacity = (sum(w_tur) - sum(w_comp)) / 1e6
    zk, cfueltot, lcoe = economics(pec, prod_capacity)

    # # Exergy Analysis
    # exergies, e_fgin, e_fgout = exergy_calculation(
    #     mass_flow,
    #     enthalpies,
    #     entropies,
    #     q_heater,
    #     fg_mlist,
    #     fg_tinlist,
    #     fg_toutlist,
    #     equipment_length,
    # )
    # # Thermo-economic Analysis
    # if hx_position == []:
    #     hotside_index = 0
    #     coldside_index = 0
    # m1, m2 = exergoeconomic_calculation(
    #     equipment,
    #     Temperatures,
    #     enumerated_equipment,
    #     exergies,
    #     w_tur,
    #     w_comp,
    #     descending_temp,
    #     e_fgin,
    #     e_fgout,
    #     zk,
    #     hotside_index,
    #     coldside_index,
    # )
    # try:
    #     costs = np.linalg.solve(m1, m2)
    # except:
    #     print("Matrix solution problem")
    #     return PENALTY_VALUE
    # Closs = costs[equipment_length + 1] * min(x for x in e_fgout if x != 0)
    # Cfuel = costs[equipment_length] * FGINLETEXERGY
    Ztot = sum(zk)
    # Cproduct = Cfuel + Ztot - Closs
    Ep = sum(w_tur) - sum(w_comp)
    # for i, j in enumerated_equipment:
    #     if j == 2:
    #         dissipation[i] = costs[i] * (exergies[i - 1] - exergies[i]) + zk[i]
    # Cdiss = sum(dissipation)
    # lcoe_calculated = (costs[-1] * Ep + Cdiss + Closs) / (Ep / 1e6)
    c = lcoe
    thermal_efficiency = (Ep) / 40.53e6
    if thermal_efficiency < 0.1575:
        j = 100 * (0.30 - thermal_efficiency)
    else:
        j = c + 1 * max(0, 0.1 - sum(q_hx) / sum(q_heater))
    # print("Succesful Completion")
    return c


# ------------------------------------------------------------------------------
class Particle:
    def __init__(self, bounds, nv):
        self.particle_position = []
        self.particle_velocity = []
        self.local_best_particle_position = []
        self.fitness_local_best_particle_position = float(
            "inf"
        )  # objective function value of the best particle position
        self.fitness_particle_position = float(
            "inf"
        )  # objective function value of the particle position

        for i in range(nv):
            self.particle_position.append(
                np.random.uniform(bounds[i][0], bounds[i][1])
            )  # generate random initial position
            self.particle_velocity.append(
                np.random.uniform(-1, 1)
            )  # generate random initial velocity

    def evaluate(self, objective_function, equipment):
        self.fitness_particle_position = objective_function(
            self.particle_position, equipment
        )
        if self.fitness_particle_position < self.fitness_local_best_particle_position:
            self.local_best_particle_position = (
                self.particle_position
            )  # update particle's local best poition
            self.fitness_local_best_particle_position = (
                self.fitness_particle_position
            )  # update fitness at particle's local best position

    def update_velocity(self, w, c1, c2, global_best_particle_position, nv):
        for i in range(nv):
            r1 = np.random.uniform(0, 1)
            r2 = np.random.uniform(0, 1)

            # local explorative position displacement component
            cognitive_velocity = (
                c1
                * r1
                * (self.local_best_particle_position[i] - self.particle_position[i])
            )

            # position displacement component towards global best
            social_velocity = (
                c2 * r2 * (global_best_particle_position[i] - self.particle_position[i])
            )

            self.particle_velocity[i] = (
                w * self.particle_velocity[i] + cognitive_velocity + social_velocity
            )

    def update_position(self, bounds, nv):
        for i in range(nv):
            self.particle_position[i] = (
                self.particle_position[i] + self.particle_velocity[i]
            )

            # check and repair to satisfy the upper bounds
            if self.particle_position[i] > bounds[i][1]:
                self.particle_position[i] = bounds[i][1]
            # check and repair to satisfy the lower bounds
            if self.particle_position[i] < bounds[i][0]:
                self.particle_position[i] = bounds[i][0]


class PSO:
    def __init__(
        self, objective_function, bounds, particle_size, iterations, nv, equipment
    ):
        fitness_global_best_particle_position = float("inf")
        global_best_particle_position = []
        swarm_particle = []
        PENALTY_VALUE = float(1e6)
        for i in range(particle_size):
            swarm_particle.append(Particle(bounds, nv))
        A = []
        total_number_of_particle_evaluation = 0
        for i in range(iterations):
            print(i)
            w = (0.4 / iterations**2) * (i - iterations) ** 2 + 0.4
            c1 = -3 * (i / iterations) + 3.5
            c2 = 3 * (i / iterations) + 0.5
            # print("iteration = ", i)
            # print(w, c1, c2)
            for j in range(particle_size):
                swarm_particle[j].evaluate(objective_function, equipment)
                total_number_of_particle_evaluation += 1
                while (
                    swarm_particle[j].fitness_particle_position == PENALTY_VALUE
                    and i == 0
                    and total_number_of_particle_evaluation < 5e3
                ):
                    swarm_particle[j] = Particle(bounds, nv)
                    swarm_particle[j].evaluate(objective_function, equipment)
                    total_number_of_particle_evaluation += 1
                if (
                    swarm_particle[j].fitness_particle_position
                    < fitness_global_best_particle_position
                ):
                    global_best_particle_position = list(
                        swarm_particle[j].particle_position
                    )
                    fitness_global_best_particle_position = float(
                        swarm_particle[j].fitness_particle_position
                    )

            for j in range(particle_size):
                swarm_particle[j].update_velocity(
                    w, c1, c2, global_best_particle_position, nv
                )
                swarm_particle[j].update_position(bounds, nv)

            A.append(fitness_global_best_particle_position)  # record the best fitness
        print("Result:")
        print("Optimal solutions:", global_best_particle_position)
        print("Objective function value:", fitness_global_best_particle_position)
        # self.result = results_analysis(global_best_particle_position, equipment)
        self.points = global_best_particle_position
        print(total_number_of_particle_evaluation)
        # plt.plot(A)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Main PSO

PSO(objective_function, bounds, particle_size, iterations, nv, equipment)
e = time.time()
print(e - s)
# layouts = np.load(
#     config.DATA_DIRECTORY / "v3.2DF_sorted_layouts.npy",
#     allow_pickle=True,
# )
# results = []
# points = []
# for layout in layouts[:12]:
#     layout = string_to_layout(layout)

#     equipment, bounds, x, splitter = bound_creation(layout)

#     # PSO Parameters
#     swarmsize_factor = 7
#     particle_size = swarmsize_factor * len(bounds)
#     if 5 in equipment:
#         particle_size += -1 * swarmsize_factor
#     if 9 in equipment:
#         particle_size += -2 * swarmsize_factor
#     iterations = 30
#     nv = len(bounds)
#     try:
#         a = PSO(objective_function, bounds, particle_size, iterations)
#         results.append(a.result)
#         points.append(a.points)
#     except:
#         results.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#         points.append([0])
# results_array = np.asarray(results)
# points = np.asarray(points, dtype=object)
# print(points)
# print(results_array)

# np.save(
#     config.DATA_DIRECTORY / "len20m2v2_final_sorted_layouts_lessthanED1_results.npy",
#     results_array,
# )
# x = [
#     78.5e5,
#     10.8,
#     32.3,
#     241.3e5,
#     10.8,
#     411.4,
#     93.18,
# ]
