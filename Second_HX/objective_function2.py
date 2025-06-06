from C_split_functions import *
from C_econ import *


def objective_function2(x, equipment):
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
    if looping == True:
        assumed_temperature = x[-2]
    print(assumed_temperature, "initial assumption")
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
    print(Temperatures, "Initial Temperatures")
    if (
        Temperatures.prod() == 0
        and looping == True
        and Temperatures[hx_position - 1].sum() > 0
    ):
        print("Entering Looping")
        assumed_inlet = (
            hx_position[np.where(Temperatures[hx_position - 1] == 0)[0].item()] - 1
        )
        Temperatures[assumed_inlet] = assumed_temperature
        enthalpies[assumed_inlet], entropies[assumed_inlet] = enthalpy_entropy(
            Temperatures[assumed_inlet], Pressures[assumed_inlet]
        )
        print(assumed_inlet, equipment[assumed_inlet])
        # print(Temperatures, "Initial guess")
        # Branch check
        # Temperatures, enthalpies, entropies = inlet_equipment_check(
        #     equipment_array, assumed_inlet, Temperatures, enthalpies, entropies
        # )
        # print(Temperatures, "After inlet equipment check")
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
                # print(Temperatures,"HX1 Converged Temperatures")
            except:
                hx_converged = False
            if hx_converged == True:
                converged = False
                while_counter2nd = 0
                while converged == False:
                    while_counter2nd += 1
                    # for while_counter2nd in range(10):
                    print(while_counter2nd, Temperatures)
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
                    # print(Temperatures, "After Turbine and Compressor")
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
                    # print(Temperatures, "After Splitter and Mixer")
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
                    # print(Temperatures, "After Splitter2 and Mixer2")
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
                            print("Looping Infeasible HX1")
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
                            print("Looping HX1 calculation error")
                            return PENALTY_VALUE
                    # print(Temperatures, "After HX1")
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
                            print("Looping Infeasible HX2")
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
                            # print("Looping HX2 calculation error")
                            return PENALTY_VALUE
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
                    # print(Temperatures, "After Splitter and Mixer")
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
                        np.round(Temperatures[assumed_inlet]) < 32
                        or np.round(Temperatures[assumed_inlet]) > 530
                    ):
                        return PENALTY_VALUE
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
                        # Temperatures, enthalpies, entropies = inlet_equipment_check(
                        #     equipment_array,
                        #     assumed_inlet,
                        #     Temperatures,
                        #     enthalpies,
                        #     entropies,
                        # )
                        assumed_temperature = Temperatures[assumed_inlet].copy()
                    else:
                        if Temperatures.prod() != 0 and while_counter2nd > 1:
                            if splitter == True:
                                splitter_position = np.where(equipment_array == 9)[0]
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
                                        mixer1, mixer2 = np.where(equipment_array == 8)[
                                            0
                                        ]
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

                    # v1
                    # if Temperatures.prod() != 0 and while_counter2nd > 1:
                    #     if splitter == True:
                    #         splitter_position = np.where(equipment_array == 9)[0]
                    #         mixer1, mixer2 = np.where(equipment_array == 7)[0]
                    #         if (
                    #             Temperatures[mixer1] != Temperatures[splitter_position]
                    #             and Temperatures[splitter_position]
                    #             != Temperatures[splitter_position - 1]
                    #         ):
                    #             if splitter2 == True:
                    #                 splitter_position = np.where(equipment_array == 10)[
                    #                     0
                    #                 ]
                    #                 mixer1, mixer2 = np.where(equipment_array == 8)[0]
                    #                 if (
                    #                     Temperatures[mixer1]
                    #                     != Temperatures[splitter_position]
                    #                     and Temperatures[splitter_position]
                    #                     != Temperatures[splitter_position - 1]
                    #                 ):
                    #                     return PENALTY_VALUE
                    #             else:
                    #                 return PENALTY_VALUE
                    # else:
                    #     Temperatures, enthalpies, entropies = inlet_equipment_check(
                    #         equipment_array,
                    #         assumed_inlet,
                    #         Temperatures,
                    #         enthalpies,
                    #         entropies,
                    #     )
                    #     assumed_temperature = Temperatures[assumed_inlet].copy()
                    # # else:
                    #     if Temperatures.prod() != 0 and while_counter2nd > 1:
                    #         if splitter == True:
                    #             splitter_position = np.where(equipment_array == 9)[0]
                    #             mixer1, mixer2 = np.where(equipment_array == 7)[0]
                    #             if (
                    #                 Temperatures[mixer1]
                    #                 == Temperatures[splitter_position]
                    #                 and Temperatures[splitter_position]
                    #                 == Temperatures[splitter_position - 1]
                    #             ):
                    #                 if splitter2 == True:
                    #                     splitter_position = np.where(
                    #                         equipment_array == 10
                    #                     )[0]
                    #                     mixer1, mixer2 = np.where(equipment_array == 8)[
                    #                         0
                    #                     ]
                    #                     if (
                    #                         Temperatures[mixer1]
                    #                         == Temperatures[splitter_position]
                    #                         and Temperatures[splitter_position]
                    #                         == Temperatures[splitter_position - 1]
                    #                     ):
                    #                         converged = True
                    #                         break
                    #                 else:
                    #                     converged = True
                    #                     break

        print(Temperatures, "Final Temperatures")
        print(assumed_temperature, "Final assumed temperature")
        print("Converged", Temperatures[assumed_inlet])
    if sum(w_tur) < sum(w_comp):
        # print("Negative Net Power Production")
        return PENALTY_VALUE
    for index in cooler_position:
        if Temperatures[index] >= Temperatures[index - 1]:
            print("Infeasible Cooler")
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
    # for i, j, k, l in zip(equipment, Temperatures, mass_flow, pec):
    #     print(f" {i}, {j:.2f}, {k:.0f}, Cost: {l:.0f}")
    Ep = sum(w_tur) - sum(w_comp)
    c = lcoe
    thermal_efficiency = (Ep) / 40.53e6
    if thermal_efficiency < 0.1575:
        j = 100 * (0.30 - thermal_efficiency)
    else:
        j = c + 1 * max(0, 0.1 - sum(q_hx) / sum(q_heater))
    # print("Succesful Completion")
    return c


layout = "GTa1bAC-2H2b2-1aT1HE"
layout = string_to_layout(layout)
equipment, bounds, x, splitter = bound_creation(layout)
print(equipment)
# x = [
#     0.99,
#     23.0,
#     8083650.370567948,
#     0.0,
#     394.1753975282282,
#     8902214.157824906,
#     23.0,
#     0.0,
#     4.0,
#     32.75299068162392,
#     30000000.0,
#     0.01,
#     329.97434984918607,
#     0.0,
#     4.0,
#     0.0,
#     160,
# ]
x = [
    0.01,
    4.0,
    16798681.548698094,
    0.0,
    413.54078609391775,
    7966852.229902331,
    4.0,
    0.0,
    18.792373859576315,
    32.0,
    30000000.0,
    0.37330157783764745,
    248.56915804153167,
    0.0,
    18.792373859576315,
    0.0,
    255.0784330910784,
    101.67140871032058,
]
x = [
    0.01,
    4.0,
    9242383.288953986,
    0.0,
    422.8739138642686,
    7966044.915509527,
    4.0,
    0.0,
    11.330463985596,
    32.0,
    30000000.0,
    0.4141230332783397,
    311.8813979750739,
    0.0,
    11.330463985596,
    0.0,
    500,
    96.39579001965141,
]
x = [
    0.01,
    4.0,
    7913748.301060512,
    0.0,
    420.8006694916358,
    7963100.757983414,
    4.0,
    0.0,
    20.906107853392502,
    32.0,
    30000000.0,
    0.3720053459846817,
    250.54460127177254,
    0.0,
    20.906107853392502,
    0.0,
    344.2155252246139,
    97.34025726746468,
]

x = [
    0.01,
    5.0614012928451615,
    8181032.267760173,
    0.0,
    411.03496988877725,
    8004671.672251339,
    5.0614012928451615,
    0.0,
    14.627648957442384,
    32.0,
    30000000.0,
    0.422615639980114,
    209.33142054103865,
    0.0,
    14.627648957442384,
    0.0,
    500,
    99.3668173685516,
]

# 162 without new_old_assumption middle point

print(objective_function2(x, equipment))
