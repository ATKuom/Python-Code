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
import config
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from ED_Test_rs import results_analysis
from econ import economics
from split_functions import (
    one_hot_encoding,
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
    T0,
    P0,
    K,
    FGINLETEXERGY,
)


def objective_function(x, equipment):
    ntur = 85  # 2019 Nabil 93
    ncomp = 82  #  89
    fg_tin = 539.76  # °C 630
    fg_m = 68.75  # kg/s 935
    cooler_pdrop = 1e5  # 0.5e5
    heater_pdrop = 0  # 1e5
    hx_pdrop = 0.5e5  # 1e5
    PENALTY_VALUE = float(1e6)

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
    if Pressures.prod() == 0:
        # print("Infeasible Pressure")
        return PENALTY_VALUE
    # it can benefit from tur_ppisition and comp_position
    # Turbine and Compressor pressure ratio calculation and checking
    tur_pratio, comp_pratio = tur_comp_pratio(
        enumerated_equipment, Pressures, equipment_length
    )

    if np.any(tur_pratio <= 1) or np.any(comp_pratio <= 1):
        # print("Turbine or Compressor pressure ratio is less than 1")
        return PENALTY_VALUE

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

        if np.any(w_tur < 0) or np.any(w_comp < 0):
            # print("Turbine or Compressor pressure ratio is less than 1")
            return PENALTY_VALUE

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
            if (
                mass_flow[hotside_index - 1] * enthalpies[hotside_index - 1]
                < mass_flow[coldside_index - 1] * enthalpies[coldside_index - 1]
            ):
                # print("Infeasible HX")
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
                # print("HX calculation error")
                return PENALTY_VALUE
        if while_counter == 3:
            # print("Infeasible Temperatures")
            return PENALTY_VALUE
        while_counter += 1

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
    # print("Succesful Completion")
    return c


# ------------------------------------------------------------------------------
class Particle:
    def __init__(self, bounds):
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
                random.uniform(bounds[i][0], bounds[i][1])
            )  # generate random initial position
            self.particle_velocity.append(
                random.uniform(-1, 1)
            )  # generate random initial velocity

    def evaluate(self, objective_function):
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

    def update_velocity(self, w, c1, c2, global_best_particle_position):
        for i in range(nv):
            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)

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

    def update_position(self, bounds):
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
    def __init__(self, objective_function, bounds, particle_size, iterations):
        fitness_global_best_particle_position = float("inf")
        global_best_particle_position = []
        swarm_particle = []
        PENALTY_VALUE = float(1e6)
        for i in range(particle_size):
            swarm_particle.append(Particle(bounds))
        A = []
        total_number_of_particle_evaluation = 0
        for i in range(iterations):
            w = (0.4 / iterations**2) * (i - iterations) ** 2 + 0.4
            c1 = -3 * (i / iterations) + 3.5
            c2 = 3 * (i / iterations) + 0.5
            # print("iteration = ", i)
            # print(w, c1, c2)
            for j in range(particle_size):
                swarm_particle[j].evaluate(objective_function)
                total_number_of_particle_evaluation += 1
                while (
                    swarm_particle[j].fitness_particle_position == PENALTY_VALUE
                    and i == 0
                    and total_number_of_particle_evaluation < 5e3
                ):
                    swarm_particle[j] = Particle(bounds)
                    swarm_particle[j].evaluate(objective_function)
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
                    w, c1, c2, global_best_particle_position
                )
                swarm_particle[j].update_position(bounds)

            A.append(fitness_global_best_particle_position)  # record the best fitness
            self.result = fitness_global_best_particle_position
            self.position = global_best_particle_position
        # print("Result:")
        # print("Optimal solutions:", global_best_particle_position)
        # print("Objective function value:", fitness_global_best_particle_position)
        # results_analysis(global_best_particle_position, equipment)
        # print(total_number_of_particle_evaluation)
        # plt.plot(A)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Main PSO
if __name__ == "__main__":
    datalist = np.load(
        config.DATA_DIRECTORY / "v21D0_candidates.npy",
        allow_pickle=True,
    )
    # index = np.load(
    #     config.DATA_DIRECTORY / "len20_valid_layouts_all.npy", allow_pickle=True
    # )
    # datalist = np.array(datalist, dtype=object)[index]
    one_hot_tensors = one_hot_encoding(datalist)
    # one_hot_tensors = np.load(
    #     config.DATA_DIRECTORY / "broken_layouts.npy", allow_pickle=True
    # )
    valid_layouts = set()
    penalty_layouts = set()
    broken_layouts = set()
    one_hot_tensors = np.array(one_hot_tensors, dtype=object)
    results = np.zeros(len(datalist))
    positions = np.zeros(len(datalist), dtype=object)
    print(len(datalist))

    for i in range(len(datalist)):
        layout = one_hot_tensors[i]
        equipment, bounds, x, splitter = bound_creation(layout)
        # PSO Parameters
        swarmsize_factor = 7
        particle_size = swarmsize_factor * len(bounds)
        if 5 in equipment:
            particle_size += -1 * swarmsize_factor
        if 9 in equipment:
            particle_size += -2 * swarmsize_factor
        iterations = 30
        nv = len(bounds)
        try:
            a = PSO(objective_function, bounds, particle_size, iterations)
            if a.result < 1e6:
                valid_layouts.add(i)
                results[i] = a.result
                positions[i] = a.position
            else:
                penalty_layouts.add(i)
        except:
            broken_layouts.add(i)
        if i % 100 == 0:
            print(len(valid_layouts), len(penalty_layouts), len(broken_layouts))
    np.save(
        config.DATA_DIRECTORY / "v21D0_candidates_positions.npy",
        positions,
    )
    np.save(config.DATA_DIRECTORY / "v21D0_candidates_results.npy", results)
    # np.save(
    #     config.DATA_DIRECTORY / "v4D0_5k+_valid.npy",
    #     np.array(list(valid_layouts)),
    # )
    # np.save(
    #     config.DATA_DIRECTORY / "v4D0_5k+_penalty.npy",
    #     np.array(list(penalty_layouts)),
    # )
    # np.save(
    #     config.DATA_DIRECTORY / "v4D0_5k+_broken.npy",
    #     np.array(list(broken_layouts)),
    # )
