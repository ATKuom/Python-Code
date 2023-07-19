import numpy as np
from designs import expert_designs
import torch
import random
import matplotlib.pyplot as plt
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
    NG_exergy,
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
print(unitsx)
temperatures = np.zeros(len(unitsx))
pressures = np.zeros(len(unitsx))
enthalpies = np.zeros(len(unitsx))
entropies = np.zeros(len(unitsx))
exergies = np.zeros(len(unitsx))
equipment = np.zeros(len(unitsx)).tolist()

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
print(equipment)
print(bounds)
particle_size = 1  # 7 * len(bounds)
iterations = 1  # 30
nv = len(bounds)
enumerated_equipment = list(enumerate(equipment))


def objective_function(x):
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

    for equip in equipment:
        if equip == 1:
            pressures[equipment.index(equip)] = x[equipment.index(equip)]
        if equip == 2:
            temperatures[equipment.index(equip)] = x[equipment.index(equip)]
        if equip == 3:
            pressures[equipment.index(equip)] = x[equipment.index(equip)]
        if equip == 4:
            temperatures[equipment.index(equip)] = x[equipment.index(equip)]
        if equip == 5:
            approach_temp = x[equipment.index(equip)]

    while pressures.prod() == 0:
        for i in range(len(pressures)):
            if pressures[i] != 0:
                if i == len(pressures) - 1:
                    if equipment[0] == 2:
                        pressures[0] = pressures[i] - cooler_pdrop
                    if equipment[0] == 4:
                        pressures[0] = pressures[i] - heater_pdrop
                    if equipment[0] == 5:
                        pressures[0] = pressures[i] - hx_pdrop

                else:
                    if equipment[i + 1] == 2:
                        pressures[i + 1] = pressures[i] - cooler_pdrop
                    if equipment[i + 1] == 4:
                        pressures[i + 1] = pressures[i] - heater_pdrop
                    if equipment[i + 1] == 5:
                        pressures[i + 1] = pressures[i] - hx_pdrop
    for equip in equipment:
        if equip == 1:
            if equipment.index(equip) != 0:
                tur_pratio = (
                    pressures[equipment.index(equip) - 1]
                    / pressures[equipment.index(equip)]
                )
            else:
                tur_pratio = pressures[-1] / pressures[equipment.index(equip)]
        if equip == 3:
            if equipment.index(equip) != 0:
                comp_pratio = (
                    pressures[equipment.index(equip)]
                    / pressures[equipment.index(equip) - 1]
                )
            else:
                comp_pratio = pressures[equipment.index(equip)] / pressures[-1]
    if tur_pratio < 1 or comp_pratio < 1:
        return PENALTY_VALUE
    while temperatures.prod() == 0:
        for i in range(len(temperatures)):
            if temperatures[i] != 0:
                if i == len(temperatures) - 1:
                    if equipment[0] == 1:
                        (
                            enthalpies[0],
                            entropies[0],
                            temperatures[0],
                            w_tur[0],
                        ) = turbine(
                            temperatures[i], pressures[i], pressures[0], ntur, m
                        )
                    if equipment[0] == 3:
                        (
                            enthalpies[0],
                            entropies[0],
                            temperatures[0],
                            w_comp[0],
                        ) = compressor(
                            temperatures[i], pressures[i], pressures[0], ncomp, m
                        )

                else:
                    if equipment[i + 1] == 1:
                        (
                            enthalpies[i + 1],
                            entropies[i + 1],
                            temperatures[i + 1],
                            w_tur[i + 1],
                        ) = turbine(
                            temperatures[i], pressures[i], pressures[i + 1], ntur, m
                        )
                    if equipment[i + 1] == 3:
                        (
                            enthalpies[i + 1],
                            entropies[i + 1],
                            temperatures[i + 1],
                            w_comp[i + 1],
                        ) = compressor(
                            temperatures[i], pressures[i], pressures[i + 1], ncomp, m
                        )
        for work in w_tur:
            if work < 0:
                return PENALTY_VALUE
        for work in w_comp:
            if work < 0:
                return PENALTY_VALUE
        hx_position = [i for i, j in enumerated_equipment if j == 5]
        if hx_position != []:
            if hx_position[0] != 0 and hx_position[1] != 0:
                if temperatures[hx_position[0] - 1] > temperatures[hx_position[1] - 1]:
                    hotside_index = hx_position[0]
                    colside_index = hx_position[1]
                else:
                    hotside_index = hx_position[1]
                    colside_index = hx_position[0]
            if hx_position[0] == 0:
                if temperatures[-1] > temperatures[hx_position[1] - 1]:
                    hotside_index = -1
                    colside_index = hx_position[1]
                else:
                    hotside_index = hx_position[1]
                    colside_index = -1
            if hx_position[1] == 0:
                if temperatures[hx_position[0] - 1] > temperatures[-1]:
                    hotside_index = hx_position[0]
                    colside_index = -1
                else:
                    hotside_index = -1
                    colside_index = hx_position[0]
        (
            temperatures[hotside_index],
            enthalpies[hotside_index],
            entropies[hotside_index],
            temperatures[colside_index],
            enthalpies[colside_index],
            entropies[colside_index],
            q_hx[colside_index],
        ) = HX_calculation(
            temperatures[hotside_index - 1],
            pressures[hotside_index - 1],
            enthalpies[hotside_index - 1],
            temperatures[colside_index - 1],
            pressures[colside_index - 1],
            enthalpies[colside_index - 1],
            approach_temp,
            hx_pdrop,
            m,
        )
    cooler_position = [i for i, j in enumerated_equipment if j == 2]
    for i in cooler_position:
        (
            enthalpies[i],
            entropies[i],
            q_cooler[i],
        ) = cooler(
            temperatures[i - 1], pressures[i - 1], temperatures[i], cooler_pdrop, m
        )
    for work in q_cooler:
        if work < 0:
            return PENALTY_VALUE
    heater_position = [i for i, j in enumerated_equipment if j == 4]
    for i in heater_position:
        (
            enthalpies[i],
            entropies[i],
            q_heater[i],
        ) = heater(
            temperatures[i - 1], pressures[i - 1], temperatures[i], heater_pdrop, m
        )
    total_heat = sum(q_heater)
    fg_tout = fg_calculation(fg_m, total_heat)
    if fg_tout < 90:
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
            index = np.where(w_tur == work)
            if temperatures[index] > 550:
                ft_tur = 1 + 1.137e-5 * (temperatures[index] - 550) ** 2
            else:
                ft_tur = 1
            cost_tur[index] = 406200 * ((work / 1e6) ** 0.8) * ft_tur

    for work in w_comp:
        if work > 0:
            cost_comp[np.where(w_comp == work)] = 1230000 * (work / 1e6) ** 0.3992

    for work in q_cooler:
        if work > 0:
            index = np.where(q_cooler == work)
            dt1_cooler = temperatures[index] - cw_temp
            dt2_cooler = temperatures[index - 1] - cw_Tout(work)
            if dt2_cooler < 0 or dt1_cooler < 0:
                return PENALTY_VALUE
            UA_cooler = (work / 1) / lmtd(dt1_cooler, dt2_cooler)  # W / °C
            if temperatures[index - 1] > 550:
                ft_cooler = 1 + 0.02141 * (temperatures[index - 1] - 550)
            else:
                ft_cooler = 1
            cost_cooler[index] = 49.45 * UA_cooler**0.7544 * ft_cooler  # $
    breakpoint()
    return 1


# The output of ML model will be one-hot encoded strings of the layouts
# The output will be cut down using start and end tokens to identify the units in the layout
# Known things from the string: Number of units, their placements, and their connections
# Decision variables are added to the PSO based on their units and placements
# Unit input T and P of i unit = T(i-1), P(i-1) Unit output T and P of i unit = T(i), P(i)
# Pressures of the system are identified then calculated using turbine and compressor pressure ratios
# Temperatures of the system are identified then inputted using Cooler and heater target temperatures
# Mass flow of streams are identified then calculated using mass variable and splitter ratios
# Order of calculations can be done by checking if T(i-1) and P(i-1) are known, if so, calculate the unit function and get T(i) and P(i)

# In my mind there is two different versions of implementation, one is a while loop which constantly checks to obtain all values for T and P and when it is able to obtain them
# It can progress further with exergy analysis,economy analysis and exergoeconomic analysis to obtain the objective function value for the layout and the variables
# It is implemented like checking to see if the input values are known for the functions, if they are, calculate the function and get the output values
# The other version is finding a way to order the functions in a way that the input values are known for the functions, if they are, calculate the function and get the output values

# UNIT_FROM_CHAR = {'T': Turbine}

# def word_to_units(sequence):
#     return [UNIT_FROM_CHAR[char]() for char in sequence]


# if __name__ == "__main__":
#     layout = expert_designs
#     word_to_units(layout[1])

# [T,A,C,H,T,T,C]
# class for units?
# while look without backup or some kind of alternative ending to stop
# #####Profiling if it is slow? Finding the real cause of the speed problems not just assumptions, programs or timing each part to see the performance
# data oriented programming

# class Turbine:
#     DecisionVariables = namedtuple('DecisionVariables', 'pressures[i]')
#     var: DecisionVariables

#     def __init__(self, *var):
#         if len(var) == 0:
#             self.var = #assign randomly
#         self.var = DecisionVariables(var)

#     def outlet_parameters(self, *inlet_parameters):
#         # do computation based on var
#         return outlet_parameters


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
        self.fitness_particle_position = objective_function(self.particle_position)
        if self.fitness_particle_position < self.fitness_local_best_particle_position:
            self.local_best_particle_position = (
                self.particle_position
            )  # update particle's local best poition
            self.fitness_local_best_particle_position = (
                self.fitness_particle_position
            )  # update fitness at particle's local best position

    def update_velocity(self, w, c1, c2, global_best_particle_position):
        for i in range(nv):
            r1 = random.uniform(0, 2)
            r2 = random.uniform(0, 2)

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

        for i in range(iterations):
            w = (0.4 / iterations**2) * (i - iterations) ** 2 + 0.4
            c1 = -3 * (i / iterations) + 3.5
            c2 = 3 * (i / iterations) + 0.5
            print("iteration = ", i)
            print(w, c1, c2)
            for j in range(particle_size):
                swarm_particle[j].evaluate(objective_function)
                while swarm_particle[j].fitness_particle_position == PENALTY_VALUE:
                    swarm_particle[j] = Particle(bounds)
                    swarm_particle[j].evaluate(objective_function)

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
        print("Result:")
        print("Optimal solutions:", global_best_particle_position)
        print("Objective function value:", fitness_global_best_particle_position)
        # result_analyses(global_best_particle_position)
        plt.plot(A)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Main PSO
# PSO(objective_function, bounds, particle_size, iterations)
x = [
    78.5e5,
    10.8,
    32.3,
    241.3e5,
    10.8,
    411.4,
    93.18,
]
objective_function(x)
