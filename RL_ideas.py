import numpy as np
import config
import torch
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from LSTM_batch_pack import LSTMtry, model, classes
from LSTM_generation import generation
import torch
import config
import thermo_validity
from econ import economics
from split_functions import (
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
    layout_to_string,
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
            # print("Turbine or Compressor output is less than 0")
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
                # print("Infeasible HX1")
                return PENALTY_VALUE
            if (
                mass_flow[hotside_index - 1] * enthalpies[hotside_index - 1]
                < mass_flow[coldside_index - 1] * enthalpies[coldside_index - 1]
            ):
                # print("Infeasible HX2")
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
        # print("Matrix solution problem")
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
        # print("Result:")
        # print("Optimal solutions:", global_best_particle_position)
        # print("Objective function value:", fitness_global_best_particle_position)
        self.result = objective_function(global_best_particle_position, equipment)
        self.points = global_best_particle_position

        # plt.plot(A)


max_steps = 50
step = 0
lr = 0.0005
y = 0.99
swarmsize_factor = 7
iterations = 30
model.load_state_dict(torch.load(config.MODEL_DIRECTORY / "v26D10_m1.pt"))
optim = torch.optim.Adam(model.parameters(), lr=lr)
empty_tensor = torch.zeros(1, 1, 12)
episode = 0
while step <= max_steps:
    obs = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ).reshape(1, -1, 12)
    episode += 1
    done = False
    Actions, States, Rewards = [], [], []
    while not done:
        # print(obs)
        # pause = input("Press Enter to continue")
        probs = F.softmax(model(obs), dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample().item()
        action_tensor = empty_tensor.clone()
        action_tensor[0, 0, action] = 1.0
        obs_ = torch.cat((obs[0], action_tensor[0]), 0).reshape(1, -1, 12)
        if action == 11 or obs_.shape[1] == 30:
            done = True
            generated_layout = layout_to_string(obs_)
            print(generated_layout, len(generated_layout[0]))
            validity_check = thermo_validity.validity(generated_layout)
            if validity_check == []:
                # print("invalid")
                rew = 0
            else:
                layout = string_to_layout(generated_layout[0])
                equipment, bounds, x, splitter = bound_creation(layout)
                particle_size = swarmsize_factor * len(bounds)
                if 5 in equipment:
                    particle_size += -1 * swarmsize_factor
                if 9 in equipment:
                    particle_size += -2 * swarmsize_factor
                nv = len(bounds)
                try:
                    rew = PSO(
                        objective_function, bounds, particle_size, iterations
                    ).result
                    if rew > 250:
                        rew = 0
                    print(rew)
                except:
                    # print("PSO error")
                    rew = 0
        else:
            rew = 0
        Actions.append(torch.tensor(action, dtype=torch.int))
        States.append(obs)
        Rewards.append(rew)

        obs = obs_.clone().detach()
        step += 1

    DiscountedReturns = []
    for t in range(len(Rewards)):
        G = 0.0
        for k, r in enumerate(Rewards[t:]):
            G += (y**k) * r
        DiscountedReturns.append(G)

    for State, Action, G in zip(States, Actions, DiscountedReturns):
        probs = F.softmax(model(State), dim=1)
        dist = torch.distributions.Categorical(probs=probs)
        log_prob = dist.log_prob(Action)
        loss = -log_prob * G
        optim.zero_grad()
        loss.backward()
        optim.step()
breakpoint()
# %% Play
quit()
for _ in range(5):
    Rewards = []
    obs = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ).reshape(1, -1, 12)
    done = False

    while not done:
        probs = F.softmax(model(obs), dim=1)
        c = torch.distributions.Categorical(probs=probs)
        action = c.sample().item()
        action_tensor = empty_tensor.clone()
        action_tensor[0, 0, action] = 1.0
        obs_ = torch.cat((obs[0], action_tensor[0]), 0).reshape(1, -1, 12)
        if action == 11:
            done = True
        else:
            rew = 0

        obs = obs_.clone().detach()

        Rewards.append(rew)

    print(f"Reward: {sum(Rewards)}")

for i in range(1):
    generated_layout = ["GTACHE"]
