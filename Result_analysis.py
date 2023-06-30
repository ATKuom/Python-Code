import random
import matplotlib.pyplot as plt
import numpy as np
from functions import (
    Pressure_calculation,
    pinch_calculation,
    specific_heat,
    temperature,
    lmtd,
    enthalpy_entropy,
    h0,
    s0,
    T0,
    K,
)
from econ import economics


# ------------------------------------------------------------------------------
def objective_function(x):
    x = [422.7297351545009, 560, 1.001, 1.0010240596110296, 200]
    t3 = x[0]
    t6 = x[1]
    tur_pratio = x[2]
    comp_pratio = x[3]
    m = x[4]

    ##Parameters
    ntur = 0.93  # turbine efficiency     2019 Nabil
    ncomp = 0.89  # compressor efficiency 2019 Nabil
    gamma = 1.28  # 1.28 or 1.33 can be used based on the assumption
    U_hx = 500  # Mean estimation from engineering toolbox
    U_c = U_hx
    cw_temp = 15
    cp_gas = 1151  # j/kgK
    PENALTY_VALUE = float(1e9)
    pec = list()

    p1, p2, p3, p4, p5, p6 = Pressure_calculation(tur_pratio, comp_pratio)
    if p1 == 0:
        return PENALTY_VALUE

    # Turbine
    (h6, s6) = enthalpy_entropy(t6, p6)
    t1 = (t6 + K) - ntur * ((t6 + K) - (t6 + K) / (tur_pratio ** (1 - 1 / gamma))) - K
    (h1, s1) = enthalpy_entropy(t1, p1)
    w_tur = m * (h6 - h1)
    if w_tur < 0:
        return PENALTY_VALUE
    ##Compressor
    (h3, s3) = enthalpy_entropy(t3, p3)
    t4 = (t3 + K) + ((t3 + K) * (comp_pratio ** (1 - 1 / gamma)) - (t3 + K)) / ncomp - K
    (h4, s4) = enthalpy_entropy(t4, p4)
    w_comp = m * (h4 - h3)
    if w_comp < 0:
        return PENALTY_VALUE
    ##Heat Exchanger
    t2, t5 = pinch_calculation(t1, h1, t4, h4, p2, p5, m)
    (h2, s2) = enthalpy_entropy(t2, p2)
    q_hx = h1 - h2

    ##Cooler
    if t3 > t2:
        return PENALTY_VALUE
    q_c = h2 - h3

    ##Heater
    (h5, s5) = enthalpy_entropy(t5, p5)
    q_heater = m * (h6 - h5)

    e1 = m * (h1 - h0 - T0 * (s1 - s0))
    e2 = m * (h2 - h0 - T0 * (s2 - s0))
    e3 = m * (h3 - h0 - T0 * (s3 - s0))
    e4 = m * (h4 - h0 - T0 * (s4 - s0))
    e5 = m * (h5 - h0 - T0 * (s5 - s0))
    e6 = m * (h6 - h0 - T0 * (s6 - s0))

    # Economic Analysis

    if t6 > 550:
        ft_tur = 1 + 1.106e-4 * (t6 - 550) ** 2
    else:
        ft_tur = 1
    cost_tur = 182600 * ((w_tur / 1e6) ** 0.5561) * ft_tur

    dt1_cooler = t2 - cw_temp
    dt2_cooler = t3 - cw_temp
    A_cooler = q_c / (U_c * lmtd(dt1_cooler, dt2_cooler))
    cost_cooler = 32.88 * U_c * A_cooler**0.75

    cost_comp = 1230000 * (w_comp / 1e6) ** 0.3992

    if t6 > 550:
        ft_heater = 1 + 5.4e-5 * (t6 - 550) ** 2
    else:
        ft_heater = 1
    cost_heater = 820800 * (q_heater / 1e6) ** 0.7327 * ft_heater

    dt1_hx = t1 - t5
    dt2_hx = t2 - t4
    A_hx = q_hx / (U_hx * lmtd(dt1_hx, dt2_hx))
    if t1 > 550:
        ft_hx = 1 + 0.02141 * (t1 - 550)
    else:
        ft_hx = 1
    cost_hx = 49.45 * U_hx * A_hx**0.7544 * ft_hx

    pec.append(cost_tur)
    pec.append(cost_hx)
    pec.append(cost_cooler)
    pec.append(cost_heater)
    pec.append(cost_comp)
    prod_capacity = (w_tur - w_comp) / 1e6
    if w_tur < w_comp:
        return PENALTY_VALUE
    zk, cftot = economics(pec, prod_capacity)
    # [c1,c2,c3,c4,c5,c6,cw]
    m1 = np.array(
        [
            [e1, 0, 0, 0, 0, -e6, w_tur],
            [e1, e2, 0, -e4, e5, 0, 0],
            [0, e2, -e3, 0, 0, 0, 0],
            [0, 0, 0, 0, -e5, e6, 0],
            [0, 0, -e3, e4, 0, 0, -w_comp],
            [1, 0, 0, 0, 0, -1, 0],
            [1, -1, 0, 0, 0, 0, 0],
        ]
    )
    m2 = np.asarray(zk + [0, 0]).reshape(7, 1)
    costs = np.real(np.linalg.solve(m1, m2))
    Cp = costs[6] * w_tur + costs[1] * e2 + costs[5] * e6 - 2 * costs[2] * e3
    Cf = cftot * q_heater + costs[6] * w_comp + costs[5] * e6 - costs[1] * e2
    Ztot = sum(zk)
    Cl = Cf - Cp - Ztot
    Ep = w_tur + e2 + e6 + -2e3
    c = Cp / Ep
    Pressure = [p1 / 1e5, p2 / 1e5, p3 / 1e5, p4 / 1e5, p5 / 1e5, p6 / 1e5]
    unit_energy = [w_tur / 1e6, w_comp / 1e6, q_heater / 1e6, q_c / 1e6, q_hx / 1e6]
    print(
        f"""
        p6 = {Pressure[5]:.2f} bar
        p1 = {Pressure[0]:.2f} bar
        Turbine Pratio = {tur_pratio:.2f}
        Turbine output = {unit_energy[0]:.2f} MW
        p3 = {Pressure[2]:.2f} bar
        p4 = {Pressure[3]:.2f} bar
        Compressor Pratio = {comp_pratio:.2f}
        Compressor Input = {unit_energy[1]:.2f} MW
        Temperatures = {t1,t2,t3,t4,t5,t6}
        Equipment Cost = {cost_tur,cost_hx,cost_cooler,cost_comp,cost_heater}
        
        """
    )
    return c


bounds = [
    (35, 560),
    (250, 560),
    (1.001, 4),
    (1, 4),
    (50, 200),
]  # upper and lower bounds of variables
nv = len(bounds)  # number of variables
mm = -1  # if minimization mm, mm = -1; if maximization mm, mm = 1

# PARAMETERS OF PSO
particle_size = 1  # number of particles
iterations = 1  # max number of iterations
w = 0.75  # 0.72984  # inertia constant
c1 = 1  # 2.05  # cognative constant
c2 = 2  # 2.05  # social constant

# Visualization
# fig = plt.figure()
# ax = fig.add_subplot()
# fig.show()
# plt.title("Evolutionary process of the objective function value")
# plt.xlabel("Iteration")
# plt.ylabel("Objective function")


# ------------------------------------------------------------------------------
class Particle:
    def __init__(self, bounds):
        self.particle_position = []
        self.particle_velocity = []
        self.local_best_particle_position = []
        self.fitness_local_best_particle_position = (
            initial_fitness  # objective function value of the best particle position
        )
        self.fitness_particle_position = (
            initial_fitness  # objective function value of the particle position
        )

        for i in range(nv):
            self.particle_position.append(
                random.uniform(bounds[i][0], bounds[i][1])
            )  # generate random initial position
            self.particle_velocity.append(
                random.uniform(-1, 1)
            )  # generate random initial velocity

    def evaluate(self, objective_function):
        self.fitness_particle_position = objective_function(self.particle_position)
        if mm == -1:
            if (
                self.fitness_particle_position
                < self.fitness_local_best_particle_position
            ):
                self.local_best_particle_position = (
                    self.particle_position
                )  # update particle's local best poition
                self.fitness_local_best_particle_position = (
                    self.fitness_particle_position
                )  # update fitness at particle's local best position
        if mm == 1:
            if (
                self.fitness_particle_position
                > self.fitness_local_best_particle_position
            ):
                self.local_best_particle_position = (
                    self.particle_position
                )  # update particle's local best position
                self.fitness_local_best_particle_position = (
                    self.fitness_particle_position
                )  # update fitness at particle's local best position

    def update_velocity(self, global_best_particle_position):
        for i in range(nv):
            r1 = random.random()
            r2 = random.random()

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
        fitness_global_best_particle_position = initial_fitness
        global_best_particle_position = []
        swarm_particle = []
        for i in range(particle_size):
            swarm_particle.append(Particle(bounds))
        A = []

        for i in range(iterations):
            for j in range(particle_size):
                swarm_particle[j].evaluate(objective_function)

                if mm == -1:
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
                if mm == 1:
                    if (
                        swarm_particle[j].fitness_particle_position
                        > fitness_global_best_particle_position
                    ):
                        global_best_particle_position = list(
                            swarm_particle[j].particle_position
                        )
                        fitness_global_best_particle_position = float(
                            swarm_particle[j].fitness_particle_position
                        )

            for j in range(particle_size):
                swarm_particle[j].update_velocity(global_best_particle_position)
                swarm_particle[j].update_position(bounds)

            A.append(fitness_global_best_particle_position)  # record the best fitness

            # Visualization
            # ax.plot(A, color="r")
            # fig.canvas.draw()
            # ax.set_xlim(left_tur=max(0, i - iterations), right=i + 3)
        print("Result:")
        print("Optimal solutions:", global_best_particle_position)
        print("Objective function value:", fitness_global_best_particle_position)


# ------------------------------------------------------------------------------
if mm == -1:
    initial_fitness = float("inf")  # for minimization problem

if mm == 1:
    initial_fitness = -float("inf")  # for maximization problem

# ------------------------------------------------------------------------------
# Main PSO
PSO(objective_function, bounds, particle_size, iterations)
