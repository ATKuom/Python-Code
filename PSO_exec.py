import random
import matplotlib.pyplot as plt
import numpy as np
from functions import specific_heat, temperature, lmtd, enthalpy_entropy, h0, s0, T0, K
from econ import economics


# ------------------------------------------------------------------------------
def objective_function(x):
    ##Variables
    # t1 = x[0]
    # t2 = x[1]
    t3 = x[2]
    # t4 = x[3]
    # t5 = x[4]
    t6 = x[5]
    tur_pratio = x[6]
    comp_pratio = x[7]
    m = x[8]
    # p1 = x[9]
    # p2 = x[10]
    # p3 = x[11]
    # p4 = x[12]
    # p5 = x[13]
    p6 = x[14]

    ##Parameters
    ntur = 0.93  # turbine efficiency     2019 Nabil
    ncomp = 0.89  # compressor efficiency 2019 Nabil
    gamma = 1.28  # 1.28 or 1.33 can be used based on the assumption
    U_hx = 500  # Mean estimation from engineering toolbox
    U_c = U_hx
    cw_temp = 15
    cp_gas = 1151  # j/kgK
    pec = list()

    p1 = p6 / tur_pratio
    p2 = p1 - 1e5
    p3 = p2 - 0.5e5
    p4 = p3 * comp_pratio
    p5 = p4 - 1e5
    p6 = p5 - 1e5

    # Turbine
    t1 = (t6 + K) - ntur * ((t6 + K) - (t6 + K) / (tur_pratio ** (1 - 1 / gamma))) - K
    (h1, s1) = enthalpy_entropy(t1, p1)
    ##Compressor
    t4 = (t3 + K) + ((t3 + K) * (tur_pratio ** (1 - 1 / gamma)) - (t3 + K)) / ncomp - K
    (h4, s4) = enthalpy_entropy(t4, p4)
    ##Heat Exchanger
    t2 = [t2 for t2 in range(int(t4) + 5, int(t1))]
    if len(t2) == 0:
        return float("inf")
    h2 = list()
    for temp in t2:
        a, _ = enthalpy_entropy(temp, p2)
        h2.append(a)
    h2 = np.asarray(h2)
    q_hx1 = m * h1 - m * h2
    t5 = [t2 for t2 in range(int(t4), int(t1) - 5)]
    if len(t5) == 0:
        return float("inf")
    h5 = list()
    for temp in t5:
        a, _ = enthalpy_entropy(temp, p5)
        h5.append(a)
    h5 = np.asarray(h5)
    q_hx2 = m * h5 - m * h4
    q_hx = q_hx1 - q_hx2
    index = np.where(q_hx[:-1] * q_hx[1:] < 0)[0]
    t2 = t2[index[0]]
    (h2, s2) = enthalpy_entropy(t2, p2)
    t5 = t5[index[0]]
    (h5, s5) = enthalpy_entropy(t5, p5)
    ##Heater

    (h3, s3) = enthalpy_entropy(t3, p3)
    (h6, s6) = enthalpy_entropy(t6, p6)
    e1 = m * (h1 - h0 - T0 * (s1 - s0))
    e2 = m * (h2 - h0 - T0 * (s2 - s0))
    e3 = m * (h3 - h0 - T0 * (s3 - s0))
    e4 = m * (h4 - h0 - T0 * (s4 - s0))
    e5 = m * (h5 - h0 - T0 * (s5 - s0))
    e6 = m * (h6 - h0 - T0 * (s6 - s0))

    # Economic Analysis
    w_tur = m * (h6 - h1)
    if t6 > 550:
        ft_tur = 1 + 1.106e-4 * (t6 - 550) ** 2
    elif t1 > 550:
        ft_tur = 1 + 1.106e-4 * (t1 - 550) ** 2
    else:
        ft_tur = 1
    cost_tur = 182600 * (w_tur**0.5561) * ft_tur

    q_c = h2 - h3
    dt1_cooler = t2 - cw_temp
    dt2_cooler = t3 - cw_temp
    A_cooler = q_c / (U_c * lmtd(dt1_cooler, dt2_cooler))
    cost_cooler = 32.88 * U_c * A_cooler**0.75

    w_comp = m * (h4 - h3)
    cost_comp = 1230000 * w_comp**0.3992

    q_heater = m * (h6 - h5)
    if t6 > 550:
        ft_heater = 1 + 5.4e-5 * (t6 - 550) ** 2
    elif t5 > 550:
        ft_heater = 1 + 5.4e-5 * (t5 - 550) ** 2
    else:
        ft_heater = 1
    cost_heater = 820800 * q_heater**0.7327 * ft_heater

    q_hx = h1 - h2
    dt1_hx = t1 - t5
    dt2_hx = t2 - t4
    A_hx = q_hx / (U_hx * lmtd(dt1_hx, dt2_hx))
    if t1 > 550:
        ft_hx = 1 + 0.02141 * (t1 - 550)
    elif t5 > 550:
        ft_hx = 1 + 0.02141 * (t5 - 550)
    else:
        ft_hx = 1
    cost_HX = 49.45 * U_hx * A_hx**0.7544 * ft_hx

    pec.append(cost_tur)
    pec.append(cost_HX)
    pec.append(cost_cooler)
    pec.append(cost_heater)
    pec.append(cost_comp)
    prod_capacity = (w_tur - w_comp) / 1e6
    zk, cftot = economics(pec, prod_capacity)

    # [c1,c2,c3,c4,c5,c6,cw,cheat]
    m1 = np.array(
        [
            [e1, 0, 0, 0, 0, -e6, w_tur, 0],
            [e1, e2, 0, -e4, e5, 0, 0, 0],
            [0, e2, -e3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -e5, e6, 0, q_heater],
            [0, 0, -e3, e4, 0, 0, -w_comp, 0],
            [1, 0, 0, 0, 0, -1, 0, 0],
            [1, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    m2 = np.asarray(zk + [0, 0, cftot]).reshape(8, 1)
    costs = np.linalg.solve(m1, m2)
    Cp = costs[6] * w_tur + costs[1] * e2 + costs[5] * e6 - 2 * costs[2] * e3
    Cf = cftot * q_heater + costs[6] * w_comp + costs[5] * e6 - costs[1] * e2
    Zk = sum(zk)
    Cl = Cp - Cf - Zk
    Ep = w_tur + e2 + e6 + -2e3
    z = Cp / Ep
    return z


bounds = [
    (35, 560),
    (35, 560),
    (35, 560),
    (35, 560),
    (35, 560),
    (235, 560),
    (1, 25),
    (1, 25),
    (50, 200),
    (74e5, 250e5),
    (74e5, 250e5),
    (74e5, 250e5),
    (74e5, 250e5),
    (74e5, 250e5),
    (74e5, 250e5),
]  # upper and lower bounds of variables
nv = len(bounds)  # number of variables
mm = -1  # if minimization mm, mm = -1; if maximization mm, mm = 1

# PARAMETERS OF PSO
particle_size = 40  # number of particles
iterations = 5  # max number of iterations
w = 0.72984  # inertia constant
c1 = 2.05  # cognative constant
c2 = 2.05  # social constant

# Visualization
fig = plt.figure()
ax = fig.add_subplot()
fig.show()
plt.title("Evolutionary process of the objective function value")
plt.xlabel("Iteration")
plt.ylabel("Objective function")


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
        plt.plot(A)


# ------------------------------------------------------------------------------
if mm == -1:
    initial_fitness = float("inf")  # for minimization problem

if mm == 1:
    initial_fitness = -float("inf")  # for maximization problem

# ------------------------------------------------------------------------------
# Main PSO
PSO(objective_function, bounds, particle_size, iterations)
plt.show()
