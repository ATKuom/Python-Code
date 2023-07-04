import random
import matplotlib.pyplot as plt
import numpy as np
from Result_analysis import result_analyses
from econ import economics
from functions import (
    Pressure_calculation,
    pinch_calculation,
    lmtd,
    enthalpy_entropy,
    h0,
    s0,
    T0,
    K,
)


# ------------------------------------------------------------------------------
def objective_function(x):
    t3 = x[0]
    t6 = x[1]
    tur_pratio = x[2]
    comp_pratio = x[3]
    m = x[4]

    ##Parameters
    ntur = 0.93  # turbine efficiency     2019 Nabil
    ncomp = 0.89  # compressor efficiency 2019 Nabil
    gamma = 1.28  # 1.28 or 1.33 can be used based on the assumption
    air_temp = 15  # °C
    exhaust_Tin = 630  # °C
    exhaust_m = 935  # kg/s
    cp_gas = 1151  # j/kgK
    PENALTY_VALUE = float(1e6)
    pec = list()

    p1, p2, p3, p4, p5, p6 = Pressure_calculation(tur_pratio, comp_pratio)
    if p6 > 300e5 or p3 < 74e5:
        # print(p1 / 1e5, p6 / 1e5, "Out of bounds pressures")
        return PENALTY_VALUE

    # Turbine
    (h6, s6) = enthalpy_entropy(t6, p6)  # J/kg, J/kgK = °C,Pa
    t1 = (
        (t6 + K) - ntur * ((t6 + K) - (t6 + K) / (tur_pratio ** (1 - 1 / gamma))) - K
    )  # °C
    (h1, s1) = enthalpy_entropy(t1, p1)
    w_tur = m * (h6 - h1)  # W = kg/s*J/kg

    if w_tur < 0:
        # print("negative turbine work")
        return PENALTY_VALUE

    ##Compressor
    (h3, s3) = enthalpy_entropy(t3, p3)
    t4 = (
        (t3 + K) + ((t3 + K) * (comp_pratio ** (1 - 1 / gamma)) - (t3 + K)) / ncomp - K
    )  # °C
    (h4, s4) = enthalpy_entropy(t4, p4)
    w_comp = m * (h4 - h3)  # W = kg/s*J/kg

    if w_comp < 0:
        # print("negative compressor work")
        return PENALTY_VALUE

    ##Heat Exchanger
    t2, t5 = pinch_calculation(t1, h1, t4, h4, p2, p5, m)  # °C
    if t2 == 0 or t5 == 0:
        # print(t1, t4)
        # print("t2 or t5 = 0 ")
        return PENALTY_VALUE

    (h2, s2) = enthalpy_entropy(t2, p2)
    q_hx = m * (h1 - h2)  # W = kg/s*J/kg

    ##Cooler
    if t3 > t2:
        # print("negative cooler work")
        return PENALTY_VALUE
    q_c = m * (h2 - h3)  # W = kg/s*J/kg

    ##Heater
    (h5, s5) = enthalpy_entropy(t5, p5)
    q_heater = m * (h6 - h5)  # W = kg/s*J/kg
    exhaust_Tout = exhaust_Tin - q_heater / (
        exhaust_m * cp_gas
    )  # °C = °C - W/(kg/s*J/kgK)

    e1 = m * ((h1 - h0) - (T0 + K) * (s1 - s0))  # W = kg/s*(J - °C*J/kgK)
    e2 = m * ((h2 - h0) - (T0 + K) * (s2 - s0))
    e3 = m * ((h3 - h0) - (T0 + K) * (s3 - s0))
    e4 = m * ((h4 - h0) - (T0 + K) * (s4 - s0))
    e5 = m * ((h5 - h0) - (T0 + K) * (s5 - s0))
    e6 = m * ((h6 - h0) - (T0 + K) * (s6 - s0))

    # Economic Analysis

    if t6 > 550:
        ft_tur = 1 + 1.106e-4 * (t6 - 550) ** 2
    else:
        ft_tur = 1
    cost_tur = 182600 * ((w_tur / 1e6) ** 0.5561) * ft_tur  # $

    dt1_cooler = t2 - air_temp  # °C
    dt2_cooler = t3 - air_temp  # °C
    UA_cooler = q_c / lmtd(dt1_cooler, dt2_cooler)  # W / °C
    cost_cooler = 32.88 * UA_cooler**0.75

    cost_comp = 1230000 * (w_comp / 1e6) ** 0.3992  # $

    dt1_heater = exhaust_Tin - t6  # °C
    dt2_heater = exhaust_Tout - t5  # °C
    UA_heater = q_heater / lmtd(dt1_heater, dt2_heater)  # W / °C
    if t6 > 550:
        ft_heater = 1 + 0.02141 * (t6 - 550)
    else:
        ft_heater = 1
    cost_heater = 49.45 * UA_heater**0.7544 * ft_heater  # $

    dt1_hx = t1 - t5  # °C
    dt2_hx = t2 - t4  # °C
    UA_hx = q_hx / lmtd(dt1_hx, dt2_hx)  # W / °C
    if t1 > 550:
        ft_hx = 1 + 0.02141 * (t1 - 550)
    else:
        ft_hx = 1
    cost_hx = 49.45 * UA_hx**0.7544 * ft_hx  # $

    pec.append(cost_tur)
    pec.append(cost_hx)
    pec.append(cost_cooler)
    pec.append(cost_heater)
    pec.append(cost_comp)
    prod_capacity = (w_tur - w_comp) / 1e6  # MW
    zk, cftot = economics(pec, prod_capacity)  # $/h
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
    )  # W
    m2 = np.asarray(zk + [0, 0]).reshape(7, 1)
    try:
        costs = np.linalg.solve(m1, m2)  # $/Wh
    except:
        return PENALTY_VALUE
    Cp = costs[6] * w_tur + costs[1] * e2 + costs[5] * e6 - 2 * costs[2] * e3  # $/h
    Cf = cftot * q_heater + costs[6] * w_comp + costs[5] * e6 - costs[1] * e2  # $/h
    Ztot = sum(zk)  # $/h
    Cl = Cf - Cp - Ztot  # $/h
    Ep = (w_tur + e2 + e6 + -2 * e3) / 1e6  # MW
    c = Cp / Ep  # $/MWh
    return c


bounds = [
    (35, 560),
    (250, 560),
    (1, 300 / 74),
    (1, 300 / 74),
    (50, 200),
]  # upper and lower bounds of variables
nv = len(bounds)  # number of variables

# PARAMETERS OF PSO
particle_size = 7 * len(bounds)  # number of particles
iterations = 100  # max number of iterations

# Visualization
fig = plt.figure()
ax = fig.add_subplot()
fig.show()
plt.title("Evolutionary process of the objective function value")
plt.xlabel("Iteration")
plt.ylabel("Objective function ($/MWh)")


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
        result_analyses(global_best_particle_position)
        plt.plot(A)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Main PSO
PSO(objective_function, bounds, particle_size, iterations)
plt.show()
