import random
import matplotlib.pyplot as plt
import numpy as np
from RS_3 import result_analyses
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

bounds = [
    (32, 100),
    (180, 530),
    (74e5, 300e5),
    (74e5, 300e5),
    (50, 160),
    (4, 11),
]  # upper and lower bounds of variables

# PARAMETERS OF PSO
particle_size = 7 * len(bounds)  # number of particles
iterations = 30  # max number of iterations
nv = len(bounds)  # number of variables


# ------------------------------------------------------------------------------
def objective_function(x):
    t3 = x[0]
    t6 = x[1]
    p1 = x[2]
    p4 = x[3]
    m = x[4]
    approach_temp = x[5]

    ##Parameters
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

    p2 = p1 - hx_pdrop
    p3 = p2 - cooler_pdrop
    p5 = p4 - hx_pdrop
    p6 = p5 - heater_pdrop
    tur_pratio = p6 / p1
    comp_pratio = p4 / p3
    if tur_pratio < 1 or comp_pratio < 1:
        return PENALTY_VALUE
    # Turbine
    h1, s1, t1, w_tur = turbine(t6, p6, p1, ntur, m)
    if w_tur < 0:
        # print("negative turbine work")
        return PENALTY_VALUE

    ##Compressor
    h4, s4, t4, w_comp = compressor(t3, p3, p4, ncomp, m)
    if w_comp > w_tur:
        # print("negative net power production")
        return PENALTY_VALUE

    ##Heat Exchanger
    t2, h2, s2, t5, h5, s5, q_hx = HX_calculation(
        t1, p1, h1, t4, p4, h4, approach_temp, hx_pdrop, m
    )
    if t2 == 0:
        return PENALTY_VALUE
    ##Cooler
    if t3 > t2:
        # print("negative cooler work")
        return PENALTY_VALUE
    h3, s3, q_cooler = cooler(t2, p2, t3, cooler_pdrop, m)

    ##Heater
    h6, s6, q_heater = heater(t5, p5, t6, heater_pdrop, m)
    fg_tout = fg_calculation(fg_m, q_heater)
    if fg_tout < 90:
        # print("too low flue gas stack temperature")
        return PENALTY_VALUE
    hout_fg, sout_fg = h_s_fg(fg_tout, P0)

    # Exergy Analysis
    e8 = m * ((h1 - h0) - (T0 + K) * (s1 - s0))  # W = kg/s*(J - °C*J/kgK)
    e9 = m * ((h2 - h0) - (T0 + K) * (s2 - s0))
    e10 = m * ((h3 - h0) - (T0 + K) * (s3 - s0))
    e11 = m * ((h4 - h0) - (T0 + K) * (s4 - s0))
    e12 = m * ((h5 - h0) - (T0 + K) * (s5 - s0))
    e7 = m * ((h6 - h0) - (T0 + K) * (s6 - s0))
    e_fgin = fg_m * ((hin_fg - h0_fg) - (T0 + K) * (sin_fg - s0_fg)) + 0.5e6
    e_fgout = fg_m * ((hout_fg - h0_fg) - (T0 + K) * (sout_fg - s0_fg)) + 0.5e6
    e_fuel = NG_exergy()

    # Economic Analysis
    if t6 > 550:
        ft_tur = 1 + 1.137e-5 * (t6 - 550) ** 2
    else:
        ft_tur = 1
    cost_tur = 406200 * ((w_tur / 1e6) ** 0.8) * ft_tur  # $

    dt1_cooler = t3 - cw_temp  # °C
    dt2_cooler = t2 - cw_Tout(q_cooler)  # °C
    if dt2_cooler < 0 or dt1_cooler < 0:
        return PENALTY_VALUE
    UA_cooler = (q_cooler / 1) / lmtd(dt1_cooler, dt2_cooler)  # W / °C
    if t2 > 550:
        ft_cooler = 1 + 0.02141 * (t2 - 550)
    else:
        ft_cooler = 1
    cost_cooler = 49.45 * UA_cooler**0.7544 * ft_cooler  # $

    cost_comp = 1230000 * (w_comp / 1e6) ** 0.3992  # $

    dt1_heater = fg_tin - t6  # °C
    dt2_heater = fg_tout - t5  # °C
    if dt2_heater < 0 or dt1_heater < 0:
        return PENALTY_VALUE
    UA_heater = (q_heater / 1e3) / lmtd(dt1_heater, dt2_heater)  # W / °C
    cost_heater = 5000 * UA_heater  # Thesis 97/pdf116

    dt1_hx = t1 - t5  # °C
    dt2_hx = t2 - t4  # °C
    UA_hx = (q_hx / 1) / lmtd(dt1_hx, dt2_hx)  # W / °C
    if t1 > 550:
        ft_hx = 1 + 0.02141 * (t1 - 550)
    else:
        ft_hx = 1
    cost_hx = 49.45 * UA_hx**0.7544 * ft_hx  # $
    cost_motor = 211400 * (w_comp / 1e6) ** 0.6227
    cost_generator = 108900 * (w_tur / 1e6) ** 0.5463 + 177200 * (w_tur / 1e6) ** 0.2434
    w_gt = 22.4e6
    pec = [
        2.916e6,
        1.944e6,
        3.888e6,
        0.972e6,
        cost_heater,  # 2.523e6,
        cost_tur,  # 2.734e6,
        cost_hx,  # 1.231e6,
        cost_comp,  # 1.939e6,
        cost_generator,
        cost_motor,
        cost_cooler,  # 1.327e6,
        (w_tur - w_comp) / 1e1,
    ]

    prod_capacity = (w_tur - w_comp + w_gt) / 1e6  # MW
    zk, cfueltot, lcoe = economics(pec, prod_capacity)  # $/h
    neg_zk = [-1 * i for i in zk]
    cfuel = 3.8e-9 * 3600  # cfueltot / e_fuel
    # 3.8e-9 * 3600  # $/MJ
    # cfueltot / e_fuel / 3600  # $/J
    e1 = 0.08e6
    e2 = 27.62e6
    e3 = e_fuel
    e4 = 73.97e6
    e5 = e_fgin
    e6 = e_fgout
    e13 = 0.52e6
    e14 = 1.33e6
    e101 = 29.72e6
    e102 = 22.85e6
    e103 = w_comp
    e104 = w_tur
    e105 = w_gt
    e106 = w_comp
    e107 = 0.95 * w_tur
    e108 = e107 + e105 - e106
    m1 = np.array(
        [  # [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c101,c102,c103,c104,c105,c106,c107,c108]
            # GT compressor
            [e1, -e2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e101, 0, 0, 0, 0, 0, 0, 0],
            # GT combustor
            [0, e2, e3, -e4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # GT expander
            [
                0,
                0,
                0,
                e4,
                -e5,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -e101,
                -e102,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            # GT generator
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e102, 0, 0, -e105, 0, 0, 0],
            # Primary heater
            [0, 0, 0, 0, e5, -e6, -e7, 0, 0, 0, 0, e12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # SCO2 expander
            [0, 0, 0, 0, 0, 0, e7, -e8, 0, 0, 0, 0, 0, 0, 0, 0, 0, -e104, 0, 0, 0, 0],
            # Recuperator
            [0, 0, 0, 0, 0, 0, 0, e8, -e9, 0, e11, -e12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Cooler
            # [0, 0, 0, 0, 0, 0, 0, 0, e9, -e10, 0, 0, e13, -e14, 0, 0, 0, 0, 0, 0, 0, 0],
            # SCO2 compressor
            [0, 0, 0, 0, 0, 0, 0, 0, 0, e10, -e11, 0, 0, 0, 0, 0, e103, 0, 0, 0, 0, 0],
            # SCO2 generator
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e104, 0, 0, -e107, 0],
            # SCO2 motor
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -e103, 0, 0, e106, 0, 0],
            # Power summarizer
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                e105,
                -e106,
                e107,
                -e108,
            ],
            # GT expander aux1
            [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # GT expander aux2
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
            # Primary heater aux1
            [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # SCO2 expander aux1
            [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Recuperator aux1
            [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Cooler aux1
            [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Cooler aux2
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            # Power summarizer aux1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1],
            # Fuel cost
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Air cost
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Cooling water cost
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    m2 = np.asarray(
        neg_zk[:-2]
        + [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            cfuel,
            0,
            0,
        ]
    ).reshape(
        -1,
    )
    try:
        # costs1, _, _, _ = np.linalg.lstsq(m1, m2, rcond=None)  # $/Wh
        # costs2, _ = optimize.nnls(m1, m2[:, 0])
        # print(costs1 / 3600 * 1e9)
        # print(costs2 / 3600 * 1e9)
        costs = np.linalg.solve(m1, m2)

    except:
        return PENALTY_VALUE

    Cl = costs[5] * e_fgout  # $/h
    Cf = costs[2] * e_fuel  # $/h
    Ztot = sum(zk)  # $/h
    Cp = Cf + Ztot - Cl  # $/h
    Ep = e108  # W
    cdiss = (e9 * costs[8] - e10 * costs[9]) + zk[-2]
    lcoex = (costs[21] * Ep + cdiss + Cl) / (Ep / 1e6)
    c = lcoex
    thermal_efficiency = (w_tur - w_comp) / 40.53e6
    if thermal_efficiency < 0.1575:
        j = 1000 * (0.30 - thermal_efficiency)
    else:
        j = c + max(0, 0.1 - q_hx / q_heater)
    return c


# Visualization
# fig = plt.figure()
# ax = fig.add_subplot()
# fig.show()
# plt.title("Evolutionary process of the objective function value")
# plt.xlabel("Iteration")
# plt.ylabel("Objective function ($/MWh)")


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
        # plt.plot(A)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Main PSO
PSO(objective_function, bounds, particle_size, iterations)
# plt.show()
