import random
import matplotlib.pyplot as plt
import numpy as np
from pyfluids import Fluid, FluidsList, Input, Mixture
from PSO_ED2_RS import result_analyses
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
    [0.01, 0.99],
    [180.0, 530.0],
    [4.0, 11.0],
    [0.0, 0.0],
    [180.0, 530.0],
    [7400000.0, 30000000.0],
    [32.0, 38.0],
    [7400000.0, 30000000.0],
    [50, 160],
]  # upper and lower bounds of variables

# PARAMETERS OF PSO
particle_size = 7 * len(bounds)  # number of particles
iterations = 300  # max number of iterations
nv = len(bounds)  # number of variables


# ------------------------------------------------------------------------------
def objective_function(x):
    split_ratio = x[0]
    t2 = x[1]
    approach_temp = x[2]
    mixer = x[3]
    t6 = x[4]
    p7 = x[5]
    t9 = x[6]
    p10 = x[7]
    m_set = x[8]

    ##Parameters
    ntur = 85  # turbine efficiency     2019 Nabil
    ncomp = 82  # compressor efficiency 2019 Nabil
    cw_temp = 19  # °C
    fg_tin = 539.76  # °C
    fg_m = 68.75  # kg/s
    cooler_pdrop = 1e5
    heater_pdrop = 0
    hx_pdrop = 0.5e5
    PENALTY_VALUE = float(1e6)
    pec = list()

    m = np.ones(10) * m_set
    m[0] = split_ratio * m_set
    m[1] = m[0]
    m[2] = (1 - split_ratio) * m_set
    m[3] = m[2]

    p8 = p7 - hx_pdrop
    p9 = p8 - cooler_pdrop
    p1 = p10
    p2 = p1 - heater_pdrop
    p3 = p1
    p4 = p3 - hx_pdrop
    p5 = min(p4, p2)
    p6 = p5 - heater_pdrop
    tur_pratio = p6 / p7
    comp_pratio = p10 / p9

    if tur_pratio < 1 or comp_pratio < 1:
        return PENALTY_VALUE

    ##Turbine
    h7, s7, t7, w_tur = turbine(t6, p6, p7, ntur, m[6])
    if w_tur < 0:
        # print("negative turbine work")
        return PENALTY_VALUE

    ##Compressor
    h10, s10, t10, w_comp = compressor(t9, p9, p10, ncomp, m[9])
    if w_comp > w_tur:
        # print("negative net power production")
        return PENALTY_VALUE
    ##Splitter
    h1, s1, t1 = h10, s10, t10
    ##Heater1
    if t1 > t2:
        # print("negative heater1 work")
        return PENALTY_VALUE
    h2, s2, q_heater1 = heater(t1, p1, t2, heater_pdrop, m[1])
    ##Mixer1
    h3, s3, t3 = h1, s1, t1

    ##Heat Exchanger
    t8, h8, s8, t4, h4, s4, q_hx = HX_calculation(
        t7, p7, h7, t3, p3, h3, approach_temp, hx_pdrop, m[7], m[3]
    )
    if t8 == 0:
        return PENALTY_VALUE
    ##Mixer2
    if p4 == p2:
        inlet1 = Fluid(FluidsList.CarbonDioxide).with_state(
            Input.temperature(t2),
            Input.pressure(p2),
        )
        inlet2 = Fluid(FluidsList.CarbonDioxide).with_state(
            Input.temperature(t4),
            Input.pressure(p4),
        )
        outlet = Fluid(FluidsList.CarbonDioxide).mixing(m[1], inlet1, m[3], inlet2)
        t5 = outlet.temperature
        h5 = outlet.enthalpy
        s5 = outlet.entropy
    if p2 > p4:
        hp_inlet = (
            Fluid(FluidsList.CarbonDioxide)
            .with_state(
                Input.temperature(t2),
                Input.pressure(p2),
            )
            .isenthalpic_expansion_to_pressure(p4)
        )
        lp_inlet = Fluid(FluidsList.CarbonDioxide).with_state(
            Input.temperature(t4),
            Input.pressure(p4),
        )
        outlet = Fluid(FluidsList.CarbonDioxide).mixing(m[1], hp_inlet, m[3], lp_inlet)
        t5 = outlet.temperature
        h5 = outlet.enthalpy
        s5 = outlet.entropy
    else:
        hp_inlet = (
            Fluid(FluidsList.CarbonDioxide)
            .with_state(
                Input.temperature(t4),
                Input.pressure(p4),
            )
            .isenthalpic_expansion_to_pressure(p2)
        )
        lp_inlet = Fluid(FluidsList.CarbonDioxide).with_state(
            Input.temperature(t2),
            Input.pressure(p2),
        )
        outlet = Fluid(FluidsList.CarbonDioxide).mixing(m[3], hp_inlet, m[1], lp_inlet)
        t5 = outlet.temperature
        h5 = outlet.enthalpy
        s5 = outlet.entropy
    ##Heater2
    if t5 > t6:
        # print("negative heater2 work")
        return PENALTY_VALUE
    h6, s6, q_heater2 = heater(t5, p5, t6, heater_pdrop, m[5])
    ##Cooler
    if t9 > t8:
        # print("negative cooler work")
        return PENALTY_VALUE
    h9, s9, q_cooler = cooler(t8, p8, t9, cooler_pdrop, m[8])

    fg_tout = fg_calculation(fg_m, q_heater1 + q_heater2)
    if fg_tout < 90:
        # print("too low flue gas stack temperature")
        return PENALTY_VALUE

    # Exergy Analysis
    e1 = m[0] * ((h1 - h0) - (T0 + K) * (s1 - s0))  # W = kg/s*(J - °C*J/kgK)
    e2 = m[1] * ((h2 - h0) - (T0 + K) * (s2 - s0))
    e3 = m[2] * ((h3 - h0) - (T0 + K) * (s3 - s0))
    e4 = m[3] * ((h4 - h0) - (T0 + K) * (s4 - s0))
    e5 = m[4] * ((h5 - h0) - (T0 + K) * (s5 - s0))
    e6 = m[5] * ((h6 - h0) - (T0 + K) * (s6 - s0))
    e7 = m[6] * ((h7 - h0) - (T0 + K) * (s7 - s0))
    e8 = m[7] * ((h8 - h0) - (T0 + K) * (s8 - s0))
    e9 = m[8] * ((h9 - h0) - (T0 + K) * (s9 - s0))
    e10 = m[9] * ((h10 - h0) - (T0 + K) * (s10 - s0))

    if t2 > t6:
        fg_tin2 = fg_tin
        fg_tout2 = fg_calculation(fg_m, q_heater1, fg_tin2)
        fg_tin6 = fg_tout2
        fg_tout6 = fg_calculation(fg_m, q_heater2, fg_tin6)

    else:
        fg_tin6 = fg_tin
        fg_tout6 = fg_calculation(fg_m, q_heater2, fg_tin6)
        fg_tin2 = fg_tout6
        fg_tout2 = fg_calculation(fg_m, q_heater1, fg_tin2)
    hin2_fg, sin2_fg = h_s_fg(fg_tin2, P0)
    hin6_fg, sin6_fg = h_s_fg(fg_tin6, P0)
    hout2_fg, sout2_fg = h_s_fg(fg_tout2, P0)
    hout6_fg, sout6_fg = h_s_fg(fg_tout6, P0)
    e_fgin = fg_m * ((hin_fg - h0_fg) - (T0 + K) * (sin_fg - s0_fg)) + 0.5e6
    e_fgin2 = fg_m * ((hin2_fg - h0_fg) - (T0 + K) * (sin2_fg - s0_fg)) + 0.5e6
    e_fgin6 = fg_m * ((hin6_fg - h0_fg) - (T0 + K) * (sin6_fg - s0_fg)) + 0.5e6
    e_fgout2 = fg_m * ((hout2_fg - h0_fg) - (T0 + K) * (sout2_fg - s0_fg)) + 0.5e6
    e_fgout6 = fg_m * ((hout6_fg - h0_fg) - (T0 + K) * (sout6_fg - s0_fg)) + 0.5e6
    # Economic Analysis
    ##Heater1
    dt1_heater = fg_tin2 - t2  # °C
    dt2_heater = fg_tout2 - t1  # °C
    if dt2_heater < 0 or dt1_heater < 0:
        return PENALTY_VALUE
    UA_heater1 = (q_heater1 / 1e3) / lmtd(dt1_heater, dt2_heater)  # W / °C
    cost_heater1 = 5000 * UA_heater1  # Thesis 97/pdf116
    ##HXer
    dt1_hx = t7 - t4  # °C
    dt2_hx = t8 - t3  # °C
    UA_hx = (q_hx / 1) / lmtd(dt1_hx, dt2_hx)  # W / °C
    if t7 > 550:
        ft_hx = 1 + 0.02141 * (t7 - 550)
    else:
        ft_hx = 1
    cost_hx = 49.45 * UA_hx**0.7544 * ft_hx  # $
    ##Heater2
    dt1_heater = fg_tin6 - t6  # °C
    dt2_heater = fg_tout6 - t5  # °C
    if dt2_heater < 0 or dt1_heater < 0:
        return PENALTY_VALUE
    UA_heater2 = (q_heater2 / 1e3) / lmtd(dt1_heater, dt2_heater)  # W / °C
    cost_heater2 = 5000 * UA_heater2  # Thesis 97/pdf116
    ##Turbine
    if t7 > 550:
        ft_tur = 1 + 1.137e-5 * (t7 - 550) ** 2
    else:
        ft_tur = 1
    cost_tur = 406200 * ((w_tur / 1e6) ** 0.8) * ft_tur  # $
    ##Cooler
    dt1_cooler = t9 - cw_temp  # °C
    dt2_cooler = t8 - cw_Tout(q_cooler)  # °C
    if dt2_cooler < 0 or dt1_cooler < 0:
        return PENALTY_VALUE
    UA_cooler = (q_cooler / 1) / lmtd(dt1_cooler, dt2_cooler)  # W / °C
    if t8 > 550:
        ft_cooler = 1 + 0.02141 * (t8 - 550)
    else:
        ft_cooler = 1
    cost_cooler = 49.45 * UA_cooler**0.7544 * ft_cooler  # $
    ##Compressor
    cost_comp = 1230000 * (w_comp / 1e6) ** 0.3992  # $

    pec.append(cost_heater1)
    pec.append(cost_hx)
    pec.append(cost_heater2)
    pec.append(cost_tur)
    pec.append(cost_comp)
    pec.append(cost_cooler)
    prod_capacity = (w_tur - w_comp) / 1e6  # MW
    zk, cfuel, lcoe = economics(pec, prod_capacity)  # $/h
    # breakpoint()
    # m1 = np.array(
    #     [  # [c1,c2,c3,c4,c5,c6,cwt,cwcomp,ctote,cfgin,cfgout]
    #         # Turbine
    #         [e1, 0, 0, 0, 0, -e6, w_tur, 0, 0, 0, 0],
    #         # HXer
    #         [-e1, e2, 0, -e4, e5, 0, 0, 0, 0, 0, 0],
    #         # Heater
    #         [0, 0, 0, 0, -e5, e6, 0, 0, 0, -e_fgin, e_fgout],
    #         # Compressor
    #         [0, 0, -e3, e4, 0, 0, 0, -w_comp, 0, 0, 0],
    #         # [0, -e2, e3, 0, 0, 0, 0, 0, 0,0, 0],
    #         # Turbine aux1
    #         [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
    #         # HXer aux1
    #         [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         # Cost of FG
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #         # Cooler aux1
    #         [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    #         # Heater aux1
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1],
    #         # Total electricity production
    #         [0, 0, 0, 0, 0, 0, w_tur, -w_comp, -(w_tur - w_comp), 0, 0],
    #         # Total electricity aux1
    #         [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0],
    #     ]
    # )
    # m2 = np.asarray(zk[:4] + [0, 0, 8.9e-9 * 3600, 0, 0, 0, 0]).reshape(
    #     -1,
    # )

    # try:
    #     costs = np.linalg.solve(m1, m2)  # $/Wh
    # except:
    #     return PENALTY_VALUE

    # Cl = costs[10] * e_fgout  # $/h
    # Cf = costs[9] * e_fgin  # $/h
    # Ztot = sum(zk)  # $/h
    # Cp = Cf + Ztot - Cl  # $/h
    # Ep = w_tur - w_comp  # W
    # cdiss = costs[1] * e2 - costs[2] * e3 + zk[-1]
    # lcoex = (costs[-3] * Ep + cdiss + Cl) / (Ep / 1e6)
    c = lcoe
    thermal_efficiency = (w_tur - w_comp) / 40.53e6
    if thermal_efficiency < 0.1575:
        j = 1e5 * (0.30 - thermal_efficiency)
    else:
        j = c + 1e2 * max(0, 0.50 - q_hx / (q_heater1 + q_heater2)) + fg_tout
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
        k = 0
        for i in range(iterations):
            w = (0.4 / iterations**2) * (i - iterations) ** 2 + 0.4
            c1 = -3 * (i / iterations) + 3.5
            c2 = 3 * (i / iterations) + 0.5
            print("iteration = ", i)
            print(w, c1, c2)
            for j in range(particle_size):
                swarm_particle[j].evaluate(objective_function)
                k += 1
                while swarm_particle[j].fitness_particle_position == PENALTY_VALUE:
                    swarm_particle[j] = Particle(bounds)
                    swarm_particle[j].evaluate(objective_function)
                    k += 1

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
        print(k)
        # plt.plot(A)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Main PSO
PSO(objective_function, bounds, particle_size, iterations)
# plt.show()
