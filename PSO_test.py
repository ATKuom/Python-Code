import random
import matplotlib.pyplot as plt
from functions import temperature, lmtd, enthalpy_entropy, h0, s0, T0, K


# ------------------------------------------------------------------------------
def objective_function(x):
    ##Variables
    # t1 = O[0]
    t2 = x[1]
    t3 = x[2]
    # t4 = O[3]
    # t5 = x[4]
    t6 = x[5]
    tur_pratio = x[6]
    comp_pratio = x[7]
    m = x[8]
    # p1 = O[9]
    # p2 = O[10]
    # p3 = O[11]
    # p4 = O[12]
    # p5 = O[13]
    p6 = x[14]

    ##Parameters
    ntur = 0.93  # turbine efficiency     2019 Nabil
    ncomp = 0.89  # compressor efficiency 2019 Nabil
    gamma = 1.28  # 1.28 or 1.33 can be used based on the assumption
    U_hx = 500  # Mean estimation from engineering toolbox
    U_c = U_hx
    cw_temp = 15
    cp_gas = 11514  # j/kgK
    total_cost = 0

    (h6, s6) = enthalpy_entropy(t6, p6)
    e6 = m * (h6 - h0 - T0 * (s6 - s0))

    ##Turbine
    t1 = max(
        (t6 + K) - ntur * ((t6 + K) - (t6 + K) / (tur_pratio ** (1 - 1 / gamma))) - K,
        35,
    )
    p1 = p6 / tur_pratio
    (h1, s1) = enthalpy_entropy(t1, p1)
    e1 = m * (h1 - h0 - T0 * (s1 - s0))
    w_tur = max(m * (h6 - h1), 0.01)
    fuel_tur = e6 - e1
    prod_tur = w_tur
    if t6 > 550:
        ft_tur = 1 + 1.106e-4 * (t6 - 550) ** 2
    elif t1 > 550:
        ft_tur = 1 + 1.106e-4 * (t1 - 550) ** 2
    else:
        ft_tur = 1
    cost_tur = 182600 * (w_tur**0.5561) * ft_tur
    total_cost += cost_tur
    cost_prod_execo_tur = (fuel_tur + cost_tur) / w_tur

    ##Heat Exchanger Hot side
    p2 = p1 - 1e5
    (h2, s2) = enthalpy_entropy(t2, p2)
    e2 = m * (h2 - h0 - T0 * (s2 - s0))
    q_hx = max(h1 - h2, 0.1)
    fuel_HX = e1 - e2

    ##Cooler
    p3 = p2 - 0.5e5
    (h3, s3) = enthalpy_entropy(t3, p3)
    e3 = m * (h3 - h0 - T0 * (s3 - s0))
    q_c = max(h2 - h3, 0.1)
    fuel_cooler = q_c
    prod_cooler = e2 - e3
    dt1_cooler = t2 - cw_temp
    dt2_cooler = t3 - cw_temp
    A_cooler = q_c / (U_c * lmtd(dt1_cooler, dt2_cooler))
    cost_cooler = 32.88 * U_c * A_cooler**0.75
    total_cost += cost_cooler

    ##Compressor
    t4 = (t3 + K) + ((t3 + K) * (tur_pratio ** (1 - 1 / gamma)) - (t3 + K)) / ncomp - K
    p4 = p3 * comp_pratio
    (h4, s4) = enthalpy_entropy(t4, p4)
    e4 = m * (h4 - h0 - T0 * (s4 - s0))
    w_comp = m * (h4 - h3)
    fuel_comp = w_comp
    prod_comp = e4 - e3
    cost_comp = 1230000 * w_comp**0.3992
    total_cost += cost_comp

    # ##Heat Exchanger Cold Side
    p5 = p4 - 1e5
    h5 = h4 + q_hx
    t5 = temperature(h5, p5)
    (h5, s5) = enthalpy_entropy(t5, p5)
    e5 = m * (h5 - h0 - T0 * (s5 - s0))
    dt1_hx = t1 - t5
    dt2_hx = t2 - t4
    # print(dt1_hx, dt2_hx)
    A_hx = 100
    # q_hx / (U_hx * lmtd(dt1_hx, dt2_hx))
    if t1 > 550:
        ft_hx = 1 + 0.02141 * (t1 - 550)
    elif t5 > 550:
        ft_hx = 1 + 0.02141 * (t5 - 550)
    else:
        ft_hx = 1
    cost_HX = 49.45 * U_hx * A_hx**0.7544 * ft_hx
    total_cost += cost_HX

    ##Heater
    p6 = p5 - 1e5
    q_heater = (
        935 * cp_gas * (630 - 110)
    )  # 630 from the exhaust 110 is just a number. It should be based on the dew point ass
    h6 = h5 + q_heater
    if t6 > 550:
        ft_heater = 1 + 5.4e-5 * (t6 - 550) ** 2
    elif t5 > 550:
        ft_heater = 1 + 5.4e-5 * (t5 - 550) ** 2
    else:
        ft_heater = 1
    cost_heater = 820800 * q_heater**0.7327 * ft_heater
    total_cost += cost_heater
    z = total_cost
    return z


bounds = [
    (35, 560),
    (35, 560),
    (35, 560),
    (35, 560),
    (35, 560),
    (35, 560),
    (1, 25),
    (1, 25),
    (0.1, 1e6),
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
w = 0.8  # inertia constant
c1 = 1  # cognative constant
c2 = 2  # social constant

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
