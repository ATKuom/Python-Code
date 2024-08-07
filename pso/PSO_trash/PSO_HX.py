import random
import matplotlib.pyplot as plt
import numpy as np
from functions import lmtd, enthalpy_entropy, h0, s0, T0, K


# ------------------------------------------------------------------------------
def objective_function(O):
    t1 = O[0]
    t2 = O[1]
    t4 = O[2]
    t5 = O[3]
    m = O[4]
    p1 = O[5]
    p2 = O[6]
    p4 = O[7]
    p5 = O[8]
    slack1 = O[9]
    slack2 = O[10]

    U_hx = 500

    (h1, s1) = enthalpy_entropy(t1, p1)
    (h2, s2) = enthalpy_entropy(t2, p2)
    (h4, s4) = enthalpy_entropy(t4, p4)
    (h5, s5) = enthalpy_entropy(t5, p5)

    e1 = m * (h1 - h0 - T0 * (s1 - s0))
    e2 = m * (h2 - h0 - T0 * (s2 - s0))
    e4 = m * (h4 - h0 - T0 * (s4 - s0))
    e5 = m * (h5 - h0 - T0 * (s5 - s0))

    fuel_HX = e1 - e2
    prod_HX = e5 - e4

    q_hx = h1 - h2
    h5 = h4 + q_hx
    dt1 = max(t1 - t5, 1)
    dt2 = max(t2 - t4, 1)

    A_hx = q_hx / (U_hx * lmtd(dt1, dt2))
    p2 = p1 - 1e6
    p5 = p4 - 1e6
    t1 = t5 + 5 + slack2
    t2 = t4 + 5 + slack1
    if t1 > 550:
        ft_hx = 1 + 0.02141 * (t1 - 550)
    elif t5 > 550:
        ft_hx = 1 + 0.02141 * (t5 - 550)
    else:
        ft_hx = 1
    cost_hx = 49.45 * U_hx * (A_hx**0.7544) * ft_hx
    cost_prod_hx = (fuel_HX + cost_hx) / prod_HX
    z = cost_hx + slack1 + slack2

    print(
        "%3.0f,%3.0f,%3.0f,%3.0f,%3.0f,%3.0f, %3.0f, %3.0f"
        % (t1, t2, t4, t5, p1 / 1e6, p2 / 1e6, p4 / 1e6, p5 / 1e6)
    )

    return z


bounds = [
    (35, 560),
    (35, 560),
    (35, 560),
    (35, 560),
    (0.1, 100),
    (74e6, 250e6),
    (74e6, 250e6),
    (74e6, 250e6),
    (74e6, 250e6),
    (0.1, 1e15),
    (0.1, 1e15),
]  # upper and lower bounds of variables
nv = len(bounds)  # number of variables
mm = -1  # if minimization mm, mm = -1; if maximization mm, mm = 1

# PARAMETERS OF PSO
particle_size = 14  # number of particles
iterations = 30  # max number of iterations
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
            # ax.set_xlim(left=max(0, i - iterations), right=i + 3)
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
